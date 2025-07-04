"""
対局画面学習APIルート
"""

from pathlib import Path

from flask import Blueprint, jsonify, request
from flask_socketio import emit

from .....utils.logger import get_logger
from ..scene_dataset import SceneDataset
from ..scene_trainer import SceneTrainer

# ブループリント定義
scene_training_bp = Blueprint("scene_training", __name__, url_prefix="/api/scene_training")

# モジュールレベルのロガー
_logger = get_logger(__name__)

# グローバル変数
_trainer = None
_current_session = None


@scene_training_bp.route("/datasets", methods=["GET"])
def get_dataset_info():
    """データセット情報を取得"""
    try:
        # プロジェクトルートからの絶対パスを使用
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        db_path = project_root / "web_interface" / "data" / "training" / "game_scene_labels.db"

        dataset_info = {}

        for split in ["train", "val", "test"]:
            try:
                dataset = SceneDataset(db_path=str(db_path), split=split, transform=None)

                # データセットの統計情報を取得
                game_scenes = 0
                non_game_scenes = 0
                # video_ids = set()  # 未使用のため削除

                for i in range(len(dataset)):
                    try:
                        item = dataset[i]
                        if isinstance(item, tuple) and len(item) >= 2:
                            _, label = item
                            if label == 1:
                                game_scenes += 1
                            else:
                                non_game_scenes += 1
                        else:
                            _logger.error(f"予期しないデータ形式 (index={i}): {type(item)}, {item}")
                    except Exception as e:
                        _logger.error(f"データ読み込みエラー (index={i}): {type(e).__name__}: {e}")
                        import traceback

                        _logger.error(f"トレースバック:\n{traceback.format_exc()}")

                # 統計情報からvideo数を取得
                stats = dataset.get_statistics()
                video_count = stats.get("videos", 0)

                dataset_info[split] = {
                    "total_samples": len(dataset),
                    "game_scenes": game_scenes,
                    "non_game_scenes": non_game_scenes,
                    "videos": video_count,
                }

            except Exception as e:
                _logger.error(f"{split}データセットのエラー: {e}")
                dataset_info[split] = {
                    "error": str(e),
                    "total_samples": 0,
                    "game_scenes": 0,
                    "non_game_scenes": 0,
                    "videos": 0,
                }

        return jsonify(dataset_info)

    except Exception as e:
        _logger.error(f"データセット情報取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@scene_training_bp.route("/prepare", methods=["POST"])
def prepare_training():
    """学習の準備状態を確認"""
    try:
        # プロジェクトルートからの絶対パスを使用
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        db_path = project_root / "web_interface" / "data" / "training" / "game_scene_labels.db"

        # 各分割のデータ数を確認
        has_train = False
        has_val = False

        try:
            train_dataset = SceneDataset(str(db_path), split="train")
            has_train = len(train_dataset) > 0
        except Exception:
            pass

        try:
            val_dataset = SceneDataset(str(db_path), split="val")
            has_val = len(val_dataset) > 0
        except Exception:
            pass

        ready = has_train and has_val

        return jsonify({"ready": ready, "has_train": has_train, "has_val": has_val})

    except Exception as e:
        _logger.error(f"準備確認エラー: {e}")
        return jsonify({"error": str(e)}), 500


@scene_training_bp.route("/start", methods=["POST"])
def start_training():
    """学習を開始"""
    global _trainer, _current_session

    try:
        data = request.get_json()
        epochs = data.get("epochs", 10)
        batch_size = data.get("batch_size", 32)
        learning_rate = data.get("learning_rate", 0.001)

        # プロジェクトルートからの絶対パスを使用
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        db_path = project_root / "web_interface" / "data" / "training" / "game_scene_labels.db"
        model_save_path = project_root / "web_interface" / "models" / "game_scene"

        # トレーナーを作成
        _trainer = SceneTrainer(output_dir=str(model_save_path))

        # セッションIDを生成
        import uuid

        session_id = str(uuid.uuid4())
        _current_session = session_id

        # 非同期で学習を開始
        import threading

        def train_thread():
            try:
                # WebSocket進捗通知用のコールバック
                def progress_callback(
                    message, current_epoch=0, total_epochs=epochs, train_loss=0.0, val_acc=0.0
                ):
                    try:
                        emit(
                            "scene_training_progress",
                            {
                                "session_id": session_id,
                                "message": message,
                                "current_epoch": current_epoch,
                                "total_epochs": total_epochs,
                                "train_loss": train_loss,
                                "val_accuracy": val_acc,
                                "progress": current_epoch / total_epochs if total_epochs > 0 else 0,
                            },
                            namespace="/",
                        )
                    except Exception as e:
                        _logger.warning(f"WebSocket通知エラー: {e}")

                # 開始通知
                progress_callback("学習を開始します")

                # データセットを準備
                train_dataset = SceneDataset(str(db_path), split="train")
                val_dataset = SceneDataset(str(db_path), split="val")

                progress_callback(
                    f"データセット準備完了: 学習{len(train_dataset)}件, 検証{len(val_dataset)}件"
                )

                # トレーナーに進捗コールバックを設定
                _trainer.set_progress_callback(progress_callback)

                # 学習実行
                results = _trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

                # 完了通知
                progress_callback(
                    f"学習が完了しました - 最終精度: {results.get('final_val_acc', 0):.2f}",
                    current_epoch=epochs,
                    total_epochs=epochs,
                )

                _logger.info("学習が完了しました")

            except Exception as e:
                _logger.error(f"学習エラー: {e}")
                # エラー通知
                emit(
                    "scene_training_error",
                    {
                        "session_id": session_id,
                        "error": str(e),
                        "message": f"学習に失敗しました: {e}",
                    },
                    namespace="/",
                )

        thread = threading.Thread(target=train_thread)
        thread.start()

        return jsonify({"session_id": session_id, "status": "started"})

    except Exception as e:
        _logger.error(f"学習開始エラー: {e}")
        return jsonify({"error": str(e)}), 500


@scene_training_bp.route("/status/<session_id>", methods=["GET"])
def get_training_status(session_id: str):
    """学習の状態を取得"""
    global _trainer, _current_session

    if _current_session != session_id:
        return jsonify({"error": "Invalid session ID"}), 404

    if _trainer is None:
        return jsonify({"error": "No training in progress"}), 404

    try:
        # トレーナーから進捗情報を取得
        status = _trainer.get_training_status()

        # セッションIDが一致することを確認
        if status.get("session_id") != _current_session:
            return jsonify({"error": "Session mismatch"}), 500

        # レスポンス形式を整形
        progress = {
            "session_id": session_id,
            "status": "training" if status["is_training"] else "completed",
            "current_epoch": status["current_epoch"],
            "total_epochs": status["total_epochs"],
            "train_loss": status["train_loss"],
            "val_accuracy": status["val_accuracy"],
            "progress": status["progress"],
        }

        return jsonify(progress)

    except Exception as e:
        _logger.error(f"状態取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@scene_training_bp.route("/stop/<session_id>", methods=["POST"])
def stop_training(session_id: str):
    """学習を停止"""
    global _trainer, _current_session

    if _current_session != session_id:
        return jsonify({"error": "Invalid session ID"}), 404

    if _trainer is None:
        return jsonify({"error": "No training in progress"}), 404

    try:
        # トレーナーに停止を要求
        _trainer.stop_training()

        return jsonify({"success": True, "message": "学習停止を要求しました"})

    except Exception as e:
        _logger.error(f"停止エラー: {e}")
        return jsonify({"error": str(e)}), 500
