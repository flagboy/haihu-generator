"""
麻雀牌検出システム - Webインターフェース
メインアプリケーション
"""

import os

# 既存システムのインポート
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataset_manager import DatasetManager
from src.training.frame_extractor import FrameExtractor
from src.training.game_scene.labeling.api.scene_routes import scene_labeling_bp
from src.training.game_scene.utils.frame_skip_manager import FrameSkipManager
from src.training.labeling.api import websocket as labeling_websocket
from src.training.labeling.api.routes import labeling_bp

# 統合された手牌ラベリングシステムのインポート
from src.training.labeling.core.hand_area_detector import (
    UnifiedHandAreaDetector as HandAreaDetector,
)
from src.training.labeling.core.tile_splitter import TileSplitter
from src.training.labeling.core.video_processor import EnhancedVideoProcessor as HandFrameExtractor
from src.training.learning.training_manager import TrainingConfig, TrainingManager
from src.training.semi_auto_labeler import SemiAutoLabeler
from src.utils.config import ConfigManager
from src.utils.logger import LoggerMixin

# Webアプリケーション設定
app = Flask(__name__)
app.config["SECRET_KEY"] = "mahjong-tile-detection-system-2024"
app.config["UPLOAD_FOLDER"] = "web_interface/uploads"
# app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB - 制限なしに変更
app.config["MAX_CONTENT_LENGTH"] = None  # ファイルサイズ制限なし

# WebSocket設定
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# アップロードディレクトリ作成
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


class WebInterfaceManager(LoggerMixin):
    """Webインターフェース管理クラス"""

    def __init__(self):
        """初期化"""
        self.config_manager = ConfigManager()
        self.dataset_manager = DatasetManager(self.config_manager)
        self.training_manager = TrainingManager(self.config_manager)
        self.frame_extractor = FrameExtractor(self.config_manager)
        self.semi_auto_labeler = SemiAutoLabeler(self.config_manager)

        # セッション管理
        self.active_sessions: dict[str, dict] = {}
        self.training_sessions: dict[str, str] = {}  # session_id -> training_session_id

        # 手牌ラベリング用コンポーネント
        self.hand_area_detector = HandAreaDetector()
        self.tile_splitter = TileSplitter()
        self.hand_frame_extractors: dict[str, HandFrameExtractor] = {}  # video_id -> extractor
        self.labeling_sessions: dict[str, dict] = {}  # session_id -> labeling data

        # フレームスキップマネージャー
        self.frame_skip_manager = FrameSkipManager()

        self.logger.info("WebInterfaceManager初期化完了")

    def get_dataset_statistics(self) -> dict[str, Any]:
        """データセット統計情報を取得"""
        return self.dataset_manager.get_dataset_statistics()

    def get_training_sessions(self) -> list[dict[str, Any]]:
        """学習セッション一覧を取得"""
        return self.training_manager.list_sessions()

    def get_dataset_versions(self) -> list[dict[str, Any]]:
        """データセットバージョン一覧を取得"""
        return self.dataset_manager.list_versions()


# グローバルマネージャーインスタンス
web_manager = WebInterfaceManager()


@app.route("/")
def index():
    """メインページ"""
    return render_template("index.html")


@app.route("/labeling")
def labeling():
    """ラベリングページ"""
    return render_template("labeling.html")


@app.route("/training")
def training():
    """学習管理ページ"""
    return render_template("training.html")


@app.route("/data_management")
def data_management():
    """データ管理ページ"""
    return render_template("data_management.html")


@app.route("/scene_labeling")
def scene_labeling():
    """対局画面ラベリングページ"""
    return render_template("scene_labeling.html")


# API エンドポイント


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """動画アップロード"""
    try:
        if "video" not in request.files:
            return jsonify({"error": "動画ファイルが選択されていません"}), 400

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "ファイル名が空です"}), 400

        if file:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 動画情報を取得
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            video_info = {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "filepath": filepath,
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "upload_time": datetime.now().isoformat(),
            }

            return jsonify({"success": True, "video_info": video_info})

    except Exception as e:
        web_manager.logger.error(f"動画アップロードエラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/extract_frames", methods=["POST"])
def extract_frames():
    """フレーム抽出"""
    try:
        data = request.get_json()
        video_path = data.get("video_path")
        extract_config = data.get("config", {})

        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "動画ファイルが見つかりません"}), 400

        # フレーム抽出を非同期で実行
        session_id = str(uuid.uuid4())
        web_manager.active_sessions[session_id] = {
            "type": "frame_extraction",
            "status": "running",
            "start_time": datetime.now(),
            "video_path": video_path,
            "config": extract_config,
        }

        # WebSocketで進捗を通知しながらフレーム抽出
        socketio.start_background_task(
            target=extract_frames_background,
            session_id=session_id,
            video_path=video_path,
            config=extract_config,
        )

        return jsonify({"success": True, "session_id": session_id})

    except Exception as e:
        web_manager.logger.error(f"フレーム抽出エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/statistics")
def get_dataset_statistics():
    """データセット統計情報API"""
    try:
        stats = web_manager.get_dataset_statistics()
        return jsonify(stats)
    except Exception as e:
        web_manager.logger.error(f"統計情報取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/versions")
def get_dataset_versions():
    """データセットバージョン一覧API"""
    try:
        versions = web_manager.get_dataset_versions()
        return jsonify(versions)
    except Exception as e:
        web_manager.logger.error(f"バージョン一覧取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/sessions")
def get_training_sessions():
    """学習セッション一覧API"""
    try:
        sessions = web_manager.get_training_sessions()
        return jsonify(sessions)
    except Exception as e:
        web_manager.logger.error(f"学習セッション取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """学習開始API"""
    try:
        data = request.get_json()

        # TrainingConfigを作成
        config = TrainingConfig(
            model_type=data.get("model_type", "detection"),
            model_name=data.get("model_name", "default"),
            dataset_version_id=data.get("dataset_version_id"),
            epochs=data.get("epochs", 100),
            batch_size=data.get("batch_size", 32),
            learning_rate=data.get("learning_rate", 0.001),
            validation_split=data.get("validation_split", 0.2),
            test_split=data.get("test_split", 0.1),
            early_stopping_patience=data.get("early_stopping_patience", 10),
            save_best_only=data.get("save_best_only", True),
            use_data_augmentation=data.get("use_data_augmentation", True),
            transfer_learning=data.get("transfer_learning", False),
            pretrained_model_path=data.get("pretrained_model_path"),
            gpu_enabled=data.get("gpu_enabled", True),
            num_workers=data.get("num_workers", 4),
            seed=data.get("seed", 42),
        )

        # 学習を非同期で開始
        session_id = str(uuid.uuid4())
        socketio.start_background_task(
            target=start_training_background, session_id=session_id, config=config
        )

        return jsonify({"success": True, "session_id": session_id})

    except Exception as e:
        web_manager.logger.error(f"学習開始エラー: {e}")
        return jsonify({"error": str(e)}), 500


# 手牌ラベリングAPI


@app.route("/api/labeling/hand_areas", methods=["GET"])
def get_hand_areas():
    """手牌領域設定を取得"""
    try:
        # 現在の設定を取得
        regions = web_manager.hand_area_detector.regions
        frame_size = web_manager.hand_area_detector.frame_size

        return jsonify({"regions": regions, "frame_size": frame_size})
    except Exception as e:
        web_manager.logger.error(f"手牌領域取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/hand_areas", methods=["POST"])
def set_hand_areas():
    """手牌領域を設定"""
    try:
        data = request.get_json()
        frame_size = data.get("frame_size")
        regions = data.get("regions")

        if frame_size:
            web_manager.hand_area_detector.set_frame_size(frame_size[0], frame_size[1])

        if regions:
            for player, region in regions.items():
                web_manager.hand_area_detector.set_region(
                    player, region["x"], region["y"], region["w"], region["h"]
                )

        return jsonify({"success": True})
    except Exception as e:
        web_manager.logger.error(f"手牌領域設定エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/split_tiles", methods=["POST"])
def split_tiles():
    """手牌領域から牌を分割"""
    try:
        data = request.get_json()
        video_id = data.get("video_id")
        frame_number = data.get("frame_number")
        player = data.get("player", "bottom")

        if not video_id or frame_number is None:
            return jsonify({"error": "video_idとframe_numberが必要です"}), 400

        # フレーム抽出器を取得
        if video_id not in web_manager.hand_frame_extractors:
            return jsonify({"error": "動画が読み込まれていません"}), 400

        extractor = web_manager.hand_frame_extractors[video_id]

        # フレームを取得
        frame = extractor.extract_frame(frame_number)
        if frame is None:
            return jsonify({"error": "フレームを取得できません"}), 400

        # 手牌領域を抽出
        hand_region = web_manager.hand_area_detector.extract_hand_region(frame, player)
        if hand_region is None:
            return jsonify({"error": "手牌領域を抽出できません"}), 400

        # 牌を分割
        tile_bboxes = web_manager.tile_splitter.split_hand_auto(hand_region)

        # 結果を返す
        result = {"player": player, "frame_number": frame_number, "tiles": []}

        for i, (x, y, w, h) in enumerate(tile_bboxes):
            result["tiles"].append(
                {
                    "index": i,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "label": None,
                    "confidence": None,
                }
            )

        return jsonify(result)
    except Exception as e:
        web_manager.logger.error(f"牌分割エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/auto_label", methods=["POST"])
def auto_label_tiles():
    """AIを使用して牌を自動ラベリング"""
    try:
        data = request.get_json()
        video_id = data.get("video_id")
        frame_number = data.get("frame_number")
        player = data.get("player", "bottom")

        if not video_id or frame_number is None:
            return jsonify({"error": "video_idとframe_numberが必要です"}), 400

        # フレーム抽出器を取得
        if video_id not in web_manager.hand_frame_extractors:
            return jsonify({"error": "動画が読み込まれていません"}), 400

        extractor = web_manager.hand_frame_extractors[video_id]

        # フレームを取得
        frame = extractor.extract_frame(frame_number)
        if frame is None:
            return jsonify({"error": "フレームを取得できません"}), 400

        # 手牌領域を抽出
        hand_region = web_manager.hand_area_detector.extract_hand_region(frame, player)
        if hand_region is None:
            return jsonify({"error": "手牌領域を抽出できません"}), 400

        # 牌を分割
        tile_images = web_manager.tile_splitter.split_hand_auto(hand_region)

        # 各牌を分類
        result = {"player": player, "frame_number": frame_number, "tiles": []}

        for i, tile_img in enumerate(tile_images):
            # 牌を分類（semi_auto_labelerを使用）
            enhanced_tile = web_manager.tile_splitter.enhance_tile_image(tile_img)
            classification_result = web_manager.semi_auto_labeler.tile_classifier.classify_tile(
                enhanced_tile
            )

            result["tiles"].append(
                {
                    "index": i,
                    "label": classification_result.tile_name,
                    "confidence": classification_result.confidence,
                }
            )

        return jsonify(result)
    except Exception as e:
        web_manager.logger.error(f"自動ラベリングエラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/save_annotations", methods=["POST"])
def save_labeling_annotations():
    """ラベリング結果を保存"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        annotations = data.get("annotations")

        if not session_id or not annotations:
            return jsonify({"error": "session_idとannotationsが必要です"}), 400

        # セッションデータに保存
        if session_id not in web_manager.labeling_sessions:
            web_manager.labeling_sessions[session_id] = {
                "created_at": datetime.now(),
                "annotations": [],
            }

        web_manager.labeling_sessions[session_id]["annotations"].extend(annotations)
        web_manager.labeling_sessions[session_id]["updated_at"] = datetime.now()

        # データベースに保存（dataset_managerを使用）
        for annotation in annotations:
            web_manager.dataset_manager.add_frame_annotation(
                video_id=annotation["video_id"],
                frame_number=annotation["frame_number"],
                tiles=annotation["tiles"],
                annotator="web_interface",
                metadata={"session_id": session_id, "player": annotation.get("player", "bottom")},
            )

        return jsonify({"success": True, "saved_count": len(annotations)})
    except Exception as e:
        web_manager.logger.error(f"アノテーション保存エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/load_video", methods=["POST"])
def load_video_for_labeling():
    """ラベリング用に動画を読み込み"""
    try:
        data = request.get_json()
        video_path = data.get("video_path")
        video_id = data.get("video_id", str(uuid.uuid4()))

        if not video_path or not os.path.exists(video_path):
            return jsonify({"error": "動画ファイルが見つかりません"}), 400

        # フレーム抽出器を作成（フレームスキップマネージャーを渡す）
        output_dir = f"data/hand_training/frames/{video_id}"
        extractor = HandFrameExtractor(
            video_path, output_dir, frame_skip_manager=web_manager.frame_skip_manager
        )
        web_manager.hand_frame_extractors[video_id] = extractor

        # 動画情報を返す
        return jsonify(
            {
                "video_id": video_id,
                "fps": extractor.fps,
                "frame_count": extractor.frame_count,
                "width": extractor.width,
                "height": extractor.height,
                "duration": extractor.frame_count / extractor.fps if extractor.fps > 0 else 0,
            }
        )
    except Exception as e:
        web_manager.logger.error(f"動画読み込みエラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/labeling/export", methods=["POST"])
def export_labeling_data():
    """ラベリングデータをエクスポート"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        format_type = data.get("format", "json")  # json, coco, yolo

        if session_id not in web_manager.labeling_sessions:
            return jsonify({"error": "セッションが見つかりません"}), 400

        annotations = web_manager.labeling_sessions[session_id]["annotations"]

        if format_type == "json":
            # シンプルなJSON形式
            return jsonify(
                {
                    "session_id": session_id,
                    "annotations": annotations,
                    "export_time": datetime.now().isoformat(),
                }
            )
        elif format_type == "coco":
            # COCO形式に変換
            # TODO: COCO形式への変換実装
            return jsonify({"error": "COCO形式は未実装です"}), 501
        elif format_type == "yolo":
            # YOLO形式に変換
            # TODO: YOLO形式への変換実装
            return jsonify({"error": "YOLO形式は未実装です"}), 501
        else:
            return jsonify({"error": "不明なフォーマット"}), 400

    except Exception as e:
        web_manager.logger.error(f"エクスポートエラー: {e}")
        return jsonify({"error": str(e)}), 500


# WebSocket イベントハンドラー


@socketio.on("connect")
def handle_connect():
    """WebSocket接続"""
    web_manager.logger.info(f"WebSocket接続: {request.sid}")
    emit("connected", {"message": "接続しました"})


@socketio.on("disconnect")
def handle_disconnect():
    """WebSocket切断"""
    web_manager.logger.info(f"WebSocket切断: {request.sid}")


@socketio.on("join_session")
def handle_join_session(data):
    """セッションルームに参加"""
    session_id = data.get("session_id")
    if session_id:
        join_room(session_id)
        emit("joined_session", {"session_id": session_id})


@socketio.on("leave_session")
def handle_leave_session(data):
    """セッションルームから退出"""
    session_id = data.get("session_id")
    if session_id:
        leave_room(session_id)
        emit("left_session", {"session_id": session_id})


# バックグラウンドタスク


def extract_frames_background(session_id: str, video_path: str, config: dict):
    """フレーム抽出バックグラウンドタスク"""
    try:
        session_info = web_manager.active_sessions[session_id]

        # 進捗通知
        socketio.emit(
            "frame_extraction_progress",
            {"session_id": session_id, "status": "starting", "message": "フレーム抽出を開始します"},
            room=session_id,
        )

        # フレーム抽出実行
        result = web_manager.frame_extractor.extract_frames(
            video_path=video_path,
            output_dir=config.get("output_dir", "data/frames"),
            interval_seconds=config.get("interval_seconds", 1.0),
            quality_threshold=config.get("quality_threshold", 0.5),
        )

        session_info["status"] = "completed"
        session_info["result"] = result
        session_info["end_time"] = datetime.now()

        socketio.emit(
            "frame_extraction_progress",
            {
                "session_id": session_id,
                "status": "completed",
                "result": result,
                "message": "フレーム抽出が完了しました",
            },
            room=session_id,
        )

    except Exception as e:
        session_info = web_manager.active_sessions.get(session_id, {})
        session_info["status"] = "failed"
        session_info["error"] = str(e)
        session_info["end_time"] = datetime.now()

        socketio.emit(
            "frame_extraction_progress",
            {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "message": f"フレーム抽出に失敗しました: {e}",
            },
            room=session_id,
        )


def start_training_background(session_id: str, config: TrainingConfig):
    """学習バックグラウンドタスク"""
    try:
        # 進捗通知
        socketio.emit(
            "training_progress",
            {"session_id": session_id, "status": "starting", "message": "学習を開始します"},
            room=session_id,
        )

        # 学習実行
        training_session_id = web_manager.training_manager.start_training(config)
        web_manager.training_sessions[session_id] = training_session_id

        # 学習進捗を定期的に通知
        while True:
            status = web_manager.training_manager.get_session_status(training_session_id)
            if not status:
                break

            socketio.emit(
                "training_progress",
                {
                    "session_id": session_id,
                    "training_session_id": training_session_id,
                    "status": status["status"],
                    "progress": status.get("current_progress"),
                    "metrics": status.get("final_metrics"),
                    "message": f"学習進捗: {status['status']}",
                },
                room=session_id,
            )

            if status["status"] in ["completed", "failed", "stopped"]:
                break

            socketio.sleep(5)  # 5秒間隔で更新

    except Exception as e:
        socketio.emit(
            "training_progress",
            {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "message": f"学習に失敗しました: {e}",
            },
            room=session_id,
        )


# 対局画面学習API
@app.route("/api/scene_training/prepare", methods=["POST"])
def prepare_scene_training():
    """対局画面学習データの準備"""
    try:
        from src.training.game_scene.learning.scene_dataset import SceneDataset

        # データセットを作成
        train_dataset = SceneDataset(split="train")
        val_dataset = SceneDataset(split="val")
        test_dataset = SceneDataset(split="test")

        # 統計情報を取得
        stats = {
            "train": train_dataset.get_statistics(),
            "val": val_dataset.get_statistics(),
            "test": test_dataset.get_statistics(),
        }

        return jsonify(
            {
                "success": True,
                "statistics": stats,
                "ready": all(s["total_samples"] > 0 for s in stats.values()),
            }
        )
    except Exception as e:
        web_manager.logger.error(f"学習データ準備エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scene_training/start", methods=["POST"])
def start_scene_training():
    """対局画面分類モデルの学習開始"""
    try:
        from src.training.game_scene.learning.scene_dataset import SceneDataset
        from src.training.game_scene.learning.scene_trainer import SceneTrainer

        data = request.get_json()

        # パラメータ取得
        epochs = data.get("epochs", 50)
        batch_size = data.get("batch_size", 32)
        learning_rate = data.get("learning_rate", 0.001)

        # データセット準備
        train_dataset = SceneDataset(split="train")
        val_dataset = SceneDataset(split="val")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return jsonify({"error": "学習データが不足しています"}), 400

        # トレーナー初期化
        trainer = SceneTrainer()

        # 学習を非同期で開始
        session_id = str(uuid.uuid4())

        def train_background():
            try:
                results = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

                # 結果をWebSocketで通知
                socketio.emit(
                    "scene_training_complete",
                    {"session_id": session_id, "results": results},
                    room=session_id,
                )
            except Exception as e:
                socketio.emit(
                    "scene_training_error",
                    {"session_id": session_id, "error": str(e)},
                    room=session_id,
                )

        socketio.start_background_task(train_background)

        return jsonify({"success": True, "session_id": session_id, "message": "学習を開始しました"})

    except Exception as e:
        web_manager.logger.error(f"学習開始エラー: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/scene_training/datasets")
def get_scene_datasets():
    """対局画面データセットの情報を取得"""
    try:
        from src.training.game_scene.learning.scene_dataset import SceneDataset

        datasets = {}
        for split in ["train", "val", "test"]:
            try:
                dataset = SceneDataset(split=split)
                datasets[split] = dataset.get_statistics()
            except Exception as e:
                datasets[split] = {"error": str(e), "total_samples": 0}

        return jsonify(datasets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# APIブループリントを登録
app.register_blueprint(labeling_bp)
app.register_blueprint(scene_labeling_bp)

# WebSocketを初期化
labeling_websocket.init_socketio(app)

if __name__ == "__main__":
    # 開発サーバー起動
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
