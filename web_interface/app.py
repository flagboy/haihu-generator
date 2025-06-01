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
from src.training.learning.training_manager import TrainingConfig, TrainingManager
from src.training.semi_auto_labeler import SemiAutoLabeler
from src.utils.config import ConfigManager
from src.utils.logger import LoggerMixin

# Webアプリケーション設定
app = Flask(__name__)
app.config["SECRET_KEY"] = "mahjong-tile-detection-system-2024"
app.config["UPLOAD_FOLDER"] = "web_interface/uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

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


if __name__ == "__main__":
    # 開発サーバー起動
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
