"""
対局画面ラベリングAPIルート
"""

import base64
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, jsonify, request

from .....utils.logger import LoggerMixin
from ...core.game_scene_classifier import GameSceneClassifier
from ..scene_labeling_session import SceneLabelingSession

# ブループリント定義
scene_labeling_bp = Blueprint("scene_labeling", __name__, url_prefix="/api/scene_labeling")

# グローバル変数でセッションを管理
_sessions: dict[str, SceneLabelingSession] = {}
_classifier = None


class SceneLabelingAPI(LoggerMixin):
    """対局画面ラベリングAPI"""

    @classmethod
    def get_classifier(cls) -> GameSceneClassifier:
        """分類器を取得（シングルトン）"""
        global _classifier
        if _classifier is None:
            # TODO: 学習済みモデルパスを設定から取得
            model_path = "models/game_scene_classifier.pth"
            if Path(model_path).exists():
                _classifier = GameSceneClassifier(model_path=model_path)
            else:
                _classifier = GameSceneClassifier()
        return _classifier


@scene_labeling_bp.route("/sessions", methods=["POST"])
def create_session():
    """ラベリングセッションを作成"""
    try:
        data = request.get_json()
        video_path = data.get("video_path")

        if not video_path or not Path(video_path).exists():
            return jsonify({"error": "動画ファイルが見つかりません"}), 400

        # セッション作成
        session_id = str(uuid.uuid4())
        classifier = SceneLabelingAPI.get_classifier()

        session = SceneLabelingSession(
            session_id=session_id, video_path=video_path, classifier=classifier
        )

        _sessions[session_id] = session

        # セッション情報を返す
        return jsonify(
            {
                "session_id": session_id,
                "video_info": {
                    "path": video_path,
                    "total_frames": session.total_frames,
                    "fps": session.fps,
                    "width": session.width,
                    "height": session.height,
                    "duration": session.total_frames / session.fps if session.fps > 0 else 0,
                },
                "statistics": session.get_statistics(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@scene_labeling_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """セッション情報を取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]

    return jsonify(
        {
            "session_id": session_id,
            "video_info": {
                "path": session.video_path,
                "total_frames": session.total_frames,
                "fps": session.fps,
                "width": session.width,
                "height": session.height,
            },
            "statistics": session.get_statistics(),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/frame/<int:frame_number>", methods=["GET"])
def get_frame(session_id: str, frame_number: int):
    """フレーム画像を取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    frame = session.get_frame(frame_number)

    if frame is None:
        return jsonify({"error": "フレームを取得できません"}), 404

    # 自動推論結果を取得
    auto_result = None
    if session.classifier:
        is_game, confidence = session.classifier.classify_frame(frame)
        auto_result = {"is_game_scene": is_game, "confidence": confidence}

    # 既存のラベルを確認
    existing_label = session.labels.get(frame_number)

    # 画像をBase64エンコード
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify(
        {
            "frame_number": frame_number,
            "image": f"data:image/jpeg;base64,{image_base64}",
            "label": {
                "is_game_scene": existing_label.is_game_scene,
                "confidence": existing_label.confidence,
                "annotator": existing_label.annotator,
            }
            if existing_label
            else None,
            "auto_result": auto_result,
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/label", methods=["POST"])
def label_frame(session_id: str):
    """フレームにラベルを付与"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    frame_number = data.get("frame_number")
    is_game_scene = data.get("is_game_scene")
    annotator = data.get("annotator", "manual")

    if frame_number is None or is_game_scene is None:
        return jsonify({"error": "frame_numberとis_game_sceneが必要です"}), 400

    session = _sessions[session_id]
    label = session.label_frame(frame_number, is_game_scene, annotator)

    return jsonify(
        {
            "success": True,
            "label": {
                "frame_number": label.frame_number,
                "is_game_scene": label.is_game_scene,
                "annotator": label.annotator,
                "created_at": label.created_at,
            },
            "statistics": session.get_statistics(),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/batch_label", methods=["POST"])
def batch_label_frames(session_id: str):
    """複数フレームに一括ラベル付与"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    start_frame = data.get("start_frame")
    end_frame = data.get("end_frame")
    is_game_scene = data.get("is_game_scene")
    annotator = data.get("annotator", "manual")

    if start_frame is None or end_frame is None or is_game_scene is None:
        return jsonify({"error": "必要なパラメータが不足しています"}), 400

    session = _sessions[session_id]
    labels = session.batch_label_frames(start_frame, end_frame, is_game_scene, annotator)

    return jsonify(
        {"success": True, "labeled_count": len(labels), "statistics": session.get_statistics()}
    )


@scene_labeling_bp.route("/sessions/<session_id>/next_unlabeled", methods=["GET"])
def get_next_unlabeled(session_id: str):
    """次の未ラベルフレームを取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    start_from = request.args.get("start_from", type=int)

    next_frame = session.get_next_unlabeled_frame(start_from)

    if next_frame is None:
        return jsonify({"next_frame": None, "message": "全てのフレームがラベル済みです"})

    return jsonify({"next_frame": next_frame})


@scene_labeling_bp.route("/sessions/<session_id>/uncertainty_frame", methods=["GET"])
def get_uncertainty_frame(session_id: str):
    """不確実性の高いフレームを取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    frame_number = session.get_uncertainty_frame()

    if frame_number is None:
        return jsonify({"frame_number": None, "message": "不確実なフレームはありません"})

    return jsonify({"frame_number": frame_number})


@scene_labeling_bp.route("/sessions/<session_id>/auto_label", methods=["POST"])
def auto_label(session_id: str):
    """自動ラベリング実行"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    data = request.get_json()
    frame_numbers = data.get("frame_numbers", [])
    sample_interval = data.get("sample_interval", 30)

    session = _sessions[session_id]

    if not frame_numbers:
        # 全フレームを対象にサンプリング
        frame_numbers = list(range(0, session.total_frames, sample_interval))

    # 自動ラベリング実行
    success_count = 0
    for frame_num in frame_numbers:
        if frame_num not in session.labels:  # 既存ラベルは上書きしない
            label = session.auto_label_frame(frame_num)
            if label:
                success_count += 1

    return jsonify(
        {
            "success": True,
            "labeled_count": success_count,
            "total_attempted": len(frame_numbers),
            "statistics": session.get_statistics(),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/segments", methods=["GET"])
def get_segments(session_id: str):
    """セグメント情報を取得"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    segments = session.export_segments()

    return jsonify(
        {
            "segments": segments,
            "total_segments": len(segments),
            "game_segments": sum(1 for s in segments if s["scene_type"] == "game"),
            "non_game_segments": sum(1 for s in segments if s["scene_type"] == "non_game"),
        }
    )


@scene_labeling_bp.route("/sessions/<session_id>/close", methods=["POST"])
def close_session(session_id: str):
    """セッションを終了"""
    if session_id not in _sessions:
        return jsonify({"error": "セッションが見つかりません"}), 404

    session = _sessions[session_id]
    statistics = session.get_statistics()

    # セグメントをエクスポート
    segments = session.export_segments()

    # セッションを閉じる
    session.close()
    del _sessions[session_id]

    return jsonify(
        {"success": True, "final_statistics": statistics, "exported_segments": len(segments)}
    )


@scene_labeling_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """アクティブなセッション一覧を取得"""
    sessions_info = []

    for session_id, session in _sessions.items():
        sessions_info.append(
            {
                "session_id": session_id,
                "video_path": session.video_path,
                "statistics": session.get_statistics(),
            }
        )

    return jsonify({"sessions": sessions_info, "total": len(sessions_info)})
