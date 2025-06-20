"""
フレーム関連ルート

フレームの取得、検索などのエンドポイント
"""

from flask import Blueprint, jsonify

from ..middleware.error_handler import error_handler
from ..middleware.request_logger import request_logger
from ..middleware.session_validator import validate_session
from ..schemas.response_schemas import FrameResponse
from ..services import FrameService
from .session_routes import get_session_service

# Blueprint定義
frame_bp = Blueprint("frames", __name__)

# サービスインスタンス
_frame_service = FrameService()


@frame_bp.route("/sessions/<session_id>/frames/<int:frame_number>", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def get_frame(session_id: str, frame_number: int):
    """
    指定フレームを取得

    Args:
        session_id: セッションID
        frame_number: フレーム番号

    Response:
        {
            "frame_number": 100,
            "timestamp": 3.33,
            "image": "data:image/jpeg;base64,...",
            "label": "game",
            "confidence": 0.95,
            "is_labeled": true,
            "metadata": {}
        }
    """
    from flask import g

    session = g.current_session
    frame_info = _frame_service.get_frame(session, frame_number)

    response = FrameResponse(
        frame_number=frame_info["frame_number"],
        timestamp=frame_info["timestamp"],
        image_data=frame_info["image"],
        label=frame_info["label"],
        confidence=frame_info["confidence"],
        is_labeled=frame_info["is_labeled"],
        metadata=frame_info.get("metadata", {}),
    )

    return jsonify(response.to_dict())


@frame_bp.route("/sessions/<session_id>/frames/next_unlabeled", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def get_next_unlabeled_frame(session_id: str):
    """
    次の未ラベルフレームを取得

    Args:
        session_id: セッションID

    Query Parameters:
        current: 現在のフレーム番号（オプション）

    Response:
        フレーム情報または404
    """
    from flask import g, request

    session = g.current_session
    current_frame = request.args.get("current", type=int)

    frame_info = _frame_service.get_next_unlabeled_frame(session, current_frame)

    if frame_info is None:
        from ..middleware.error_handler import NotFoundError

        raise NotFoundError("未ラベルフレーム")

    response = FrameResponse(
        frame_number=frame_info["frame_number"],
        timestamp=frame_info["timestamp"],
        image_data=frame_info["image"],
        label=frame_info["label"],
        confidence=frame_info["confidence"],
        is_labeled=frame_info["is_labeled"],
        metadata=frame_info.get("metadata", {}),
    )

    return jsonify(response.to_dict())


@frame_bp.route("/sessions/<session_id>/frames/uncertain", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def get_uncertainty_frame(session_id: str):
    """
    不確実性の高いフレームを取得

    Args:
        session_id: セッションID

    Query Parameters:
        threshold: 不確実性の閾値（デフォルト: 0.5）

    Response:
        フレーム情報または404
    """
    from flask import g, request

    session = g.current_session
    threshold = request.args.get("threshold", 0.5, type=float)

    frame_info = _frame_service.get_uncertainty_frame(session, threshold)

    if frame_info is None:
        from ..middleware.error_handler import NotFoundError

        raise NotFoundError("不確実性の高いフレーム")

    response = FrameResponse(
        frame_number=frame_info["frame_number"],
        timestamp=frame_info["timestamp"],
        image_data=frame_info["image"],
        label=frame_info["label"],
        confidence=frame_info["confidence"],
        is_labeled=frame_info["is_labeled"],
        metadata=frame_info.get("metadata", {}),
    )

    return jsonify(response.to_dict())


@frame_bp.route("/sessions/<session_id>/segments", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def get_frame_segments(session_id: str):
    """
    フレームセグメントを取得

    Args:
        session_id: セッションID

    Response:
        {
            "segments": [
                {
                    "label": "game",
                    "start_frame": 0,
                    "end_frame": 299,
                    "frame_count": 300
                },
                ...
            ]
        }
    """
    from flask import g

    session = g.current_session
    segments = _frame_service.get_frame_segments(session)

    return jsonify({"segments": segments})
