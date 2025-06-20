"""
セッション管理ルート

セッションの作成、取得、削除などのエンドポイント
"""

from flask import Blueprint, jsonify, request

from ..middleware.error_handler import error_handler
from ..middleware.request_logger import request_logger
from ..schemas.request_schemas import CreateSessionRequest
from ..schemas.response_schemas import SessionListResponse, SessionResponse, SuccessResponse
from ..services import SessionService

# Blueprint定義
session_bp = Blueprint("sessions", __name__)

# サービスインスタンス（シングルトン）
_session_service = SessionService()


def get_session_service() -> SessionService:
    """セッションサービスを取得"""
    return _session_service


@session_bp.route("/sessions", methods=["POST"])
@error_handler
@request_logger
def create_session():
    """
    セッションを作成または再開

    Request:
        {
            "video_path": "path/to/video.mp4",
            "session_id": "existing-session-id" (optional),
            "metadata": {} (optional)
        }

    Response:
        {
            "session_id": "session-id",
            "video_path": "path/to/video.mp4",
            "total_frames": 1000,
            "labeled_frames": 0,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "status": "active",
            "metadata": {}
        }
    """
    data = request.get_json()

    # リクエストスキーマの作成と検証
    req = CreateSessionRequest(
        video_path=data.get("video_path", ""),
        session_id=data.get("session_id"),
        metadata=data.get("metadata", {}),
    )

    errors = req.validate()
    if errors:
        from ..middleware.error_handler import ValidationError

        raise ValidationError("リクエストが無効です", {"errors": errors})

    # セッション作成
    session_info = _session_service.create_session(
        video_path=req.video_path, session_id=req.session_id, metadata=req.metadata
    )

    # レスポンス作成
    response = SessionResponse(
        session_id=session_info["session_id"],
        video_path=session_info["video_path"],
        total_frames=session_info["total_frames"],
        labeled_frames=session_info["labeled_frames"],
        created_at=session_info["created_at"],
        updated_at=session_info["updated_at"],
        status=session_info["status"],
        metadata=session_info.get("metadata", {}),
    )

    return jsonify(response.to_dict()), 201


@session_bp.route("/sessions", methods=["GET"])
@error_handler
@request_logger
def list_sessions():
    """
    セッション一覧を取得

    Response:
        {
            "sessions": [...],
            "total": 10
        }
    """
    sessions_info = _session_service.list_sessions()

    # レスポンス作成
    sessions = [
        SessionResponse(
            session_id=info["session_id"],
            video_path=info["video_path"],
            total_frames=info["total_frames"],
            labeled_frames=info["labeled_frames"],
            created_at=info["created_at"],
            updated_at=info["updated_at"],
            status=info["status"],
            metadata=info.get("metadata", {}),
        )
        for info in sessions_info
    ]

    response = SessionListResponse(sessions=sessions, total=len(sessions))
    return jsonify(response.to_dict())


@session_bp.route("/sessions/<session_id>", methods=["GET"])
@error_handler
@request_logger
def get_session(session_id: str):
    """
    セッション情報を取得

    Args:
        session_id: セッションID

    Response:
        セッション情報
    """
    session_info = _session_service.get_session_info(session_id)

    response = SessionResponse(
        session_id=session_info["session_id"],
        video_path=session_info["video_path"],
        total_frames=session_info["total_frames"],
        labeled_frames=session_info["labeled_frames"],
        created_at=session_info["created_at"],
        updated_at=session_info["updated_at"],
        status=session_info["status"],
        metadata=session_info.get("metadata", {}),
    )

    return jsonify(response.to_dict())


@session_bp.route("/sessions/<session_id>", methods=["DELETE"])
@error_handler
@request_logger
def delete_session(session_id: str):
    """
    セッションを削除

    Args:
        session_id: セッションID

    Response:
        {"success": true}
    """
    _session_service.delete_session(session_id)
    response = SuccessResponse(message=f"セッション {session_id} を削除しました")
    return jsonify(response.to_dict())


@session_bp.route("/sessions/<session_id>/close", methods=["POST"])
@error_handler
@request_logger
def close_session(session_id: str):
    """
    セッションを終了

    Args:
        session_id: セッションID

    Response:
        {"success": true}
    """
    _session_service.close_session(session_id)
    response = SuccessResponse(message=f"セッション {session_id} を終了しました")
    return jsonify(response.to_dict())


@session_bp.route("/sessions/clear", methods=["POST"])
@error_handler
@request_logger
def clear_sessions():
    """
    全セッションをクリア

    Response:
        {"success": true, "data": {"cleared_count": 5}}
    """
    count = _session_service.clear_all_sessions()
    response = SuccessResponse(
        message=f"{count}個のセッションをクリアしました", data={"cleared_count": count}
    )
    return jsonify(response.to_dict())
