"""
自動ラベリング関連ルート

AI分類器を使用した自動ラベリングのエンドポイント
"""

from flask import Blueprint, jsonify, request

from ..middleware.error_handler import error_handler
from ..middleware.request_logger import request_logger
from ..middleware.session_validator import validate_session
from ..schemas.request_schemas import AutoLabelRequest
from ..schemas.response_schemas import SuccessResponse
from ..services import AutoLabelService
from .session_routes import get_session_service

# Blueprint定義
auto_label_bp = Blueprint("auto_label", __name__)

# サービスインスタンス（分類器は初期化時に設定）
_auto_label_service: AutoLabelService | None = None


def init_auto_label_service(classifier=None):
    """
    自動ラベリングサービスを初期化

    Args:
        classifier: GameSceneClassifier インスタンス
    """
    global _auto_label_service
    _auto_label_service = AutoLabelService(classifier)


def get_auto_label_service() -> AutoLabelService:
    """自動ラベリングサービスを取得"""
    if _auto_label_service is None:
        from ..middleware.error_handler import InternalError

        raise InternalError("自動ラベリングサービスが初期化されていません")
    return _auto_label_service


@auto_label_bp.route("/sessions/<session_id>/auto_label", methods=["POST"])
@error_handler
@request_logger
@validate_session(get_session_service)
def auto_label_frames(session_id: str):
    """
    フレームを自動的にラベル付け

    Args:
        session_id: セッションID

    Request:
        {
            "confidence_threshold": 0.8,
            "max_frames": 100,
            "skip_labeled": true
        }

    Response:
        {
            "success": true,
            "data": {
                "summary": {
                    "processed": 100,
                    "labeled": 85,
                    "skipped": 10,
                    "error": 5,
                    "success_rate": 0.85
                },
                "results": [
                    {
                        "frame_number": 100,
                        "label": "game",
                        "confidence": 0.95,
                        "labeled": true
                    },
                    ...
                ]
            }
        }
    """
    from flask import g

    from ..middleware.error_handler import ValidationError

    session = g.current_session
    data = request.get_json() or {}

    # リクエストの検証
    req = AutoLabelRequest(
        confidence_threshold=data.get("confidence_threshold", 0.8),
        max_frames=data.get("max_frames"),
        skip_labeled=data.get("skip_labeled", True),
    )

    errors = req.validate()
    if errors:
        raise ValidationError("リクエストが無効です", {"errors": errors})

    # 自動ラベリング実行
    service = get_auto_label_service()
    result = service.auto_label_frames(
        session,
        confidence_threshold=req.confidence_threshold,
        max_frames=req.max_frames,
        skip_labeled=req.skip_labeled,
    )

    response = SuccessResponse(
        message=f"{result['summary']['labeled']}個のフレームに自動でラベルを付けました",
        data=result,
    )
    return jsonify(response.to_dict())


@auto_label_bp.route("/sessions/<session_id>/predict/<int:frame_number>", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def predict_frame(session_id: str, frame_number: int):
    """
    単一フレームの予測

    Args:
        session_id: セッションID
        frame_number: フレーム番号

    Response:
        {
            "frame_number": 100,
            "prediction": {
                "label": "game",
                "confidence": 0.95,
                "probabilities": {
                    "game": 0.95,
                    "menu": 0.03,
                    "loading": 0.02
                }
            }
        }
    """
    from flask import g

    session = g.current_session

    # 予測実行
    service = get_auto_label_service()
    result = service.predict_frame(session, frame_number)

    return jsonify(result)


@auto_label_bp.route("/sessions/<session_id>/auto_label/status", methods=["GET"])
@error_handler
@request_logger
def get_auto_label_status(session_id: str):
    """
    自動ラベリングサービスの状態を取得

    Args:
        session_id: セッションID（使用されないが、RESTfulな一貫性のため）

    Response:
        {
            "available": true,
            "classifier_loaded": true,
            "model_info": {
                "name": "GameSceneClassifier",
                "version": "1.0.0"
            }
        }
    """
    try:
        service = get_auto_label_service()
        classifier_loaded = service._classifier is not None

        return jsonify(
            {
                "available": True,
                "classifier_loaded": classifier_loaded,
                "model_info": (
                    {
                        "name": "GameSceneClassifier",
                        "version": "1.0.0",  # 実際のバージョン情報を取得可能にする
                    }
                    if classifier_loaded
                    else None
                ),
            }
        )
    except Exception:
        return jsonify(
            {
                "available": False,
                "classifier_loaded": False,
                "model_info": None,
            }
        )
