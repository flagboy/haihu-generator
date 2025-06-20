"""
ラベリング関連ルート

ラベル付け、統計取得などのエンドポイント
"""

from flask import Blueprint, jsonify, request

from ..middleware.error_handler import error_handler
from ..middleware.request_logger import request_logger
from ..middleware.session_validator import validate_session
from ..schemas.request_schemas import BatchLabelRequest, LabelRequest
from ..schemas.response_schemas import SuccessResponse
from ..services import LabelingService
from .session_routes import get_session_service

# Blueprint定義
labeling_bp = Blueprint("labeling", __name__)

# サービスインスタンス
_labeling_service = LabelingService()


@labeling_bp.route("/sessions/<session_id>/label", methods=["POST"])
@error_handler
@request_logger
@validate_session(get_session_service)
def label_frame(session_id: str):
    """
    フレームにラベルを付ける

    Args:
        session_id: セッションID

    Request:
        {
            "frame_number": 100,
            "label": "game",
            "confidence": 0.95,
            "metadata": {}
        }

    Response:
        {
            "success": true,
            "data": {
                "frame_number": 100,
                "label": "game",
                "confidence": 0.95
            }
        }
    """
    from flask import g

    from ..middleware.error_handler import ValidationError

    session = g.current_session
    data = request.get_json()

    # リクエストの検証
    req = LabelRequest(
        frame_number=data.get("frame_number", -1),
        label=data.get("label", ""),
        confidence=data.get("confidence"),
        metadata=data.get("metadata", {}),
    )

    errors = req.validate()
    if errors:
        raise ValidationError("リクエストが無効です", {"errors": errors})

    # ラベル付け実行
    result = _labeling_service.label_frame(
        session,
        frame_number=req.frame_number,
        label=req.label,
        confidence=req.confidence,
        metadata=req.metadata,
    )

    response = SuccessResponse(
        message=f"フレーム {req.frame_number} にラベルを付けました", data=result
    )
    return jsonify(response.to_dict())


@labeling_bp.route("/sessions/<session_id>/batch_label", methods=["POST"])
@error_handler
@request_logger
@validate_session(get_session_service)
def batch_label_frames(session_id: str):
    """
    複数フレームに一括でラベルを付ける

    Args:
        session_id: セッションID

    Request:
        {
            "labels": [
                {
                    "frame_number": 100,
                    "label": "game",
                    "confidence": 0.95,
                    "metadata": {}
                },
                ...
            ]
        }

    Response:
        {
            "success": true,
            "data": {
                "results": [...],
                "summary": {
                    "total": 10,
                    "success": 9,
                    "error": 1
                }
            }
        }
    """
    from flask import g

    from ..middleware.error_handler import ValidationError

    session = g.current_session
    data = request.get_json()

    # リクエストの検証
    labels_data = data.get("labels", [])
    if not labels_data:
        raise ValidationError("ラベルデータが必要です")

    label_requests = []
    for label_data in labels_data:
        req = LabelRequest(
            frame_number=label_data.get("frame_number", -1),
            label=label_data.get("label", ""),
            confidence=label_data.get("confidence"),
            metadata=label_data.get("metadata", {}),
        )
        errors = req.validate()
        if errors:
            raise ValidationError(
                f"ラベルデータが無効です (フレーム {label_data.get('frame_number')})",
                {"errors": errors},
            )
        label_requests.append(req)

    batch_req = BatchLabelRequest(labels=label_requests)

    # バッチラベル付け実行
    result = _labeling_service.batch_label_frames(session, batch_req)

    message = (
        f"{result['summary']['success']}個のフレームにラベルを付けました"
        if result["success"]
        else f"{result['summary']['error']}個のエラーが発生しました"
    )

    response = SuccessResponse(message=message, data=result)
    return jsonify(response.to_dict())


@labeling_bp.route("/sessions/<session_id>/statistics", methods=["GET"])
@error_handler
@request_logger
@validate_session(get_session_service)
def get_label_statistics(session_id: str):
    """
    ラベル統計を取得

    Args:
        session_id: セッションID

    Response:
        {
            "total_frames": 1000,
            "labeled_frames": 800,
            "unlabeled_frames": 200,
            "progress": 0.8,
            "label_distribution": {
                "game": 500,
                "menu": 200,
                "loading": 100
            },
            "average_confidence": 0.92
        }
    """
    from flask import g

    session = g.current_session
    statistics = _labeling_service.get_label_statistics(session)

    return jsonify(statistics)
