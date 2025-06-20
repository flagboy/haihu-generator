"""APIスキーマ定義"""

from .request_schemas import (
    AutoLabelRequest,
    BatchLabelRequest,
    CreateSessionRequest,
    LabelRequest,
)
from .response_schemas import (
    ErrorResponse,
    FrameResponse,
    SessionListResponse,
    SessionResponse,
    SuccessResponse,
)

__all__ = [
    # リクエスト
    "CreateSessionRequest",
    "LabelRequest",
    "BatchLabelRequest",
    "AutoLabelRequest",
    # レスポンス
    "SessionResponse",
    "SessionListResponse",
    "FrameResponse",
    "SuccessResponse",
    "ErrorResponse",
]
