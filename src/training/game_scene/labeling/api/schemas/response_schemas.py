"""
レスポンススキーマ定義

APIレスポンスの形式定義
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionResponse:
    """セッション情報レスポンス"""

    session_id: str
    video_path: str
    total_frames: int
    labeled_frames: int
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "session_id": self.session_id,
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "labeled_frames": self.labeled_frames,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "status": self.status,
            "progress": self.labeled_frames / self.total_frames if self.total_frames > 0 else 0,
        }


@dataclass
class SessionListResponse:
    """セッション一覧レスポンス"""

    sessions: list[SessionResponse]
    total: int

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "sessions": [session.to_dict() for session in self.sessions],
            "total": self.total,
        }


@dataclass
class FrameResponse:
    """フレーム情報レスポンス"""

    frame_number: int
    timestamp: float
    image_data: str  # Base64エンコードされた画像
    label: str | None = None
    confidence: float | None = None
    is_labeled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "image": self.image_data,
            "label": self.label,
            "confidence": self.confidence,
            "is_labeled": self.is_labeled,
            "metadata": self.metadata,
        }


@dataclass
class SuccessResponse:
    """成功レスポンス"""

    success: bool = True
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        result: dict[str, Any] = {"success": self.success}
        if self.message:
            result["message"] = self.message
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class ErrorResponse:
    """エラーレスポンス"""

    error: str
    code: int = 400
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "error": {
                "message": self.error,
                "code": self.code,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        return result
