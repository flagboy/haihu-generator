"""
動画エンティティ

動画の情報を表現するドメインモデル
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Video:
    """動画エンティティ"""

    video_id: str
    name: str
    path: str
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後の処理"""
        # 作成日時が未設定の場合は現在時刻を設定
        if self.created_at is None:
            self.created_at = datetime.now()

        # 更新日時を現在時刻に設定
        self.updated_at = datetime.now()

    def update_metadata(self, key: str, value: Any):
        """
        メタデータを更新

        Args:
            key: キー
            value: 値
        """
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "video_id": self.video_id,
            "name": self.name,
            "path": self.path,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Video":
        """
        辞書から生成

        Args:
            data: 動画データ

        Returns:
            動画エンティティ
        """
        return cls(
            video_id=data["video_id"],
            name=data["name"],
            path=data["path"],
            duration=data.get("duration"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            metadata=data.get("metadata", {}),
        )
