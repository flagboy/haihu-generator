"""
バージョンエンティティ

データセットのバージョン情報を表現するドメインモデル
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DatasetVersion:
    """データセットバージョンエンティティ"""

    version_id: str
    version: str
    description: str = ""
    created_at: datetime | None = None
    frame_count: int = 0
    tile_count: int = 0
    export_path: str | None = None
    checksum: str | None = None

    def __post_init__(self):
        """初期化後の処理"""
        # 作成日時が未設定の場合は現在時刻を設定
        if self.created_at is None:
            self.created_at = datetime.now()

    def update_statistics(self, frame_count: int, tile_count: int):
        """
        統計情報を更新

        Args:
            frame_count: フレーム数
            tile_count: 牌数
        """
        self.frame_count = frame_count
        self.tile_count = tile_count

    def to_dict(self) -> dict[str, any]:
        """辞書形式に変換"""
        return {
            "version_id": self.version_id,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "frame_count": self.frame_count,
            "tile_count": self.tile_count,
            "export_path": self.export_path,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> "DatasetVersion":
        """
        辞書から生成

        Args:
            data: バージョンデータ

        Returns:
            バージョンエンティティ
        """
        return cls(
            version_id=data["version_id"],
            version=data["version"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            frame_count=data.get("frame_count", 0),
            tile_count=data.get("tile_count", 0),
            export_path=data.get("export_path"),
            checksum=data.get("checksum"),
        )
