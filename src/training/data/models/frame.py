"""
フレームエンティティ

動画のフレーム情報を表現するドメインモデル
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Frame:
    """フレームエンティティ"""

    frame_id: str
    video_id: str
    image_path: str
    timestamp: float
    width: int
    height: int
    quality_score: float = 1.0
    is_valid: bool = True
    scene_type: str = "game"
    game_phase: str = "unknown"
    annotated_at: datetime | None = None
    annotator: str = "unknown"
    notes: str = ""

    def __post_init__(self):
        """初期化後の処理"""
        # アノテーション日時が未設定の場合は現在時刻を設定
        if self.annotated_at is None:
            self.annotated_at = datetime.now()

    def is_annotated(self) -> bool:
        """アノテーション済みかどうか"""
        return self.annotator != "unknown" and self.annotator != ""

    def update_quality_score(self, score: float):
        """
        品質スコアを更新

        Args:
            score: 品質スコア（0.0-1.0）
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("品質スコアは0.0から1.0の範囲で指定してください")
        self.quality_score = score

    def mark_as_invalid(self, reason: str = ""):
        """
        無効なフレームとしてマーク

        Args:
            reason: 無効な理由
        """
        self.is_valid = False
        if reason:
            self.notes = f"無効: {reason}"

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "scene_type": self.scene_type,
            "game_phase": self.game_phase,
            "annotated_at": self.annotated_at.isoformat() if self.annotated_at else None,
            "annotator": self.annotator,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Frame":
        """
        辞書から生成

        Args:
            data: フレームデータ

        Returns:
            フレームエンティティ
        """
        return cls(
            frame_id=data["frame_id"],
            video_id=data["video_id"],
            image_path=data["image_path"],
            timestamp=data.get("timestamp", 0.0),
            width=data.get("width", 1920),
            height=data.get("height", 1080),
            quality_score=data.get("quality_score", 1.0),
            is_valid=data.get("is_valid", True),
            scene_type=data.get("scene_type", "game"),
            game_phase=data.get("game_phase", "unknown"),
            annotated_at=datetime.fromisoformat(data["annotated_at"])
            if data.get("annotated_at")
            else None,
            annotator=data.get("annotator", "unknown"),
            notes=data.get("notes", ""),
        )
