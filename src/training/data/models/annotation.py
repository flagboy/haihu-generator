"""
アノテーションエンティティ

牌のアノテーション情報を表現するドメインモデル
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BoundingBox:
    """バウンディングボックス"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """幅を取得"""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """高さを取得"""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """面積を取得"""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """中心X座標を取得"""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """中心Y座標を取得"""
        return (self.y1 + self.y2) / 2

    def to_yolo_format(
        self, image_width: int, image_height: int
    ) -> tuple[float, float, float, float]:
        """
        YOLO形式に変換

        Args:
            image_width: 画像の幅
            image_height: 画像の高さ

        Returns:
            (center_x, center_y, width, height) の正規化された値
        """
        center_x = self.center_x / image_width
        center_y = self.center_y / image_height
        width = self.width / image_width
        height = self.height / image_height
        return center_x, center_y, width, height


@dataclass
class TileAnnotation:
    """牌アノテーションエンティティ"""

    annotation_id: str
    frame_id: str
    tile_id: str
    bbox: BoundingBox
    confidence: float = 1.0
    area_type: str = "unknown"
    is_face_up: bool = True
    is_occluded: bool = False
    occlusion_ratio: float = 0.0
    annotator: str = "unknown"
    notes: str = ""

    def __post_init__(self):
        """初期化後の処理"""
        # 信頼度の範囲チェック
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("信頼度は0.0から1.0の範囲で指定してください")

        # 遮蔽率の範囲チェック
        if not 0.0 <= self.occlusion_ratio <= 1.0:
            raise ValueError("遮蔽率は0.0から1.0の範囲で指定してください")

    def is_reliable(self, threshold: float = 0.8) -> bool:
        """
        信頼できるアノテーションかどうか

        Args:
            threshold: 信頼度の閾値

        Returns:
            信頼できるかどうか
        """
        return self.confidence >= threshold and not self.is_occluded

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return {
            "annotation_id": self.annotation_id,
            "frame_id": self.frame_id,
            "tile_id": self.tile_id,
            "x1": self.bbox.x1,
            "y1": self.bbox.y1,
            "x2": self.bbox.x2,
            "y2": self.bbox.y2,
            "confidence": self.confidence,
            "area_type": self.area_type,
            "is_face_up": self.is_face_up,
            "is_occluded": self.is_occluded,
            "occlusion_ratio": self.occlusion_ratio,
            "annotator": self.annotator,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileAnnotation":
        """
        辞書から生成

        Args:
            data: アノテーションデータ

        Returns:
            アノテーションエンティティ
        """
        bbox = BoundingBox(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
        )

        return cls(
            annotation_id=data["annotation_id"],
            frame_id=data["frame_id"],
            tile_id=data["tile_id"],
            bbox=bbox,
            confidence=data.get("confidence", 1.0),
            area_type=data.get("area_type", "unknown"),
            is_face_up=data.get("is_face_up", True),
            is_occluded=data.get("is_occluded", False),
            occlusion_ratio=data.get("occlusion_ratio", 0.0),
            annotator=data.get("annotator", "unknown"),
            notes=data.get("notes", ""),
        )
