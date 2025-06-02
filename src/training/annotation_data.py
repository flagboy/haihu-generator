"""
ラベリングデータの構造定義

教師データのアノテーション情報を管理するデータクラス
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logger import LoggerMixin


@dataclass
class BoundingBox:
    """バウンディングボックス情報"""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        """座標の正規化"""
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> int:
        """幅を取得"""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """高さを取得"""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """中心座標を取得"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> int:
        """面積を取得"""
        return self.width * self.height

    def to_yolo_format(
        self, image_width: int, image_height: int
    ) -> tuple[float, float, float, float]:
        """YOLO形式に変換 (center_x, center_y, width, height) - 正規化済み"""
        center_x = (self.x1 + self.x2) / 2 / image_width
        center_y = (self.y1 + self.y2) / 2 / image_height
        width = self.width / image_width
        height = self.height / image_height
        return (center_x, center_y, width, height)

    @classmethod
    def from_yolo_format(
        cls,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
    ) -> "BoundingBox":
        """YOLO形式から変換"""
        w = width * image_width
        h = height * image_height
        x1 = int((center_x * image_width) - (w / 2))
        y1 = int((center_y * image_height) - (h / 2))
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        return cls(x1, y1, x2, y2)


@dataclass
class TileAnnotation:
    """個別牌のアノテーション情報"""

    tile_id: str  # 牌の種類 (例: "1m", "2p", "3s", "東", "白" など)
    bbox: BoundingBox
    confidence: float = 1.0  # アノテーションの信頼度 (人手: 1.0, AI予測: 0.0-1.0)
    area_type: str = "unknown"  # "hand", "discard", "call", "unknown"
    is_face_up: bool = True  # 表向きかどうか
    is_occluded: bool = False  # 遮蔽されているかどうか
    occlusion_ratio: float = 0.0  # 遮蔽率 (0.0-1.0)
    annotator: str = "unknown"  # アノテーター (human, ai_model_name)
    notes: str = ""  # 備考

    def __post_init__(self):
        """バリデーション"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.occlusion_ratio <= 1.0:
            raise ValueError("occlusion_ratio must be between 0.0 and 1.0")
        if self.area_type not in ["hand", "discard", "call", "unknown"]:
            raise ValueError("area_type must be one of: hand, discard, call, unknown")


@dataclass
class FrameAnnotation:
    """フレーム単位のアノテーション情報"""

    frame_id: str
    image_path: str
    image_width: int
    image_height: int
    timestamp: float  # 動画内の時刻（秒）
    tiles: list[TileAnnotation]
    quality_score: float = 1.0  # フレーム品質スコア (0.0-1.0)
    is_valid: bool = True  # 有効なフレームかどうか
    scene_type: str = "game"  # "game", "menu", "transition", "other"
    game_phase: str = "unknown"  # "deal", "play", "call", "win", "unknown"
    annotated_at: datetime | None = None
    annotator: str = "unknown"
    notes: str = ""

    def __post_init__(self):
        """初期化後処理"""
        if self.annotated_at is None:
            self.annotated_at = datetime.now()
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")

    @property
    def tile_count(self) -> int:
        """牌の数を取得"""
        return len(self.tiles)

    @property
    def tile_types(self) -> list[str]:
        """牌の種類一覧を取得"""
        return [tile.tile_id for tile in self.tiles]

    def get_tiles_by_area(self, area_type: str) -> list[TileAnnotation]:
        """指定エリアの牌を取得"""
        return [tile for tile in self.tiles if tile.area_type == area_type]

    def get_high_confidence_tiles(self, threshold: float = 0.8) -> list[TileAnnotation]:
        """高信頼度の牌を取得"""
        return [tile for tile in self.tiles if tile.confidence >= threshold]


@dataclass
class VideoAnnotation:
    """動画単位のアノテーション情報"""

    video_id: str
    video_path: str
    video_name: str
    duration: float  # 動画の長さ（秒）
    fps: float
    width: int
    height: int
    frames: list[FrameAnnotation]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: str = "1.0"
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """初期化後処理"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def frame_count(self) -> int:
        """フレーム数を取得"""
        return len(self.frames)

    @property
    def total_tiles(self) -> int:
        """総牌数を取得"""
        return sum(frame.tile_count for frame in self.frames)

    @property
    def annotated_frames(self) -> list[FrameAnnotation]:
        """アノテーション済みフレームを取得"""
        return [frame for frame in self.frames if frame.tiles]

    def get_frames_by_quality(self, min_quality: float = 0.7) -> list[FrameAnnotation]:
        """品質スコア以上のフレームを取得"""
        return [frame for frame in self.frames if frame.quality_score >= min_quality]

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        annotated = self.annotated_frames
        if not annotated:
            return {
                "total_frames": self.frame_count,
                "annotated_frames": 0,
                "annotation_ratio": 0.0,
                "total_tiles": 0,
                "avg_tiles_per_frame": 0.0,
                "tile_types": [],
                "area_distribution": {},
                "quality_stats": {},
            }

        all_tiles = []
        for frame in annotated:
            all_tiles.extend(frame.tiles)

        tile_types = list({tile.tile_id for tile in all_tiles})

        area_counts = {}
        for tile in all_tiles:
            area_counts[tile.area_type] = area_counts.get(tile.area_type, 0) + 1

        qualities = [frame.quality_score for frame in self.frames]

        return {
            "total_frames": self.frame_count,
            "annotated_frames": len(annotated),
            "annotation_ratio": len(annotated) / self.frame_count if self.frame_count > 0 else 0.0,
            "total_tiles": len(all_tiles),
            "avg_tiles_per_frame": len(all_tiles) / len(annotated) if annotated else 0.0,
            "tile_types": sorted(tile_types),
            "tile_type_count": len(tile_types),
            "area_distribution": area_counts,
            "quality_stats": {
                "mean": np.mean(qualities) if qualities else 0.0,
                "std": np.std(qualities) if qualities else 0.0,
                "min": min(qualities) if qualities else 0.0,
                "max": max(qualities) if qualities else 0.0,
            },
        }


class AnnotationData(LoggerMixin):
    """アノテーションデータ管理クラス"""

    def __init__(self):
        """初期化"""
        self.video_annotations: dict[str, VideoAnnotation] = {}

    def create_video_annotation(self, video_path: str, video_info: dict[str, Any]) -> str:
        """動画アノテーションを作成"""
        video_id = str(uuid.uuid4())
        video_name = Path(video_path).stem

        annotation = VideoAnnotation(
            video_id=video_id,
            video_path=video_path,
            video_name=video_name,
            duration=video_info.get("duration", 0.0),
            fps=video_info.get("fps", 30.0),
            width=video_info.get("width", 1920),
            height=video_info.get("height", 1080),
            frames=[],
        )

        self.video_annotations[video_id] = annotation
        self.logger.info(f"動画アノテーション作成: {video_name} (ID: {video_id})")

        return video_id

    def add_frame_annotation(self, video_id: str, frame_annotation: FrameAnnotation) -> bool:
        """フレームアノテーションを追加"""
        if video_id not in self.video_annotations:
            self.logger.error(f"動画ID {video_id} が見つかりません")
            return False

        self.video_annotations[video_id].frames.append(frame_annotation)
        self.video_annotations[video_id].updated_at = datetime.now()

        return True

    def get_video_annotation(self, video_id: str) -> VideoAnnotation | None:
        """動画アノテーションを取得"""
        return self.video_annotations.get(video_id)

    def save_to_json(self, output_path: str) -> bool:
        """JSONファイルに保存"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # dataclassをdictに変換（datetimeは文字列に変換）
            data = {}
            for video_id, annotation in self.video_annotations.items():
                data[video_id] = self._annotation_to_dict(annotation)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"アノテーションデータを保存: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"アノテーションデータの保存に失敗: {e}")
            return False

    def load_from_json(self, input_path: str) -> bool:
        """JSONファイルから読み込み"""
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                self.logger.error(f"ファイルが見つかりません: {input_path}")
                return False

            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)

            self.video_annotations = {}
            for video_id, annotation_dict in data.items():
                annotation = self._dict_to_annotation(annotation_dict)
                self.video_annotations[video_id] = annotation

            self.logger.info(f"アノテーションデータを読み込み: {input_path}")
            return True

        except Exception as e:
            self.logger.error(f"アノテーションデータの読み込みに失敗: {e}")
            return False

    def _annotation_to_dict(self, annotation: VideoAnnotation) -> dict[str, Any]:
        """VideoAnnotationを辞書に変換"""
        result = asdict(annotation)

        # datetimeを文字列に変換
        if result.get("created_at"):
            result["created_at"] = annotation.created_at.isoformat()
        if result.get("updated_at"):
            result["updated_at"] = annotation.updated_at.isoformat()

        # フレームのdatetimeも変換
        for i, frame in enumerate(annotation.frames):
            if frame.annotated_at and result["frames"][i].get("annotated_at"):
                result["frames"][i]["annotated_at"] = frame.annotated_at.isoformat()

        return result

    def _dict_to_annotation(self, data: dict[str, Any]) -> VideoAnnotation:
        """辞書からVideoAnnotationを作成"""
        # datetimeを復元
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # フレームデータを復元
        frames = []
        for frame_data in data.get("frames", []):
            if frame_data.get("annotated_at"):
                frame_data["annotated_at"] = datetime.fromisoformat(frame_data["annotated_at"])

            # タイルデータを復元
            tiles = []
            for tile_data in frame_data.get("tiles", []):
                bbox_data = tile_data["bbox"]
                bbox = BoundingBox(**bbox_data)
                tile_data["bbox"] = bbox
                tiles.append(TileAnnotation(**tile_data))

            frame_data["tiles"] = tiles
            frames.append(FrameAnnotation(**frame_data))

        data["frames"] = frames
        return VideoAnnotation(**data)

    def export_yolo_format(self, output_dir: str, class_mapping: dict[str, int]) -> bool:
        """YOLO形式でエクスポート"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            images_dir = output_dir / "images"
            labels_dir = output_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            for video_annotation in self.video_annotations.values():
                for frame in video_annotation.frames:
                    if not frame.tiles:
                        continue

                    # ラベルファイルを作成
                    label_file = labels_dir / f"{frame.frame_id}.txt"
                    with open(label_file, "w") as f:
                        for tile in frame.tiles:
                            if tile.tile_id in class_mapping:
                                class_id = class_mapping[tile.tile_id]
                                center_x, center_y, width, height = tile.bbox.to_yolo_format(
                                    frame.image_width, frame.image_height
                                )
                                f.write(
                                    f"{class_id} {center_x:.6f} {center_y:.6f} "
                                    f"{width:.6f} {height:.6f}\n"
                                )

            # クラスマッピングを保存
            with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                for tile_name, _class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
                    f.write(f"{tile_name}\n")

            self.logger.info(f"YOLO形式でエクスポート完了: {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"YOLO形式エクスポートに失敗: {e}")
            return False

    def get_all_statistics(self) -> dict[str, Any]:
        """全動画の統計情報を取得"""
        if not self.video_annotations:
            return {}

        total_stats = {
            "video_count": len(self.video_annotations),
            "total_frames": 0,
            "total_annotated_frames": 0,
            "total_tiles": 0,
            "all_tile_types": set(),
            "area_distribution": {},
            "quality_stats": [],
        }

        for annotation in self.video_annotations.values():
            stats = annotation.get_statistics()
            total_stats["total_frames"] += stats["total_frames"]
            total_stats["total_annotated_frames"] += stats["annotated_frames"]
            total_stats["total_tiles"] += stats["total_tiles"]
            total_stats["all_tile_types"].update(stats["tile_types"])

            for area, count in stats["area_distribution"].items():
                total_stats["area_distribution"][area] = (
                    total_stats["area_distribution"].get(area, 0) + count
                )

            total_stats["quality_stats"].extend(
                [frame.quality_score for frame in annotation.frames]
            )

        # 統計値を計算
        total_stats["all_tile_types"] = sorted(total_stats["all_tile_types"])
        total_stats["annotation_ratio"] = (
            total_stats["total_annotated_frames"] / total_stats["total_frames"]
            if total_stats["total_frames"] > 0
            else 0.0
        )

        if total_stats["quality_stats"]:
            qualities = total_stats["quality_stats"]
            total_stats["overall_quality"] = {
                "mean": np.mean(qualities),
                "std": np.std(qualities),
                "min": min(qualities),
                "max": max(qualities),
            }

        return total_stats
