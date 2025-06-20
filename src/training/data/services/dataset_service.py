"""
データセットサービス

データセット操作のビジネスロジックを提供
"""

import uuid
from pathlib import Path
from typing import Any

from ....utils.config import ConfigManager
from ....utils.logger import LoggerMixin
from ...annotation_data import AnnotationData, FrameAnnotation, VideoAnnotation
from ..database import DatabaseConnection, DatabaseMigration
from ..models import BoundingBox, Frame, TileAnnotation, Video
from ..repositories import (
    AnnotationRepository,
    FrameRepository,
    VideoRepository,
)


class DatasetService(LoggerMixin):
    """データセットサービスクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # データベース設定
        db_path = Path(
            self.config.get("training", {}).get("database_path", "data/training/dataset.db")
        )
        self.connection = DatabaseConnection(db_path)

        # マイグレーション実行
        migration = DatabaseMigration(self.connection)
        migration.create_tables()

        # リポジトリの初期化
        self.video_repository = VideoRepository(self.connection)
        self.frame_repository = FrameRepository(self.connection)
        self.annotation_repository = AnnotationRepository(self.connection)

        self.logger.info("DatasetService初期化完了")

    def save_annotation_data(self, annotation_data: AnnotationData) -> bool:
        """
        アノテーションデータを保存

        Args:
            annotation_data: アノテーションデータ

        Returns:
            保存成功かどうか
        """
        try:
            with self.connection.transaction():
                for video_annotation in annotation_data.video_annotations.values():
                    # 動画を保存
                    video = self._video_annotation_to_model(video_annotation)
                    if not self.video_repository.exists(video.video_id):
                        self.video_repository.create(video)
                    else:
                        self.video_repository.update(video)

                    # フレームとアノテーションを保存
                    for frame_annotation in video_annotation.frames:
                        # フレームを保存
                        frame = self._frame_annotation_to_model(
                            frame_annotation, video_annotation.video_id
                        )
                        if not self.frame_repository.exists(frame.frame_id):
                            self.frame_repository.create(frame)
                        else:
                            self.frame_repository.update(frame)

                        # 既存のアノテーションを削除
                        self.annotation_repository.delete_by_frame_id(frame.frame_id)

                        # 新しいアノテーションを保存
                        for tile_annotation in frame_annotation.tiles:
                            annotation = self._tile_annotation_to_model(
                                tile_annotation, frame.frame_id
                            )
                            self.annotation_repository.create(annotation)

            self.logger.info("アノテーションデータを保存しました")
            return True

        except Exception as e:
            self.logger.error(f"アノテーションデータの保存に失敗: {e}")
            return False

    def load_annotation_data(self, video_id: str | None = None) -> AnnotationData:
        """
        アノテーションデータを読み込み

        Args:
            video_id: 動画ID（Noneの場合は全て）

        Returns:
            アノテーションデータ
        """
        annotation_data = AnnotationData()

        try:
            # 動画を取得
            if video_id:
                video = self.video_repository.find_by_id(video_id)
                videos = [video] if video else []
            else:
                videos = self.video_repository.find_all()

            for video in videos:
                # フレームを取得
                frames = self.frame_repository.find_by_video_id(video.video_id)

                # VideoAnnotationを作成
                video_annotation = VideoAnnotation(
                    video_id=video.video_id,
                    video_path=video.path,
                    video_name=video.name,
                    duration=video.duration or 0.0,
                    fps=video.fps or 30.0,
                    width=video.width or 1920,
                    height=video.height or 1080,
                    frames=[],
                    created_at=video.created_at,
                    updated_at=video.updated_at,
                    metadata=video.metadata,
                )

                # フレームごとの処理
                for frame in frames:
                    # アノテーションを取得
                    annotations = self.annotation_repository.find_by_frame_id(frame.frame_id)

                    # TileAnnotationに変換
                    tiles = []
                    for annotation in annotations:
                        from ...annotation_data import BoundingBox as OldBBox
                        from ...annotation_data import TileAnnotation as OldTileAnnotation

                        old_bbox = OldBBox(
                            x1=annotation.bbox.x1,
                            y1=annotation.bbox.y1,
                            x2=annotation.bbox.x2,
                            y2=annotation.bbox.y2,
                        )

                        tile = OldTileAnnotation(
                            tile_id=annotation.tile_id,
                            bbox=old_bbox,
                            confidence=annotation.confidence,
                            area_type=annotation.area_type,
                            is_face_up=annotation.is_face_up,
                            is_occluded=annotation.is_occluded,
                            occlusion_ratio=annotation.occlusion_ratio,
                            annotator=annotation.annotator,
                            notes=annotation.notes,
                        )
                        tiles.append(tile)

                    # FrameAnnotationを作成
                    frame_annotation = FrameAnnotation(
                        frame_id=frame.frame_id,
                        image_path=frame.image_path,
                        image_width=frame.width,
                        image_height=frame.height,
                        timestamp=frame.timestamp,
                        tiles=tiles,
                        quality_score=frame.quality_score,
                        is_valid=frame.is_valid,
                        scene_type=frame.scene_type,
                        game_phase=frame.game_phase,
                        annotated_at=frame.annotated_at,
                        annotator=frame.annotator,
                        notes=frame.notes,
                    )

                    video_annotation.frames.append(frame_annotation)

                annotation_data.video_annotations[video.video_id] = video_annotation

            self.logger.info(
                f"アノテーションデータを読み込み: {len(annotation_data.video_annotations)}動画"
            )

        except Exception as e:
            self.logger.error(f"アノテーションデータの読み込みに失敗: {e}")

        return annotation_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """データセット統計情報を取得"""
        try:
            video_count = self.video_repository.count()
            frame_count = self.frame_repository.count()
            annotation_count = self.annotation_repository.count()

            # 牌種類別統計
            tile_distribution = self.annotation_repository.get_tile_distribution()

            # エリア別統計
            area_types = ["hand", "discard", "wall", "dora"]
            area_distribution = {}
            for area_type in area_types:
                annotations = self.annotation_repository.find_by_area_type(area_type)
                area_distribution[area_type] = len(annotations)

            return {
                "video_count": video_count,
                "frame_count": frame_count,
                "annotation_count": annotation_count,
                "tile_distribution": tile_distribution,
                "area_distribution": area_distribution,
                "average_annotations_per_frame": annotation_count / frame_count
                if frame_count > 0
                else 0,
            }

        except Exception as e:
            self.logger.error(f"統計情報の取得に失敗: {e}")
            return {}

    def _video_annotation_to_model(self, video_annotation: VideoAnnotation) -> Video:
        """VideoAnnotationをVideoモデルに変換"""
        return Video(
            video_id=video_annotation.video_id,
            name=video_annotation.video_name,
            path=video_annotation.video_path,
            duration=video_annotation.duration,
            fps=video_annotation.fps,
            width=video_annotation.width,
            height=video_annotation.height,
            created_at=video_annotation.created_at,
            updated_at=video_annotation.updated_at,
            metadata=video_annotation.metadata or {},
        )

    def _frame_annotation_to_model(self, frame_annotation: FrameAnnotation, video_id: str) -> Frame:
        """FrameAnnotationをFrameモデルに変換"""
        return Frame(
            frame_id=frame_annotation.frame_id,
            video_id=video_id,
            image_path=frame_annotation.image_path,
            timestamp=frame_annotation.timestamp,
            width=frame_annotation.image_width,
            height=frame_annotation.image_height,
            quality_score=frame_annotation.quality_score,
            is_valid=frame_annotation.is_valid,
            scene_type=frame_annotation.scene_type,
            game_phase=frame_annotation.game_phase,
            annotated_at=frame_annotation.annotated_at,
            annotator=frame_annotation.annotator,
            notes=frame_annotation.notes,
        )

    def _tile_annotation_to_model(self, tile_annotation: Any, frame_id: str) -> TileAnnotation:
        """TileAnnotationをモデルに変換"""
        bbox = BoundingBox(
            x1=tile_annotation.bbox.x1,
            y1=tile_annotation.bbox.y1,
            x2=tile_annotation.bbox.x2,
            y2=tile_annotation.bbox.y2,
        )

        return TileAnnotation(
            annotation_id=str(uuid.uuid4()),
            frame_id=frame_id,
            tile_id=tile_annotation.tile_id,
            bbox=bbox,
            confidence=tile_annotation.confidence,
            area_type=tile_annotation.area_type,
            is_face_up=tile_annotation.is_face_up,
            is_occluded=tile_annotation.is_occluded,
            occlusion_ratio=tile_annotation.occlusion_ratio,
            annotator=tile_annotation.annotator,
            notes=tile_annotation.notes,
        )
