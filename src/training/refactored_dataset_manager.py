"""
リファクタリングされたDatasetManager

新しいアーキテクチャを使用してデータセット管理機能を提供
既存のインターフェースを維持しながら内部実装を改善
"""

from pathlib import Path
from typing import Any

from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from .annotation_data import AnnotationData
from .data.services import DatasetService, ExportService, VersionService


class RefactoredDatasetManager(LoggerMixin):
    """リファクタリングされたDatasetManagerクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # サービスの初期化
        self.dataset_service = DatasetService(config_manager)
        self.export_service = ExportService()
        self.version_service = VersionService(config_manager)

        # 互換性のためのプロパティ
        self.dataset_root = Path(
            self.config.get("training", {}).get("dataset_root", "data/training")
        )
        self.db_path = Path(
            self.config.get("training", {}).get("database_path", "data/training/dataset.db")
        )

        self.logger.info("RefactoredDatasetManager初期化完了")

    def save_annotation_data(self, annotation_data: AnnotationData) -> bool:
        """
        アノテーションデータをデータベースに保存

        Args:
            annotation_data: アノテーションデータ

        Returns:
            保存成功かどうか
        """
        return self.dataset_service.save_annotation_data(annotation_data)

    def load_annotation_data(self, video_id: str | None = None) -> AnnotationData:
        """
        データベースからアノテーションデータを読み込み

        Args:
            video_id: 特定の動画IDを指定（Noneの場合は全て）

        Returns:
            アノテーションデータ
        """
        return self.dataset_service.load_annotation_data(video_id)

    def create_dataset_version(
        self, annotation_data: AnnotationData, version: str, description: str = ""
    ) -> str | None:
        """
        データセットのバージョンを作成

        Args:
            annotation_data: アノテーションデータ
            version: バージョン名
            description: 説明

        Returns:
            バージョンID
        """
        dataset_version = self.version_service.create_version(annotation_data, version, description)
        return dataset_version.version_id if dataset_version else None

    def export_dataset(
        self, version_id: str, export_format: str = "yolo", output_dir: str | None = None
    ) -> bool:
        """
        データセットをエクスポート

        Args:
            version_id: バージョンID
            export_format: エクスポート形式 ("yolo", "coco", "pascal_voc")
            output_dir: 出力ディレクトリ

        Returns:
            エクスポート成功かどうか
        """
        export_path = self.version_service.export_version(version_id, export_format, output_dir)
        return export_path is not None

    def get_dataset_statistics(self) -> dict[str, Any]:
        """データセット全体の統計情報を取得"""
        stats = self.dataset_service.get_dataset_statistics()

        # 追加情報
        version_count = len(self.version_service.list_versions())
        stats.update(
            {
                "version_count": version_count,
                "dataset_root": str(self.dataset_root),
                "database_path": str(self.db_path),
            }
        )

        return stats

    def list_versions(self) -> list[dict[str, Any]]:
        """データセットバージョン一覧を取得"""
        versions = self.version_service.list_versions()
        return [
            {
                "id": v.version_id,
                "version": v.version,
                "description": v.description,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "frame_count": v.frame_count,
                "tile_count": v.tile_count,
            }
            for v in versions
        ]

    def cleanup_old_versions(self, keep_count: int = 5) -> bool:
        """古いバージョンをクリーンアップ"""
        deleted_count = self.version_service.cleanup_old_versions(keep_count)
        return deleted_count > 0

    # 以下、互換性のためのメソッド（内部実装は新しいアーキテクチャを使用）

    def list_videos(self) -> list[dict]:
        """動画一覧を取得"""
        videos = self.dataset_service.video_repository.find_all()
        return [
            {
                "id": v.video_id,
                "name": v.name,
                "path": v.path,
                "upload_date": v.created_at.isoformat() if v.created_at else None,
            }
            for v in videos
        ]

    def get_video_info(self, video_id: int) -> dict | None:
        """動画情報を取得"""
        # 互換性のため、intをstrに変換
        video = self.dataset_service.video_repository.find_by_id(str(video_id))
        if not video:
            return None

        return {
            "id": video.video_id,
            "name": video.name,
            "path": video.path,
            "upload_date": video.created_at.isoformat() if video.created_at else None,
            "fps": video.fps or 30,
            "width": video.width or 1920,
            "height": video.height or 1080,
            "duration": video.duration or 0,
        }

    def get_frame_count(self, video_id: int) -> int:
        """動画のフレーム数を取得"""
        return self.dataset_service.frame_repository.count_by_video_id(str(video_id))

    def get_annotation_count(self, video_id: int | None = None, frame_id: int | None = None) -> int:
        """アノテーション数を取得"""
        if frame_id is not None:
            return self.dataset_service.annotation_repository.count_by_frame_id(str(frame_id))
        elif video_id is not None:
            frames = self.dataset_service.frame_repository.find_by_video_id(str(video_id))
            total = 0
            for frame in frames:
                total += self.dataset_service.annotation_repository.count_by_frame_id(
                    frame.frame_id
                )
            return total
        else:
            return self.dataset_service.annotation_repository.count()

    def list_frames(self, video_id: int | None = None) -> list[dict]:
        """フレーム一覧を取得"""
        if video_id is not None:
            frames = self.dataset_service.frame_repository.find_by_video_id(str(video_id))
        else:
            frames = self.dataset_service.frame_repository.find_all()

        return [
            {
                "id": f.frame_id,
                "video_id": f.video_id,
                "frame_number": int(f.timestamp * 30),  # FPSを仮定してフレーム番号を計算
                "timestamp": f.timestamp,
                "path": f.image_path,
            }
            for f in frames
        ]

    def get_frame_annotations(self, frame_id: int) -> list[dict]:
        """フレームのアノテーションを取得"""
        annotations = self.dataset_service.annotation_repository.find_by_frame_id(str(frame_id))

        return [
            {
                "id": a.annotation_id,
                "frame_id": a.frame_id,
                "class_id": a.tile_id,
                "x": a.bbox.x1,
                "y": a.bbox.y1,
                "width": a.bbox.width,
                "height": a.bbox.height,
                "confidence": a.confidence,
            }
            for a in annotations
        ]

    def delete_video(self, video_id: int):
        """動画と関連データを削除"""
        str_video_id = str(video_id)

        # フレームとアノテーションを削除
        frames = self.dataset_service.frame_repository.find_by_video_id(str_video_id)
        for frame in frames:
            self.dataset_service.annotation_repository.delete_by_frame_id(frame.frame_id)
            self.dataset_service.frame_repository.delete(frame.frame_id)

        # 動画を削除
        self.dataset_service.video_repository.delete(str_video_id)

        self.logger.info(f"動画を削除: video_id={video_id}")
