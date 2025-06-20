"""
バージョンサービス

データセットのバージョン管理機能を提供
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from ....utils.config import ConfigManager
from ....utils.file_io import FileIOHelper
from ....utils.logger import LoggerMixin
from ...annotation_data import AnnotationData
from ..database import DatabaseConnection
from ..models import DatasetVersion
from ..repositories import VersionRepository
from .export_service import ExportService


class VersionService(LoggerMixin):
    """バージョンサービスクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # データベース接続
        db_path = Path(
            self.config.get("training", {}).get("database_path", "data/training/dataset.db")
        )
        self.connection = DatabaseConnection(db_path)
        self.version_repository = VersionRepository(self.connection)

        # バージョンディレクトリ
        self.versions_dir = (
            Path(self.config.get("training", {}).get("dataset_root", "data/training")) / "versions"
        )
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # エクスポートサービス
        self.export_service = ExportService()

        self.logger.info("VersionService初期化完了")

    def create_version(
        self,
        annotation_data: AnnotationData,
        version_name: str,
        description: str = "",
    ) -> DatasetVersion | None:
        """
        データセットのバージョンを作成

        Args:
            annotation_data: アノテーションデータ
            version_name: バージョン名
            description: 説明

        Returns:
            作成されたバージョンまたはNone
        """
        try:
            # 既存のバージョン名をチェック
            if self.version_repository.find_by_version(version_name):
                self.logger.error(f"バージョン名が既に存在します: {version_name}")
                return None

            # バージョンIDを生成
            version_id = str(uuid.uuid4())
            version_dir = self.versions_dir / version_id
            version_dir.mkdir(exist_ok=True)

            # アノテーションデータを保存
            annotation_path = version_dir / "annotations.json"
            annotation_data.save_to_json(str(annotation_path))

            # 統計情報を保存
            stats = annotation_data.get_all_statistics()
            stats_path = version_dir / "statistics.json"
            FileIOHelper.save_json(stats, stats_path, pretty=True)

            # チェックサムを計算
            checksum = self.export_service.calculate_checksum(version_dir)

            # バージョンエンティティを作成
            version = DatasetVersion(
                version_id=version_id,
                version=version_name,
                description=description,
                created_at=datetime.now(),
                frame_count=stats.get("total_annotated_frames", 0),
                tile_count=stats.get("total_tiles", 0),
                export_path=str(version_dir),
                checksum=checksum,
            )

            # データベースに保存
            if self.version_repository.create(version):
                self.logger.info(f"バージョン作成: {version_name} (ID: {version_id})")
                return version
            else:
                # 失敗時はディレクトリを削除
                shutil.rmtree(version_dir)
                return None

        except Exception as e:
            self.logger.error(f"バージョン作成に失敗: {e}")
            return None

    def export_version(
        self,
        version_id: str,
        export_format: str = "yolo",
        output_dir: str | None = None,
    ) -> Path | None:
        """
        バージョンをエクスポート

        Args:
            version_id: バージョンID
            export_format: エクスポート形式
            output_dir: 出力ディレクトリ

        Returns:
            エクスポートディレクトリパスまたはNone
        """
        try:
            # バージョンを取得
            version = self.version_repository.find_by_id(version_id)
            if not version:
                self.logger.error(f"バージョンが見つかりません: {version_id}")
                return None

            # アノテーションデータを読み込み
            version_dir = Path(version.export_path)
            annotation_path = version_dir / "annotations.json"

            annotation_data = AnnotationData()
            annotation_data.load_from_json(str(annotation_path))

            # エクスポート
            export_path = self.export_service.export_dataset(
                annotation_data=annotation_data,
                export_format=export_format,
                output_dir=output_dir,
                version_name=version.version,
            )

            return export_path

        except Exception as e:
            self.logger.error(f"バージョンエクスポートに失敗: {e}")
            return None

    def list_versions(self) -> list[DatasetVersion]:
        """
        バージョン一覧を取得

        Returns:
            バージョンのリスト
        """
        return self.version_repository.find_all()

    def get_latest_version(self) -> DatasetVersion | None:
        """
        最新のバージョンを取得

        Returns:
            最新のバージョンまたはNone
        """
        return self.version_repository.find_latest()

    def cleanup_old_versions(self, keep_count: int = 5) -> int:
        """
        古いバージョンをクリーンアップ

        Args:
            keep_count: 保持するバージョン数

        Returns:
            削除したバージョン数
        """
        try:
            # 削除対象のバージョンを取得
            old_versions = self.version_repository.get_old_versions(keep_count)
            deleted_count = 0

            for version in old_versions:
                # ディレクトリを削除
                version_dir = Path(version.export_path)
                if version_dir.exists():
                    shutil.rmtree(version_dir)

                # データベースから削除
                if self.version_repository.delete(version.version_id):
                    deleted_count += 1

            if deleted_count > 0:
                self.logger.info(f"古いバージョンを削除: {deleted_count}個")

            return deleted_count

        except Exception as e:
            self.logger.error(f"バージョンクリーンアップに失敗: {e}")
            return 0

    def get_version_info(self, version_id: str) -> dict[str, any] | None:
        """
        バージョンの詳細情報を取得

        Args:
            version_id: バージョンID

        Returns:
            バージョン情報またはNone
        """
        try:
            version = self.version_repository.find_by_id(version_id)
            if not version:
                return None

            # 統計情報を読み込み
            version_dir = Path(version.export_path)
            stats_path = version_dir / "statistics.json"
            stats = FileIOHelper.load_json(stats_path) if stats_path.exists() else {}

            return {
                "version": version.to_dict(),
                "statistics": stats,
                "directory_size": self._calculate_directory_size(version_dir),
            }

        except Exception as e:
            self.logger.error(f"バージョン情報の取得に失敗: {e}")
            return None

    def _calculate_directory_size(self, directory: Path) -> int:
        """
        ディレクトリサイズを計算

        Args:
            directory: ディレクトリパス

        Returns:
            サイズ（バイト）
        """
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
