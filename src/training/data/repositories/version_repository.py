"""
バージョンリポジトリ

データセットバージョンのCRUD操作を提供
"""

from datetime import datetime

from ..models import DatasetVersion
from .base_repository import BaseRepository


class VersionRepository(BaseRepository[DatasetVersion]):
    """バージョンリポジトリクラス"""

    def create(self, entity: DatasetVersion) -> bool:
        """バージョンを作成"""
        query = """
            INSERT INTO dataset_versions (id, version, description, created_at,
                                         frame_count, tile_count, export_path, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity.version_id,
            entity.version,
            entity.description,
            entity.created_at.isoformat() if entity.created_at else None,
            entity.frame_count,
            entity.tile_count,
            entity.export_path,
            entity.checksum,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"バージョンを作成: {entity.version}")
        return success

    def find_by_id(self, entity_id: str) -> DatasetVersion | None:
        """IDでバージョンを検索"""
        query = "SELECT * FROM dataset_versions WHERE id = ?"
        row = self.connection.fetch_one(query, (entity_id,))

        if row:
            return self._row_to_entity(row)
        return None

    def find_all(self) -> list[DatasetVersion]:
        """すべてのバージョンを取得"""
        query = "SELECT * FROM dataset_versions ORDER BY created_at DESC"
        rows = self.connection.fetch_all(query)
        return [self._row_to_entity(row) for row in rows]

    def update(self, entity: DatasetVersion) -> bool:
        """バージョンを更新"""
        query = """
            UPDATE dataset_versions
            SET version = ?, description = ?, frame_count = ?, tile_count = ?,
                export_path = ?, checksum = ?
            WHERE id = ?
        """
        params = (
            entity.version,
            entity.description,
            entity.frame_count,
            entity.tile_count,
            entity.export_path,
            entity.checksum,
            entity.version_id,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"バージョンを更新: {entity.version}")
        return success

    def delete(self, entity_id: str) -> bool:
        """バージョンを削除"""
        query = "DELETE FROM dataset_versions WHERE id = ?"
        success = self._execute_query(query, (entity_id,))
        if success:
            self.logger.info(f"バージョンを削除: {entity_id}")
        return success

    def find_by_version(self, version: str) -> DatasetVersion | None:
        """
        バージョン名でバージョンを検索

        Args:
            version: バージョン名

        Returns:
            バージョンまたはNone
        """
        query = "SELECT * FROM dataset_versions WHERE version = ?"
        row = self.connection.fetch_one(query, (version,))

        if row:
            return self._row_to_entity(row)
        return None

    def find_latest(self) -> DatasetVersion | None:
        """
        最新のバージョンを取得

        Returns:
            最新のバージョンまたはNone
        """
        query = "SELECT * FROM dataset_versions ORDER BY created_at DESC LIMIT 1"
        row = self.connection.fetch_one(query)

        if row:
            return self._row_to_entity(row)
        return None

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[DatasetVersion]:
        """
        日付範囲でバージョンを検索

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            バージョンのリスト
        """
        query = """
            SELECT * FROM dataset_versions
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
        """
        params = (start_date.isoformat(), end_date.isoformat())
        rows = self.connection.fetch_all(query, params)
        return [self._row_to_entity(row) for row in rows]

    def get_old_versions(self, keep_count: int = 5) -> list[DatasetVersion]:
        """
        古いバージョンを取得

        Args:
            keep_count: 保持するバージョン数

        Returns:
            削除対象のバージョンリスト
        """
        query = """
            SELECT * FROM dataset_versions
            ORDER BY created_at DESC
            LIMIT -1 OFFSET ?
        """
        rows = self.connection.fetch_all(query, (keep_count,))
        return [self._row_to_entity(row) for row in rows]

    def _row_to_entity(self, row) -> DatasetVersion:
        """データベース行をエンティティに変換"""
        return DatasetVersion(
            version_id=row["id"],
            version=row["version"],
            description=row["description"] or "",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            frame_count=row["frame_count"] or 0,
            tile_count=row["tile_count"] or 0,
            export_path=row["export_path"],
            checksum=row["checksum"],
        )
