"""
動画リポジトリ

動画データのCRUD操作を提供
"""

import json
from datetime import datetime

from ..models import Video
from .base_repository import BaseRepository


class VideoRepository(BaseRepository[Video]):
    """動画リポジトリクラス"""

    def create(self, entity: Video) -> bool:
        """動画を作成"""
        query = """
            INSERT INTO videos (id, name, path, duration, fps, width, height,
                               created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity.video_id,
            entity.name,
            entity.path,
            entity.duration,
            entity.fps,
            entity.width,
            entity.height,
            entity.created_at.isoformat() if entity.created_at else None,
            entity.updated_at.isoformat() if entity.updated_at else None,
            json.dumps(entity.metadata) if entity.metadata else None,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"動画を作成: {entity.video_id}")
        return success

    def find_by_id(self, entity_id: str) -> Video | None:
        """IDで動画を検索"""
        query = "SELECT * FROM videos WHERE id = ?"
        row = self.connection.fetch_one(query, (entity_id,))

        if row:
            return self._row_to_entity(row)
        return None

    def find_all(self) -> list[Video]:
        """すべての動画を取得"""
        query = "SELECT * FROM videos ORDER BY created_at DESC"
        rows = self.connection.fetch_all(query)
        return [self._row_to_entity(row) for row in rows]

    def update(self, entity: Video) -> bool:
        """動画を更新"""
        query = """
            UPDATE videos
            SET name = ?, path = ?, duration = ?, fps = ?, width = ?, height = ?,
                updated_at = ?, metadata = ?
            WHERE id = ?
        """
        params = (
            entity.name,
            entity.path,
            entity.duration,
            entity.fps,
            entity.width,
            entity.height,
            entity.updated_at.isoformat() if entity.updated_at else None,
            json.dumps(entity.metadata) if entity.metadata else None,
            entity.video_id,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"動画を更新: {entity.video_id}")
        return success

    def delete(self, entity_id: str) -> bool:
        """動画を削除"""
        query = "DELETE FROM videos WHERE id = ?"
        success = self._execute_query(query, (entity_id,))
        if success:
            self.logger.info(f"動画を削除: {entity_id}")
        return success

    def find_by_name(self, name: str) -> list[Video]:
        """
        名前で動画を検索

        Args:
            name: 動画名（部分一致）

        Returns:
            動画のリスト
        """
        query = "SELECT * FROM videos WHERE name LIKE ? ORDER BY created_at DESC"
        rows = self.connection.fetch_all(query, (f"%{name}%",))
        return [self._row_to_entity(row) for row in rows]

    def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[Video]:
        """
        日付範囲で動画を検索

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            動画のリスト
        """
        query = """
            SELECT * FROM videos
            WHERE created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
        """
        params = (start_date.isoformat(), end_date.isoformat())
        rows = self.connection.fetch_all(query, params)
        return [self._row_to_entity(row) for row in rows]

    def _row_to_entity(self, row) -> Video:
        """データベース行をエンティティに変換"""
        return Video(
            video_id=row["id"],
            name=row["name"],
            path=row["path"],
            duration=row["duration"],
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
