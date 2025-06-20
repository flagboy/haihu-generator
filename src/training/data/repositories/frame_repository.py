"""
フレームリポジトリ

フレームデータのCRUD操作を提供
"""

from datetime import datetime

from ..models import Frame
from .base_repository import BaseRepository


class FrameRepository(BaseRepository[Frame]):
    """フレームリポジトリクラス"""

    def create(self, entity: Frame) -> bool:
        """フレームを作成"""
        query = """
            INSERT INTO frames (id, video_id, image_path, timestamp, width, height,
                               quality_score, is_valid, scene_type, game_phase,
                               annotated_at, annotator, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity.frame_id,
            entity.video_id,
            entity.image_path,
            entity.timestamp,
            entity.width,
            entity.height,
            entity.quality_score,
            entity.is_valid,
            entity.scene_type,
            entity.game_phase,
            entity.annotated_at.isoformat() if entity.annotated_at else None,
            entity.annotator,
            entity.notes,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"フレームを作成: {entity.frame_id}")
        return success

    def find_by_id(self, entity_id: str) -> Frame | None:
        """IDでフレームを検索"""
        query = "SELECT * FROM frames WHERE id = ?"
        row = self.connection.fetch_one(query, (entity_id,))

        if row:
            return self._row_to_entity(row)
        return None

    def find_all(self) -> list[Frame]:
        """すべてのフレームを取得"""
        query = "SELECT * FROM frames ORDER BY video_id, timestamp"
        rows = self.connection.fetch_all(query)
        return [self._row_to_entity(row) for row in rows]

    def update(self, entity: Frame) -> bool:
        """フレームを更新"""
        query = """
            UPDATE frames
            SET image_path = ?, timestamp = ?, width = ?, height = ?,
                quality_score = ?, is_valid = ?, scene_type = ?, game_phase = ?,
                annotated_at = ?, annotator = ?, notes = ?
            WHERE id = ?
        """
        params = (
            entity.image_path,
            entity.timestamp,
            entity.width,
            entity.height,
            entity.quality_score,
            entity.is_valid,
            entity.scene_type,
            entity.game_phase,
            entity.annotated_at.isoformat() if entity.annotated_at else None,
            entity.annotator,
            entity.notes,
            entity.frame_id,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"フレームを更新: {entity.frame_id}")
        return success

    def delete(self, entity_id: str) -> bool:
        """フレームを削除"""
        query = "DELETE FROM frames WHERE id = ?"
        success = self._execute_query(query, (entity_id,))
        if success:
            self.logger.info(f"フレームを削除: {entity_id}")
        return success

    def find_by_video_id(self, video_id: str) -> list[Frame]:
        """
        動画IDでフレームを検索

        Args:
            video_id: 動画ID

        Returns:
            フレームのリスト
        """
        query = "SELECT * FROM frames WHERE video_id = ? ORDER BY timestamp"
        rows = self.connection.fetch_all(query, (video_id,))
        return [self._row_to_entity(row) for row in rows]

    def find_valid_frames(self, video_id: str | None = None) -> list[Frame]:
        """
        有効なフレームを検索

        Args:
            video_id: 動画ID（Noneの場合は全て）

        Returns:
            フレームのリスト
        """
        if video_id:
            query = "SELECT * FROM frames WHERE video_id = ? AND is_valid = 1 ORDER BY timestamp"
            rows = self.connection.fetch_all(query, (video_id,))
        else:
            query = "SELECT * FROM frames WHERE is_valid = 1 ORDER BY video_id, timestamp"
            rows = self.connection.fetch_all(query)

        return [self._row_to_entity(row) for row in rows]

    def find_by_scene_type(self, scene_type: str, video_id: str | None = None) -> list[Frame]:
        """
        シーンタイプでフレームを検索

        Args:
            scene_type: シーンタイプ
            video_id: 動画ID（オプション）

        Returns:
            フレームのリスト
        """
        if video_id:
            query = """
                SELECT * FROM frames
                WHERE scene_type = ? AND video_id = ?
                ORDER BY timestamp
            """
            rows = self.connection.fetch_all(query, (scene_type, video_id))
        else:
            query = """
                SELECT * FROM frames
                WHERE scene_type = ?
                ORDER BY video_id, timestamp
            """
            rows = self.connection.fetch_all(query, (scene_type,))

        return [self._row_to_entity(row) for row in rows]

    def count_by_video_id(self, video_id: str) -> int:
        """
        動画IDでフレーム数を取得

        Args:
            video_id: 動画ID

        Returns:
            フレーム数
        """
        query = "SELECT COUNT(*) as count FROM frames WHERE video_id = ?"
        row = self.connection.fetch_one(query, (video_id,))
        return row["count"] if row else 0

    def _row_to_entity(self, row) -> Frame:
        """データベース行をエンティティに変換"""
        return Frame(
            frame_id=row["id"],
            video_id=row["video_id"],
            image_path=row["image_path"],
            timestamp=row["timestamp"] or 0.0,
            width=row["width"] or 1920,
            height=row["height"] or 1080,
            quality_score=row["quality_score"] or 1.0,
            is_valid=bool(row["is_valid"]) if row["is_valid"] is not None else True,
            scene_type=row["scene_type"] or "game",
            game_phase=row["game_phase"] or "unknown",
            annotated_at=datetime.fromisoformat(row["annotated_at"])
            if row["annotated_at"]
            else None,
            annotator=row["annotator"] or "unknown",
            notes=row["notes"] or "",
        )
