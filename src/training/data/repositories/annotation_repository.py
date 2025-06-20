"""
アノテーションリポジトリ

牌アノテーションデータのCRUD操作を提供
"""

from ..models import BoundingBox, TileAnnotation
from .base_repository import BaseRepository


class AnnotationRepository(BaseRepository[TileAnnotation]):
    """アノテーションリポジトリクラス"""

    def create(self, entity: TileAnnotation) -> bool:
        """アノテーションを作成"""
        query = """
            INSERT INTO tile_annotations (id, frame_id, tile_id, x1, y1, x2, y2,
                                         confidence, area_type, is_face_up, is_occluded,
                                         occlusion_ratio, annotator, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            entity.annotation_id,
            entity.frame_id,
            entity.tile_id,
            entity.bbox.x1,
            entity.bbox.y1,
            entity.bbox.x2,
            entity.bbox.y2,
            entity.confidence,
            entity.area_type,
            entity.is_face_up,
            entity.is_occluded,
            entity.occlusion_ratio,
            entity.annotator,
            entity.notes,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"アノテーションを作成: {entity.annotation_id}")
        return success

    def find_by_id(self, entity_id: str) -> TileAnnotation | None:
        """IDでアノテーションを検索"""
        query = "SELECT * FROM tile_annotations WHERE id = ?"
        row = self.connection.fetch_one(query, (entity_id,))

        if row:
            return self._row_to_entity(row)
        return None

    def find_all(self) -> list[TileAnnotation]:
        """すべてのアノテーションを取得"""
        query = "SELECT * FROM tile_annotations"
        rows = self.connection.fetch_all(query)
        return [self._row_to_entity(row) for row in rows]

    def update(self, entity: TileAnnotation) -> bool:
        """アノテーションを更新"""
        query = """
            UPDATE tile_annotations
            SET tile_id = ?, x1 = ?, y1 = ?, x2 = ?, y2 = ?,
                confidence = ?, area_type = ?, is_face_up = ?, is_occluded = ?,
                occlusion_ratio = ?, annotator = ?, notes = ?
            WHERE id = ?
        """
        params = (
            entity.tile_id,
            entity.bbox.x1,
            entity.bbox.y1,
            entity.bbox.x2,
            entity.bbox.y2,
            entity.confidence,
            entity.area_type,
            entity.is_face_up,
            entity.is_occluded,
            entity.occlusion_ratio,
            entity.annotator,
            entity.notes,
            entity.annotation_id,
        )

        success = self._execute_query(query, params)
        if success:
            self.logger.info(f"アノテーションを更新: {entity.annotation_id}")
        return success

    def delete(self, entity_id: str) -> bool:
        """アノテーションを削除"""
        query = "DELETE FROM tile_annotations WHERE id = ?"
        success = self._execute_query(query, (entity_id,))
        if success:
            self.logger.info(f"アノテーションを削除: {entity_id}")
        return success

    def find_by_frame_id(self, frame_id: str) -> list[TileAnnotation]:
        """
        フレームIDでアノテーションを検索

        Args:
            frame_id: フレームID

        Returns:
            アノテーションのリスト
        """
        query = "SELECT * FROM tile_annotations WHERE frame_id = ?"
        rows = self.connection.fetch_all(query, (frame_id,))
        return [self._row_to_entity(row) for row in rows]

    def find_by_tile_id(self, tile_id: str) -> list[TileAnnotation]:
        """
        牌IDでアノテーションを検索

        Args:
            tile_id: 牌ID

        Returns:
            アノテーションのリスト
        """
        query = "SELECT * FROM tile_annotations WHERE tile_id = ?"
        rows = self.connection.fetch_all(query, (tile_id,))
        return [self._row_to_entity(row) for row in rows]

    def find_by_area_type(self, area_type: str) -> list[TileAnnotation]:
        """
        エリアタイプでアノテーションを検索

        Args:
            area_type: エリアタイプ

        Returns:
            アノテーションのリスト
        """
        query = "SELECT * FROM tile_annotations WHERE area_type = ?"
        rows = self.connection.fetch_all(query, (area_type,))
        return [self._row_to_entity(row) for row in rows]

    def find_reliable_annotations(self, confidence_threshold: float = 0.8) -> list[TileAnnotation]:
        """
        信頼できるアノテーションを検索

        Args:
            confidence_threshold: 信頼度の閾値

        Returns:
            アノテーションのリスト
        """
        query = """
            SELECT * FROM tile_annotations
            WHERE confidence >= ? AND is_occluded = 0
        """
        rows = self.connection.fetch_all(query, (confidence_threshold,))
        return [self._row_to_entity(row) for row in rows]

    def count_by_frame_id(self, frame_id: str) -> int:
        """
        フレームIDでアノテーション数を取得

        Args:
            frame_id: フレームID

        Returns:
            アノテーション数
        """
        query = "SELECT COUNT(*) as count FROM tile_annotations WHERE frame_id = ?"
        row = self.connection.fetch_one(query, (frame_id,))
        return row["count"] if row else 0

    def get_tile_distribution(self) -> dict[str, int]:
        """
        牌の分布を取得

        Returns:
            牌ID別の出現回数
        """
        query = "SELECT tile_id, COUNT(*) as count FROM tile_annotations GROUP BY tile_id"
        rows = self.connection.fetch_all(query)
        return {row["tile_id"]: row["count"] for row in rows}

    def delete_by_frame_id(self, frame_id: str) -> bool:
        """
        フレームIDでアノテーションを一括削除

        Args:
            frame_id: フレームID

        Returns:
            成功したかどうか
        """
        query = "DELETE FROM tile_annotations WHERE frame_id = ?"
        success = self._execute_query(query, (frame_id,))
        if success:
            self.logger.info(f"フレームのアノテーションを削除: {frame_id}")
        return success

    def _row_to_entity(self, row) -> TileAnnotation:
        """データベース行をエンティティに変換"""
        bbox = BoundingBox(
            x1=row["x1"],
            y1=row["y1"],
            x2=row["x2"],
            y2=row["y2"],
        )

        return TileAnnotation(
            annotation_id=row["id"],
            frame_id=row["frame_id"],
            tile_id=row["tile_id"],
            bbox=bbox,
            confidence=row["confidence"] or 1.0,
            area_type=row["area_type"] or "unknown",
            is_face_up=bool(row["is_face_up"]) if row["is_face_up"] is not None else True,
            is_occluded=bool(row["is_occluded"]) if row["is_occluded"] is not None else False,
            occlusion_ratio=row["occlusion_ratio"] or 0.0,
            annotator=row["annotator"] or "unknown",
            notes=row["notes"] or "",
        )
