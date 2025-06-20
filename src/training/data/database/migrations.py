"""
データベースマイグレーション管理

スキーマの作成と更新を管理
"""

from collections.abc import Callable

from ....utils.logger import LoggerMixin
from .connection import DatabaseConnection


class DatabaseMigration(LoggerMixin):
    """データベースマイグレーション管理クラス"""

    def __init__(self, connection: DatabaseConnection):
        """
        初期化

        Args:
            connection: データベース接続
        """
        self.connection = connection

    def create_tables(self):
        """全テーブルを作成"""
        self._create_videos_table()
        self._create_frames_table()
        self._create_tile_annotations_table()
        self._create_dataset_versions_table()
        self._create_indexes()
        self.logger.info("データベーステーブルを作成しました")

    def _create_videos_table(self):
        """動画テーブルを作成"""
        query = """
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                duration REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                created_at TEXT,
                updated_at TEXT,
                metadata TEXT
            )
        """
        self.connection.execute(query)

    def _create_frames_table(self):
        """フレームテーブルを作成"""
        query = """
            CREATE TABLE IF NOT EXISTS frames (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                image_path TEXT NOT NULL,
                timestamp REAL,
                width INTEGER,
                height INTEGER,
                quality_score REAL,
                is_valid BOOLEAN,
                scene_type TEXT,
                game_phase TEXT,
                annotated_at TEXT,
                annotator TEXT,
                notes TEXT,
                FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
            )
        """
        self.connection.execute(query)

    def _create_tile_annotations_table(self):
        """牌アノテーションテーブルを作成"""
        query = """
            CREATE TABLE IF NOT EXISTS tile_annotations (
                id TEXT PRIMARY KEY,
                frame_id TEXT,
                tile_id TEXT NOT NULL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                confidence REAL,
                area_type TEXT,
                is_face_up BOOLEAN,
                is_occluded BOOLEAN,
                occlusion_ratio REAL,
                annotator TEXT,
                notes TEXT,
                FOREIGN KEY (frame_id) REFERENCES frames (id) ON DELETE CASCADE
            )
        """
        self.connection.execute(query)

    def _create_dataset_versions_table(self):
        """データセットバージョンテーブルを作成"""
        query = """
            CREATE TABLE IF NOT EXISTS dataset_versions (
                id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                description TEXT,
                created_at TEXT,
                frame_count INTEGER,
                tile_count INTEGER,
                export_path TEXT,
                checksum TEXT
            )
        """
        self.connection.execute(query)

    def _create_indexes(self):
        """インデックスを作成"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames (video_id)",
            "CREATE INDEX IF NOT EXISTS idx_tiles_frame_id ON tile_annotations (frame_id)",
            "CREATE INDEX IF NOT EXISTS idx_tiles_tile_id ON tile_annotations (tile_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frames (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_versions_created_at ON dataset_versions (created_at)",
        ]

        for index_query in indexes:
            self.connection.execute(index_query)

    def drop_all_tables(self):
        """全テーブルを削除（テスト用）"""
        tables = ["tile_annotations", "frames", "videos", "dataset_versions"]

        with self.connection.transaction() as conn:
            cursor = conn.cursor()
            # 外部キー制約を一時的に無効化
            cursor.execute("PRAGMA foreign_keys = OFF")

            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")

            # 外部キー制約を再度有効化
            cursor.execute("PRAGMA foreign_keys = ON")

        self.logger.info("全テーブルを削除しました")

    def migrate(self, migrations: list[Callable[[], None]]):
        """
        マイグレーションを実行

        Args:
            migrations: マイグレーション関数のリスト
        """
        for migration in migrations:
            try:
                migration()
                self.logger.info(f"マイグレーション実行: {migration.__name__}")
            except Exception as e:
                self.logger.error(f"マイグレーション失敗: {migration.__name__} - {e}")
                raise
