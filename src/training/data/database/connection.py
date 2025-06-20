"""
データベース接続管理

SQLiteデータベースへの接続を管理し、
トランザクション処理を提供
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ....utils.logger import LoggerMixin


class DatabaseConnection(LoggerMixin):
    """データベース接続管理クラス"""

    def __init__(self, db_path: str | Path):
        """
        初期化

        Args:
            db_path: データベースファイルパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """データベースの初期化"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 外部キー制約を有効化
            cursor.execute("PRAGMA foreign_keys = ON")

            # WALモードを有効化（パフォーマンス向上）
            cursor.execute("PRAGMA journal_mode = WAL")

            conn.commit()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        データベース接続を取得

        Yields:
            データベース接続
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 辞書形式でアクセス可能に
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"データベースエラー: {e}")
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        トランザクション管理

        Yields:
            データベース接続
        """
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> sqlite3.Cursor:
        """
        クエリを実行

        Args:
            query: SQLクエリ
            params: パラメータ

        Returns:
            カーソル
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor

    def fetch_one(self, query: str, params: tuple[Any, ...] | None = None) -> sqlite3.Row | None:
        """
        1行取得

        Args:
            query: SQLクエリ
            params: パラメータ

        Returns:
            結果行
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple[Any, ...] | None = None) -> list[sqlite3.Row]:
        """
        全行取得

        Args:
            query: SQLクエリ
            params: パラメータ

        Returns:
            結果行のリスト
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()

    def exists(self) -> bool:
        """データベースファイルが存在するか確認"""
        return self.db_path.exists()
