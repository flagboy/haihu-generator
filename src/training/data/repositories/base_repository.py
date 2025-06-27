"""
基底リポジトリ

すべてのリポジトリの基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from ....utils.logger import LoggerMixin
from ..database import DatabaseConnection

T = TypeVar("T")


class BaseRepository[T](ABC, LoggerMixin):
    """基底リポジトリクラス"""

    def __init__(self, connection: DatabaseConnection):
        """
        初期化

        Args:
            connection: データベース接続
        """
        self.connection = connection

    @abstractmethod
    def create(self, entity: T) -> bool:
        """
        エンティティを作成

        Args:
            entity: エンティティ

        Returns:
            成功したかどうか
        """
        pass

    @abstractmethod
    def find_by_id(self, entity_id: str) -> T | None:
        """
        IDでエンティティを検索

        Args:
            entity_id: エンティティID

        Returns:
            エンティティまたはNone
        """
        pass

    @abstractmethod
    def find_all(self) -> list[T]:
        """
        すべてのエンティティを取得

        Returns:
            エンティティのリスト
        """
        pass

    @abstractmethod
    def update(self, entity: T) -> bool:
        """
        エンティティを更新

        Args:
            entity: エンティティ

        Returns:
            成功したかどうか
        """
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """
        エンティティを削除

        Args:
            entity_id: エンティティID

        Returns:
            成功したかどうか
        """
        pass

    def exists(self, entity_id: str) -> bool:
        """
        エンティティが存在するかチェック

        Args:
            entity_id: エンティティID

        Returns:
            存在するかどうか
        """
        return self.find_by_id(entity_id) is not None

    def count(self) -> int:
        """
        エンティティ数を取得

        Returns:
            エンティティ数
        """
        return len(self.find_all())

    def _execute_query(self, query: str, params: tuple[Any, ...] | None = None) -> bool:
        """
        クエリを実行

        Args:
            query: SQLクエリ
            params: パラメータ

        Returns:
            成功したかどうか
        """
        try:
            self.connection.execute(query, params)
            return True
        except Exception as e:
            self.logger.error(f"クエリ実行エラー: {e}")
            return False
