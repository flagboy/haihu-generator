"""
検証基底クラス
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...utils.logger import LoggerMixin


class ValidationLevel(Enum):
    """検証レベル"""

    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class ValidationCategory(Enum):
    """検証カテゴリ"""

    STRUCTURE = "structure"
    CONTENT = "content"
    LOGIC = "logic"
    CONSISTENCY = "consistency"


@dataclass
class ValidationIssue:
    """検証問題"""

    category: ValidationCategory
    severity: str  # "error", "warning", "info"
    message: str
    location: str | None = None
    suggestion: str | None = None


class BaseValidator(ABC, LoggerMixin):
    """検証基底クラス"""

    def __init__(self, validation_config: dict[str, Any]):
        """
        初期化

        Args:
            validation_config: 検証設定
        """
        self.validation_config = validation_config
        self.logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def validate(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """
        検証を実行

        Args:
            record_data: 牌譜データ
            validation_level: 検証レベル

        Returns:
            (スコア, 問題リスト, 統計情報)のタプル
        """
        pass

    def _calculate_penalty(self, penalty_key: str, default: float = 10.0) -> float:
        """
        ペナルティを計算

        Args:
            penalty_key: ペナルティキー
            default: デフォルト値

        Returns:
            ペナルティ値
        """
        penalties = self.validation_config.get("penalties", {})
        return penalties.get(penalty_key, default)
