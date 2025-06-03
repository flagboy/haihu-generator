"""
検証器ファクトリクラス
"""

from typing import Any

from .base_validator import BaseValidator
from .consistency_validator import ConsistencyValidator
from .content_validator import ContentValidator
from .logic_validator import LogicValidator
from .structure_validator import StructureValidator


class ValidatorFactory:
    """検証器ファクトリクラス"""

    def __init__(self, validation_config: dict[str, Any]):
        """
        初期化

        Args:
            validation_config: 検証設定
        """
        self.validation_config = validation_config
        self._validators = {}

    def get_validator(self, validator_type: str) -> BaseValidator:
        """
        検証器を取得

        Args:
            validator_type: 検証器タイプ

        Returns:
            検証器インスタンス
        """
        if validator_type not in self._validators:
            self._validators[validator_type] = self._create_validator(validator_type)

        return self._validators[validator_type]

    def _create_validator(self, validator_type: str) -> BaseValidator:
        """
        検証器を作成

        Args:
            validator_type: 検証器タイプ

        Returns:
            検証器インスタンス

        Raises:
            ValueError: 不明な検証器タイプ
        """
        if validator_type == "structure":
            return StructureValidator(self.validation_config)
        elif validator_type == "content":
            return ContentValidator(self.validation_config)
        elif validator_type == "logic":
            return LogicValidator(self.validation_config)
        elif validator_type == "consistency":
            return ConsistencyValidator(self.validation_config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")

    def get_all_validators(self) -> dict[str, BaseValidator]:
        """
        全ての検証器を取得

        Returns:
            検証器の辞書
        """
        validator_types = ["structure", "content", "logic", "consistency"]

        for validator_type in validator_types:
            if validator_type not in self._validators:
                self._validators[validator_type] = self._create_validator(validator_type)

        return self._validators

    def clear_cache(self):
        """検証器キャッシュをクリア"""
        self._validators.clear()
