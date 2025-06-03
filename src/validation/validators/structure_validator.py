"""
構造検証クラス
"""

import json
from typing import Any

from .base_validator import BaseValidator, ValidationCategory, ValidationIssue, ValidationLevel


class StructureValidator(BaseValidator):
    """構造検証クラス"""

    def validate(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """
        構造検証を実行

        Args:
            record_data: 牌譜データ
            validation_level: 検証レベル

        Returns:
            (スコア, 問題リスト, 統計情報)のタプル
        """
        issues = []
        statistics = {}
        score = 100.0

        try:
            # 必須フィールドチェック
            required_fields = ["game_info", "rounds"]

            for field in required_fields:
                if field not in record_data:
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.STRUCTURE,
                            "error",
                            f"Missing required field: {field}",
                            suggestion=f"Add {field} field to record data",
                        )
                    )
                    penalty = self._calculate_penalty("missing_field", 30)
                    score -= penalty

            # 完全に無効な構造（必須フィールドが全くない）の場合、大幅減点
            if not any(field in record_data for field in required_fields):
                issues.append(
                    ValidationIssue(
                        ValidationCategory.STRUCTURE,
                        "critical",
                        "Invalid record structure: no required fields found",
                        suggestion="Ensure the record contains at least 'game_info' or 'rounds' fields",
                    )
                )
                score -= 70  # 大幅減点（必須フィールドがない場合）

            # データ型チェック
            if "rounds" in record_data:
                if not isinstance(record_data["rounds"], list):
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.STRUCTURE, "error", "rounds field must be a list"
                        )
                    )
                    penalty = self._calculate_penalty("wrong_type", 15)
                    score -= penalty
                else:
                    statistics["total_rounds"] = len(record_data["rounds"])

            # JSON構造の妥当性
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                structure_issues = self._check_json_structure(record_data)
                issues.extend(structure_issues)
                penalty = self._calculate_penalty("invalid_structure", 10)
                score -= len(structure_issues) * penalty

            score = max(0, score)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    ValidationCategory.STRUCTURE, "error", f"Structure validation error: {str(e)}"
                )
            )
            score = 0

        return score, issues, statistics

    def _check_json_structure(self, data: dict[str, Any]) -> list[ValidationIssue]:
        """JSON構造をチェック"""
        issues = []

        # 循環参照チェック
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            issues.append(
                ValidationIssue(
                    ValidationCategory.STRUCTURE, "error", f"JSON serialization error: {str(e)}"
                )
            )

        return issues

    def check_required_fields(
        self, data: dict[str, Any], required_fields: list[str]
    ) -> list[ValidationIssue]:
        """必須フィールドをチェック"""
        issues = []

        for field in required_fields:
            if field not in data:
                issues.append(
                    ValidationIssue(
                        ValidationCategory.STRUCTURE,
                        "error",
                        f"Missing required field: {field}",
                        suggestion=f"Add {field} field to data",
                    )
                )

        return issues
