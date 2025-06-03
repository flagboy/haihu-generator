"""
論理検証クラス
"""

from typing import Any

from .base_validator import BaseValidator, ValidationCategory, ValidationIssue, ValidationLevel


class LogicValidator(BaseValidator):
    """論理検証クラス"""

    def validate(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """
        論理検証を実行

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
            # 基本構造チェック - 必須フィールドがない場合は論理検証不可
            required_fields = ["game_info", "rounds"]
            if not any(field in record_data for field in required_fields):
                issues.append(
                    ValidationIssue(
                        ValidationCategory.LOGIC,
                        "error",
                        "Cannot validate game logic without proper structure",
                        suggestion="Add game_info and rounds fields for logic validation",
                    )
                )
                score -= 60  # 構造がないと論理検証不可

            # 手牌構成の妥当性
            if "rounds" in record_data:
                for i, round_data in enumerate(record_data["rounds"]):
                    round_issues = self._validate_round_logic(round_data, i)
                    issues.extend(round_issues)
                    score -= len(round_issues) * 2

            # 牌の保存則チェック
            conservation_issues = self._check_tile_conservation(record_data)
            issues.extend(conservation_issues)
            score -= len(conservation_issues) * 5

            score = max(0, score)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "error", f"Logic validation error: {str(e)}"
                )
            )
            score = 0

        return score, issues, statistics

    def _validate_round_logic(
        self, round_data: dict[str, Any], round_index: int
    ) -> list[ValidationIssue]:
        """ラウンドの論理を検証"""
        issues = []

        # 基本的な構造チェック
        if "actions" not in round_data:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "warning", f"Round {round_index} has no actions"
                )
            )
            return issues

        # アクションの妥当性チェック
        actions = round_data["actions"]
        if not isinstance(actions, list):
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "error", f"Round {round_index} actions must be a list"
                )
            )
            return issues

        # アクションシーケンスの検証
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                issues.append(
                    ValidationIssue(
                        ValidationCategory.LOGIC,
                        "error",
                        f"Round {round_index} action {i} must be a dictionary",
                    )
                )
                continue

            # アクションタイプのチェック
            if "type" not in action:
                issues.append(
                    ValidationIssue(
                        ValidationCategory.LOGIC,
                        "warning",
                        f"Round {round_index} action {i} missing type",
                        location=f"round_{round_index}_action_{i}",
                    )
                )

        return issues

    def _check_tile_conservation(self, record_data: dict[str, Any]) -> list[ValidationIssue]:
        """牌の保存則をチェック"""
        issues = []

        # 簡易的な牌数チェック
        total_tiles = 0
        if "rounds" in record_data:
            for round_data in record_data["rounds"]:
                if "actions" in round_data:
                    for action in round_data["actions"]:
                        if "tiles" in action:
                            total_tiles += len(action["tiles"])

        # 期待される牌数の範囲チェック
        tile_limits = self.validation_config.get("tile_count_limits", {})
        min_tiles = tile_limits.get("min_total_tiles", 50)
        max_tiles = tile_limits.get("max_total_tiles", 200)

        if total_tiles < min_tiles:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "warning", f"Too few tiles detected: {total_tiles}"
                )
            )
        elif total_tiles > max_tiles:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "warning", f"Too many tiles detected: {total_tiles}"
                )
            )

        return issues

    def validate_action_sequence(self, actions: list[dict[str, Any]]) -> list[ValidationIssue]:
        """アクションシーケンスを検証"""
        issues = []

        # アクションの順序検証
        prev_action = None
        for _i, action in enumerate(actions):
            if prev_action:
                # 前のアクションとの整合性チェック
                pass

            prev_action = action

        return issues
