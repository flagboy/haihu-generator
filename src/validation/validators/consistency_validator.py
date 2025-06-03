"""
一貫性検証クラス
"""

from typing import Any

from .base_validator import BaseValidator, ValidationCategory, ValidationIssue, ValidationLevel


class ConsistencyValidator(BaseValidator):
    """一貫性検証クラス"""

    def validate(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """
        一貫性検証を実行

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
            # 時系列一貫性
            temporal_issues = self._check_temporal_consistency(record_data)
            issues.extend(temporal_issues)
            score -= len(temporal_issues) * 3

            # プレイヤー一貫性
            player_issues = self._check_player_consistency(record_data)
            issues.extend(player_issues)
            score -= len(player_issues) * 2

            # スコア一貫性
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                score_issues = self._check_score_consistency(record_data)
                issues.extend(score_issues)
                score -= len(score_issues) * 4

            score = max(0, score)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    ValidationCategory.CONSISTENCY,
                    "error",
                    f"Consistency validation error: {str(e)}",
                )
            )
            score = 0

        return score, issues, statistics

    def _check_temporal_consistency(self, record_data: dict[str, Any]) -> list[ValidationIssue]:
        """時系列一貫性をチェック"""
        issues = []

        # タイムスタンプの順序チェック
        if "rounds" in record_data:
            prev_timestamp = 0
            for i, round_data in enumerate(record_data["rounds"]):
                if "timestamp" in round_data:
                    current_timestamp = round_data["timestamp"]
                    if isinstance(current_timestamp, int | float):
                        if current_timestamp < prev_timestamp:
                            issues.append(
                                ValidationIssue(
                                    ValidationCategory.CONSISTENCY,
                                    "error",
                                    f"Timestamp inconsistency in round {i}",
                                    location=f"round_{i}",
                                    suggestion="Ensure timestamps are in ascending order",
                                )
                            )
                        prev_timestamp = current_timestamp
                    else:
                        issues.append(
                            ValidationIssue(
                                ValidationCategory.CONSISTENCY,
                                "warning",
                                f"Invalid timestamp type in round {i}",
                                location=f"round_{i}",
                            )
                        )

        return issues

    def _check_player_consistency(self, record_data: dict[str, Any]) -> list[ValidationIssue]:
        """プレイヤー一貫性をチェック"""
        issues = []

        # プレイヤー名の一貫性チェック
        if "game_info" in record_data and "players" in record_data["game_info"]:
            expected_players = set(record_data["game_info"]["players"])

            if "rounds" in record_data:
                for i, round_data in enumerate(record_data["rounds"]):
                    if "actions" in round_data:
                        for j, action in enumerate(round_data["actions"]):
                            if "player" in action:
                                player = action["player"]
                                if player not in expected_players:
                                    issues.append(
                                        ValidationIssue(
                                            ValidationCategory.CONSISTENCY,
                                            "error",
                                            f"Unknown player '{player}' in round {i}",
                                            location=f"round_{i}_action_{j}",
                                            suggestion=f"Player must be one of: {', '.join(expected_players)}",
                                        )
                                    )

        return issues

    def _check_score_consistency(self, record_data: dict[str, Any]) -> list[ValidationIssue]:
        """スコア一貫性をチェック"""
        issues = []

        # 初期スコアと最終スコアの整合性チェック
        if "game_info" in record_data:
            initial_scores = record_data["game_info"].get("initial_scores", {})
            final_scores = record_data["game_info"].get("final_scores", {})

            if initial_scores and final_scores:
                # スコアの合計チェック
                initial_total = sum(initial_scores.values())
                final_total = sum(final_scores.values())

                # 麻雀では通常、スコアの合計は変わらない
                if abs(initial_total - final_total) > 1:  # 丸め誤差を考慮
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONSISTENCY,
                            "warning",
                            f"Score total inconsistency: initial={initial_total}, final={final_total}",
                            suggestion="Check score calculations",
                        )
                    )

        return issues

    def check_action_player_consistency(
        self, actions: list[dict[str, Any]], expected_players: list[str]
    ) -> list[ValidationIssue]:
        """アクションのプレイヤー一貫性をチェック"""
        issues = []
        expected_set = set(expected_players)

        for i, action in enumerate(actions):
            if "player" in action:
                player = action["player"]
                if player not in expected_set:
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONSISTENCY,
                            "error",
                            f"Unknown player '{player}' in action {i}",
                            location=f"action_{i}",
                        )
                    )

        return issues
