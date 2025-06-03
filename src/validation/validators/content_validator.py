"""
内容検証クラス
"""

from typing import Any

from ...utils.tile_definitions import TileDefinitions
from .base_validator import BaseValidator, ValidationCategory, ValidationIssue, ValidationLevel


class ContentValidator(BaseValidator):
    """内容検証クラス"""

    def __init__(self, validation_config: dict[str, Any]):
        """
        初期化

        Args:
            validation_config: 検証設定
        """
        super().__init__(validation_config)
        self.tile_definitions = TileDefinitions()

    def validate(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """
        内容検証を実行

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
            # 基本構造チェック - 必須フィールドがない場合は大幅減点
            required_fields = ["game_info", "rounds"]
            if not any(field in record_data for field in required_fields):
                issues.append(
                    ValidationIssue(
                        ValidationCategory.CONTENT,
                        "error",
                        "No valid game content found due to missing structure",
                        suggestion="Add proper game structure with game_info and rounds",
                    )
                )
                score -= 80  # 構造がないとコンテンツも無効

            # 牌名の妥当性チェック
            tile_issues, tile_stats = self._validate_tiles(record_data)
            issues.extend(tile_issues)
            statistics.update(tile_stats)
            score -= len(tile_issues) * 3

            # プレイヤー情報チェック
            if "game_info" in record_data and "players" in record_data["game_info"]:
                players = record_data["game_info"]["players"]
                if len(players) != 4:
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONTENT,
                            "warning",
                            f"Expected 4 players, found {len(players)}",
                        )
                    )
                    penalty = self._calculate_penalty("invalid_player_count", 20)
                    score -= penalty

                statistics["player_count"] = len(players)

            # ラウンド数チェック
            if "rounds" in record_data:
                rounds = record_data["rounds"]
                if len(rounds) == 0:
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONTENT, "error", "No rounds found in record"
                        )
                    )
                    penalty = self._calculate_penalty("no_rounds", 30)
                    score -= penalty
                elif len(rounds) > 20:
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONTENT,
                            "warning",
                            f"Unusually high number of rounds: {len(rounds)}",
                        )
                    )
                    score -= 5

            score = max(0, score)

        except Exception as e:
            issues.append(
                ValidationIssue(
                    ValidationCategory.CONTENT, "error", f"Content validation error: {str(e)}"
                )
            )
            score = 0

        return score, issues, statistics

    def _validate_tiles(
        self, record_data: dict[str, Any]
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """牌の妥当性を検証"""
        issues = []
        statistics = {"total_tiles": 0, "unique_tiles": set(), "invalid_tiles": []}

        def check_tile_list(tiles, location):
            for tile in tiles:
                statistics["total_tiles"] += 1
                statistics["unique_tiles"].add(tile)

                if not self.tile_definitions.is_valid_tile(tile):
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.CONTENT,
                            "error",
                            f"Invalid tile name: {tile}",
                            location=location,
                        )
                    )
                    statistics["invalid_tiles"].append(tile)

        # ラウンドデータの牌をチェック
        if "rounds" in record_data:
            for i, round_data in enumerate(record_data["rounds"]):
                if "actions" in round_data:
                    for j, action in enumerate(round_data["actions"]):
                        if "tiles" in action:
                            check_tile_list(action["tiles"], f"round_{i}_action_{j}")

        statistics["unique_tile_count"] = len(statistics["unique_tiles"])
        return issues, statistics

    def validate_player_info(self, game_info: dict[str, Any]) -> list[ValidationIssue]:
        """プレイヤー情報を検証"""
        issues = []

        if "players" not in game_info:
            issues.append(
                ValidationIssue(
                    ValidationCategory.CONTENT,
                    "error",
                    "Missing players information",
                    suggestion="Add players list to game_info",
                )
            )
            return issues

        players = game_info["players"]
        if not isinstance(players, list):
            issues.append(
                ValidationIssue(
                    ValidationCategory.CONTENT,
                    "error",
                    "Players field must be a list",
                )
            )
        elif len(players) != 4:
            issues.append(
                ValidationIssue(
                    ValidationCategory.CONTENT,
                    "warning",
                    f"Expected 4 players, found {len(players)}",
                )
            )

        return issues
