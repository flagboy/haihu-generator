"""
品質検証システム
牌譜の品質を検証し、信頼度スコアを計算
"""

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.config import ConfigManager
from ..utils.file_io import FileIOHelper
from ..utils.logger import get_logger
from ..utils.tile_definitions import TileDefinitions


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


@dataclass
class ValidationResult:
    """検証結果"""

    success: bool
    overall_score: float
    category_scores: dict[str, float]
    issues: list[ValidationIssue]
    statistics: dict[str, Any]
    recommendations: list[str]


class QualityValidator:
    """品質検証クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)
        self.tile_definitions = TileDefinitions()

        # 検証設定
        self.validation_config = self._load_validation_config()

        # 検証ルール
        self.validation_rules = self._setup_validation_rules()

        self.logger.info("QualityValidator initialized")

    def _load_validation_config(self) -> dict[str, Any]:
        """検証設定を読み込み"""
        # 設定ファイルから値を取得
        config = self.config.get_config()
        validation_config = config.get("validation", {})
        quality_thresholds = validation_config.get("quality_thresholds", {})

        return {
            "validation_level": ValidationLevel.STANDARD,
            "score_weights": {
                "structure": 0.25,
                "content": 0.30,
                "logic": 0.30,
                "consistency": 0.15,
            },
            "thresholds": {
                "pass_score": quality_thresholds.get("acceptable", 70.0),
                "good_score": quality_thresholds.get("good", 85.0),
                "excellent_score": quality_thresholds.get("excellent", 95.0),
            },
            "tile_count_limits": {
                "min_total_tiles": 50,
                "max_total_tiles": 200,
                "expected_hand_size": 13,
            },
            "penalties": validation_config.get("penalties", {}),
        }

    def _setup_validation_rules(self) -> dict[str, Any]:
        """検証ルールを設定"""
        return {
            "structure_rules": [
                "has_required_fields",
                "valid_json_structure",
                "proper_data_types",
                "consistent_formatting",
            ],
            "content_rules": [
                "valid_tile_names",
                "reasonable_tile_counts",
                "valid_player_actions",
                "proper_game_flow",
            ],
            "logic_rules": [
                "valid_hand_compositions",
                "legal_tile_movements",
                "consistent_game_state",
                "valid_scoring",
            ],
            "consistency_rules": [
                "temporal_consistency",
                "player_consistency",
                "tile_conservation",
                "action_sequence_validity",
            ],
        }

    def validate_record_file(
        self, record_path: str, validation_level: ValidationLevel = None
    ) -> ValidationResult:
        """
        牌譜ファイルを検証

        Args:
            record_path: 牌譜ファイルパス
            validation_level: 検証レベル

        Returns:
            検証結果
        """
        try:
            self.logger.info(f"Validating record file: {record_path}")

            if validation_level is None:
                validation_level = self.validation_config["validation_level"]

            # ファイル読み込み
            record_data = self._load_record_file(record_path)
            if record_data is None:
                return ValidationResult(
                    success=False,
                    overall_score=0.0,
                    category_scores={},
                    issues=[
                        ValidationIssue(
                            ValidationCategory.STRUCTURE, "error", "Failed to load record file"
                        )
                    ],
                    statistics={},
                    recommendations=["Check file format and encoding"],
                )

            # 検証実行
            return self.validate_record_data(record_data, validation_level)

        except Exception as e:
            self.logger.error(f"Record validation failed: {e}")
            return ValidationResult(
                success=False,
                overall_score=0.0,
                category_scores={},
                issues=[
                    ValidationIssue(
                        ValidationCategory.STRUCTURE, "error", f"Validation error: {str(e)}"
                    )
                ],
                statistics={},
                recommendations=["Check file integrity and format"],
            )

    def validate_record_data(
        self, record_data: dict[str, Any], validation_level: ValidationLevel = None
    ) -> ValidationResult:
        """
        牌譜データを検証

        Args:
            record_data: 牌譜データ
            validation_level: 検証レベル

        Returns:
            検証結果
        """
        if validation_level is None:
            validation_level = self.validation_config["validation_level"]

        issues = []
        category_scores = {}
        statistics = {}

        try:
            # 構造検証
            structure_score, structure_issues, structure_stats = self._validate_structure(
                record_data, validation_level
            )
            category_scores["structure"] = structure_score
            issues.extend(structure_issues)
            statistics.update(structure_stats)

            # 内容検証
            content_score, content_issues, content_stats = self._validate_content(
                record_data, validation_level
            )
            category_scores["content"] = content_score
            issues.extend(content_issues)
            statistics.update(content_stats)

            # 論理検証
            logic_score, logic_issues, logic_stats = self._validate_logic(
                record_data, validation_level
            )
            category_scores["logic"] = logic_score
            issues.extend(logic_issues)
            statistics.update(logic_stats)

            # 一貫性検証
            consistency_score, consistency_issues, consistency_stats = self._validate_consistency(
                record_data, validation_level
            )
            category_scores["consistency"] = consistency_score
            issues.extend(consistency_issues)
            statistics.update(consistency_stats)

            # 総合スコア計算
            overall_score = self._calculate_overall_score(category_scores)

            # 推奨事項生成
            recommendations = self._generate_recommendations(issues, category_scores)

            # 成功判定
            success = overall_score >= self.validation_config["thresholds"]["pass_score"]

            result = ValidationResult(
                success=success,
                overall_score=overall_score,
                category_scores=category_scores,
                issues=issues,
                statistics=statistics,
                recommendations=recommendations,
            )

            self.logger.info(
                f"Validation completed. Score: {overall_score:.1f}, Success: {success}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return ValidationResult(
                success=False,
                overall_score=0.0,
                category_scores=category_scores,
                issues=issues
                + [
                    ValidationIssue(
                        ValidationCategory.STRUCTURE, "error", f"Validation error: {str(e)}"
                    )
                ],
                statistics=statistics,
                recommendations=["Check data format and structure"],
            )

    def _load_record_file(self, record_path: str) -> dict[str, Any] | None:
        """牌譜ファイルを読み込み"""
        try:
            file_path = Path(record_path)

            if not file_path.exists():
                self.logger.error(f"Record file not found: {record_path}")
                return None

            if file_path.suffix.lower() == ".json":
                return FileIOHelper.load_json(file_path)
            else:
                # 天鳳JSON形式以外はサポート外
                self.logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load record file: {e}")
            return None

    def _validate_structure(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """構造検証"""
        issues = []
        statistics = {}
        score = 100.0

        try:
            # 必須フィールドチェック
            required_fields = ["game_info", "rounds"] if "game_info" in record_data else []

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
                    penalty = self.validation_config.get("penalties", {}).get("missing_field", 20)
                    score -= penalty

            # データ型チェック
            if "rounds" in record_data:
                if not isinstance(record_data["rounds"], list):
                    issues.append(
                        ValidationIssue(
                            ValidationCategory.STRUCTURE, "error", "rounds field must be a list"
                        )
                    )
                    penalty = self.validation_config.get("penalties", {}).get("wrong_type", 15)
                    score -= penalty
                else:
                    statistics["total_rounds"] = len(record_data["rounds"])

            # JSON構造の妥当性
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                structure_issues = self._check_json_structure(record_data)
                issues.extend(structure_issues)
                penalty = self.validation_config.get("penalties", {}).get("invalid_structure", 10)
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

    def _validate_content(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """内容検証"""
        issues = []
        statistics = {}
        score = 100.0

        try:
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
                    penalty = self.validation_config.get("penalties", {}).get(
                        "invalid_player_count", 20
                    )
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
                    penalty = self.validation_config.get("penalties", {}).get("no_rounds", 30)
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

    def _validate_logic(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """論理検証"""
        issues = []
        statistics = {}
        score = 100.0

        try:
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

    def _validate_consistency(
        self, record_data: dict[str, Any], validation_level: ValidationLevel
    ) -> tuple[float, list[ValidationIssue], dict[str, Any]]:
        """一貫性検証"""
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
        limits = self.validation_config["tile_count_limits"]
        if total_tiles < limits["min_total_tiles"]:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "warning", f"Too few tiles detected: {total_tiles}"
                )
            )
        elif total_tiles > limits["max_total_tiles"]:
            issues.append(
                ValidationIssue(
                    ValidationCategory.LOGIC, "warning", f"Too many tiles detected: {total_tiles}"
                )
            )

        return issues

    def _check_temporal_consistency(self, record_data: dict[str, Any]) -> list[ValidationIssue]:
        """時系列一貫性をチェック"""
        issues = []

        # タイムスタンプの順序チェック
        if "rounds" in record_data:
            prev_timestamp = 0
            for i, round_data in enumerate(record_data["rounds"]):
                if "timestamp" in round_data:
                    current_timestamp = round_data["timestamp"]
                    if current_timestamp < prev_timestamp:
                        issues.append(
                            ValidationIssue(
                                ValidationCategory.CONSISTENCY,
                                "error",
                                f"Timestamp inconsistency in round {i}",
                            )
                        )
                    prev_timestamp = current_timestamp

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
                        for action in round_data["actions"]:
                            if "player" in action:
                                player = action["player"]
                                if player not in expected_players:
                                    issues.append(
                                        ValidationIssue(
                                            ValidationCategory.CONSISTENCY,
                                            "error",
                                            f"Unknown player '{player}' in round {i}",
                                        )
                                    )

        return issues

    def _calculate_overall_score(self, category_scores: dict[str, float]) -> float:
        """総合スコアを計算"""
        weights = self.validation_config["score_weights"]
        total_score = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            if category in weights:
                weight = weights[category]
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(
        self, issues: list[ValidationIssue], category_scores: dict[str, float]
    ) -> list[str]:
        """推奨事項を生成"""
        recommendations = []

        # エラーに基づく推奨
        error_count = len([issue for issue in issues if issue.severity == "error"])
        warning_count = len([issue for issue in issues if issue.severity == "warning"])

        if error_count > 0:
            recommendations.append(f"{error_count}個のエラーを修正してください")

        if warning_count > 0:
            recommendations.append(f"{warning_count}個の警告を確認してください")

        # カテゴリ別推奨
        thresholds = self.validation_config["thresholds"]

        for category, score in category_scores.items():
            if score < thresholds["pass_score"]:
                recommendations.append(f"{category}の品質を改善してください（現在: {score:.1f}）")

        # 一般的な推奨
        if not recommendations:
            overall_score = self._calculate_overall_score(category_scores)
            if overall_score >= thresholds["excellent_score"]:
                recommendations.append("優秀な品質です")
            elif overall_score >= thresholds["good_score"]:
                recommendations.append("良好な品質です")
            else:
                recommendations.append("品質の向上を検討してください")

        return recommendations

    def _serialize_statistics(self, statistics: dict[str, Any]) -> dict[str, Any]:
        """統計情報をJSONシリアライズ可能な形式に変換"""
        from enum import Enum

        serialized = {}
        for key, value in statistics.items():
            if isinstance(value, set):
                serialized[key] = list(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_statistics(value)
            elif isinstance(value, Enum):
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    def export_validation_report(self, result: ValidationResult, output_path: str):
        """検証レポートをエクスポート"""
        try:
            report_data = {
                "validation_summary": {
                    "success": result.success,
                    "overall_score": result.overall_score,
                    "category_scores": result.category_scores,
                    "total_issues": len(result.issues),
                    "error_count": len([i for i in result.issues if i.severity == "error"]),
                    "warning_count": len([i for i in result.issues if i.severity == "warning"]),
                },
                "issues": [
                    {
                        "category": issue.category.value,
                        "severity": issue.severity,
                        "message": issue.message,
                        "location": issue.location,
                        "suggestion": issue.suggestion,
                    }
                    for issue in result.issues
                ],
                "statistics": self._serialize_statistics(result.statistics),
                "recommendations": result.recommendations,
                "validation_config": self._serialize_statistics(self.validation_config),
                "timestamp": time.time(),
            }

            FileIOHelper.save_json(report_data, output_path, pretty=True)

            self.logger.info(f"Validation report exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export validation report: {e}")
