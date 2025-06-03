"""
品質検証システム（リファクタリング版）
牌譜の品質を検証し、信頼度スコアを計算
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.config import ConfigManager
from ..utils.file_io import FileIOHelper
from ..utils.logger import get_logger
from .validators import RecommendationEngine, ValidatorFactory
from .validators.base_validator import ValidationCategory, ValidationIssue, ValidationLevel


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
    """品質検証クラス（リファクタリング版）"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # 検証設定
        self.validation_config = self._load_validation_config()

        # 検証器ファクトリ
        self.validator_factory = ValidatorFactory(self.validation_config)

        # 推奨事項エンジン
        self.recommendation_engine = RecommendationEngine(self.validation_config)

        # 後方互換性のため検証ルールを属性として保持
        self.validation_rules = self.get_validation_rules()

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
            # 各カテゴリの検証を実行
            validators = self.validator_factory.get_all_validators()

            for category_name, validator in validators.items():
                score, category_issues, category_stats = validator.validate(
                    record_data, validation_level
                )

                category_scores[category_name] = score
                issues.extend(category_issues)
                statistics.update(category_stats)

            # 総合スコア計算
            overall_score = self._calculate_overall_score(category_scores)

            # 推奨事項生成
            recommendations = self.recommendation_engine.generate_recommendations(
                issues, category_scores, overall_score
            )

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

    def get_validation_rules(self) -> dict[str, Any]:
        """検証ルールを取得"""
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


# 後方互換性のためのエクスポート
__all__ = [
    "QualityValidator",
    "ValidationResult",
    "ValidationLevel",
    "ValidationCategory",
    "ValidationIssue",
]
