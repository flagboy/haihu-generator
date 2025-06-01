"""
信頼度計算モジュール
検出・分類・牌譜生成の信頼度を総合的に評価
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..utils.logger import get_logger


class ConfidenceLevel(Enum):
    """信頼度レベル"""

    VERY_LOW = "very_low"  # 0-20%
    LOW = "low"  # 20-40%
    MEDIUM = "medium"  # 40-60%
    HIGH = "high"  # 60-80%
    VERY_HIGH = "very_high"  # 80-100%


@dataclass
class ConfidenceScore:
    """信頼度スコア"""

    overall_score: float
    detection_score: float
    classification_score: float
    consistency_score: float
    temporal_score: float
    level: ConfidenceLevel
    factors: dict[str, float]
    recommendations: list[str]


class ConfidenceCalculator:
    """信頼度計算クラス"""

    def __init__(self):
        """初期化"""
        self.logger = get_logger(__name__)

        # 信頼度計算の重み
        self.weights = {
            "detection": 0.25,  # 検出精度の重み
            "classification": 0.30,  # 分類精度の重み
            "consistency": 0.25,  # 一貫性の重み
            "temporal": 0.20,  # 時系列一貫性の重み
        }

        # 閾値設定
        self.thresholds = {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "very_high": 1.0,
        }

        self.logger.info("ConfidenceCalculator initialized")

    def calculate_overall_confidence(
        self,
        detection_results: list[Any],
        classification_results: list[Any],
        game_state_data: dict[str, Any] | None = None,
    ) -> ConfidenceScore:
        """
        総合信頼度を計算

        Args:
            detection_results: 検出結果のリスト
            classification_results: 分類結果のリスト
            game_state_data: ゲーム状態データ（オプション）

        Returns:
            信頼度スコア
        """
        try:
            # 各要素の信頼度を計算
            detection_score = self._calculate_detection_confidence(detection_results)
            classification_score = self._calculate_classification_confidence(classification_results)
            consistency_score = self._calculate_consistency_confidence(
                detection_results, classification_results
            )
            temporal_score = self._calculate_temporal_confidence(detection_results, game_state_data)

            # 重み付き総合スコア
            overall_score = (
                detection_score * self.weights["detection"]
                + classification_score * self.weights["classification"]
                + consistency_score * self.weights["consistency"]
                + temporal_score * self.weights["temporal"]
            )

            # 信頼度レベルを決定
            level = self._determine_confidence_level(overall_score)

            # 影響要因を分析
            factors = {
                "detection_quality": detection_score,
                "classification_accuracy": classification_score,
                "data_consistency": consistency_score,
                "temporal_stability": temporal_score,
                "sample_size": len(detection_results),
                "variance": self._calculate_score_variance(
                    [detection_score, classification_score, consistency_score, temporal_score]
                ),
            }

            # 推奨事項を生成
            recommendations = self._generate_confidence_recommendations(
                detection_score, classification_score, consistency_score, temporal_score
            )

            confidence_score = ConfidenceScore(
                overall_score=overall_score,
                detection_score=detection_score,
                classification_score=classification_score,
                consistency_score=consistency_score,
                temporal_score=temporal_score,
                level=level,
                factors=factors,
                recommendations=recommendations,
            )

            self.logger.info(f"Overall confidence calculated: {overall_score:.3f} ({level.value})")
            return confidence_score

        except Exception as e:
            self.logger.error(f"Failed to calculate overall confidence: {e}")
            return ConfidenceScore(
                overall_score=0.0,
                detection_score=0.0,
                classification_score=0.0,
                consistency_score=0.0,
                temporal_score=0.0,
                level=ConfidenceLevel.VERY_LOW,
                factors={"error": str(e)},
                recommendations=["Error in confidence calculation"],
            )

    def _calculate_detection_confidence(self, detection_results: list[Any]) -> float:
        """検出信頼度を計算"""
        if not detection_results:
            return 0.0

        try:
            confidences = []

            for result in detection_results:
                if hasattr(result, "detections"):
                    # 検出結果から信頼度を抽出
                    for detection in result.detections:
                        if hasattr(detection, "confidence"):
                            confidences.append(detection.confidence)
                elif hasattr(result, "confidence"):
                    confidences.append(result.confidence)

            if not confidences:
                return 0.0

            # 統計的指標を計算
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            min_confidence = np.min(confidences)

            # 信頼度スコアを計算（平均値をベースに、分散と最小値で調整）
            score = mean_confidence

            # 分散が大きい場合は信頼度を下げる
            if std_confidence > 0.2:
                score *= 1 - std_confidence * 0.5

            # 最小値が低い場合は信頼度を下げる
            if min_confidence < 0.3:
                score *= 0.5 + min_confidence * 0.5

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.error(f"Detection confidence calculation failed: {e}")
            return 0.0

    def _calculate_classification_confidence(self, classification_results: list[Any]) -> float:
        """分類信頼度を計算"""
        if not classification_results:
            return 0.0

        try:
            confidences = []

            for result in classification_results:
                if hasattr(result, "classifications"):
                    # 分類結果から信頼度を抽出
                    for _, classification in result.classifications:
                        if hasattr(classification, "confidence"):
                            confidences.append(classification.confidence)
                elif hasattr(result, "confidence"):
                    confidences.append(result.confidence)

            if not confidences:
                return 0.0

            # 統計的指標を計算
            mean_confidence = np.mean(confidences)
            median_confidence = np.median(confidences)
            q25 = np.percentile(confidences, 25)

            # 分類信頼度スコア（平均値と中央値の調和平均をベースに）
            if mean_confidence + median_confidence > 0:
                harmonic_mean = (
                    2 * mean_confidence * median_confidence / (mean_confidence + median_confidence)
                )
            else:
                harmonic_mean = 0.0

            # 第1四分位数で調整（低信頼度の結果が多い場合は全体の信頼度を下げる）
            score = harmonic_mean * (0.5 + q25 * 0.5)

            return max(0.0, min(1.0, score))

        except Exception as e:
            self.logger.error(f"Classification confidence calculation failed: {e}")
            return 0.0

    def _calculate_consistency_confidence(
        self, detection_results: list[Any], classification_results: list[Any]
    ) -> float:
        """一貫性信頼度を計算"""
        try:
            if not detection_results or not classification_results:
                return 0.0

            # 検出数と分類数の一貫性
            detection_counts = []
            classification_counts = []

            for result in detection_results:
                if hasattr(result, "detections"):
                    detection_counts.append(len(result.detections))

            for result in classification_results:
                if hasattr(result, "classifications"):
                    classification_counts.append(len(result.classifications))

            if not detection_counts or not classification_counts:
                return 0.0

            # 数の一貫性を評価
            detection_mean = np.mean(detection_counts)
            classification_mean = np.mean(classification_counts)

            if detection_mean > 0:
                count_consistency = min(classification_mean / detection_mean, 1.0)
            else:
                count_consistency = 0.0

            # 分散の一貫性を評価
            detection_std = np.std(detection_counts) if len(detection_counts) > 1 else 0
            classification_std = (
                np.std(classification_counts) if len(classification_counts) > 1 else 0
            )

            # 分散が小さいほど一貫性が高い
            variance_consistency = 1.0 - min(detection_std + classification_std, 1.0) * 0.1

            # 総合一貫性スコア
            consistency_score = count_consistency * 0.7 + variance_consistency * 0.3

            return max(0.0, min(1.0, consistency_score))

        except Exception as e:
            self.logger.error(f"Consistency confidence calculation failed: {e}")
            return 0.0

    def _calculate_temporal_confidence(
        self, detection_results: list[Any], game_state_data: dict[str, Any] | None
    ) -> float:
        """時系列信頼度を計算"""
        try:
            if not detection_results:
                return 0.0

            # フレーム間の信頼度変化を分析
            frame_confidences = []

            for result in detection_results:
                if hasattr(result, "confidence_scores"):
                    combined_conf = result.confidence_scores.get("combined_confidence", 0.0)
                    frame_confidences.append(combined_conf)
                elif hasattr(result, "processing_time"):
                    # 処理時間から間接的に信頼度を推定
                    time_based_conf = min(1.0, 1.0 / (result.processing_time + 0.1))
                    frame_confidences.append(time_based_conf)

            if not frame_confidences:
                return 0.0

            # 時系列の安定性を評価
            if len(frame_confidences) > 1:
                # 変化率の標準偏差（小さいほど安定）
                changes = np.diff(frame_confidences)
                change_std = np.std(changes)
                stability_score = 1.0 - min(change_std, 1.0)
            else:
                stability_score = 1.0

            # 平均信頼度
            mean_confidence = np.mean(frame_confidences)

            # 時系列信頼度スコア
            temporal_score = mean_confidence * 0.6 + stability_score * 0.4

            return max(0.0, min(1.0, temporal_score))

        except Exception as e:
            self.logger.error(f"Temporal confidence calculation failed: {e}")
            return 0.0

    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """信頼度レベルを決定"""
        if score >= self.thresholds["high"]:
            return ConfidenceLevel.VERY_HIGH
        elif score >= self.thresholds["medium"]:
            return ConfidenceLevel.HIGH
        elif score >= self.thresholds["low"]:
            return ConfidenceLevel.MEDIUM
        elif score >= self.thresholds["very_low"]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_score_variance(self, scores: list[float]) -> float:
        """スコアの分散を計算"""
        try:
            return float(np.var(scores)) if scores else 0.0
        except Exception:
            return 0.0

    def _generate_confidence_recommendations(
        self,
        detection_score: float,
        classification_score: float,
        consistency_score: float,
        temporal_score: float,
    ) -> list[str]:
        """信頼度に基づく推奨事項を生成"""
        recommendations = []

        # 検出スコアに基づく推奨
        if detection_score < 0.6:
            recommendations.append(
                "検出精度が低いです。検出モデルの調整や閾値の見直しを検討してください"
            )

        # 分類スコアに基づく推奨
        if classification_score < 0.6:
            recommendations.append(
                "分類精度が低いです。分類モデルの再訓練や前処理の改善を検討してください"
            )

        # 一貫性スコアに基づく推奨
        if consistency_score < 0.6:
            recommendations.append(
                "データの一貫性が低いです。入力データの品質確認や処理パラメータの調整を検討してください"
            )

        # 時系列スコアに基づく推奨
        if temporal_score < 0.6:
            recommendations.append(
                "時系列の安定性が低いです。フレーム抽出間隔の調整や平滑化処理を検討してください"
            )

        # 総合的な推奨
        all_scores = [detection_score, classification_score, consistency_score, temporal_score]
        avg_score = np.mean(all_scores)

        if avg_score >= 0.8:
            recommendations.append("高い信頼度です。現在の設定を維持してください")
        elif avg_score >= 0.6:
            recommendations.append(
                "良好な信頼度です。さらなる改善のため細かな調整を検討してください"
            )
        else:
            recommendations.append("信頼度が低いです。システム全体の見直しが必要です")

        return recommendations if recommendations else ["信頼度評価が完了しました"]

    def calculate_frame_confidence(
        self, detection_result: Any, classification_result: Any
    ) -> float:
        """単一フレームの信頼度を計算"""
        try:
            detection_conf = 0.0
            classification_conf = 0.0

            # 検出信頼度
            if hasattr(detection_result, "detections") and detection_result.detections:
                detection_confidences = [
                    d.confidence for d in detection_result.detections if hasattr(d, "confidence")
                ]
                detection_conf = np.mean(detection_confidences) if detection_confidences else 0.0

            # 分類信頼度
            if hasattr(classification_result, "classifications"):
                if classification_result.classifications:
                    classification_confidences = [
                        c[1].confidence
                        for c in classification_result.classifications
                        if hasattr(c[1], "confidence")
                    ]
                    classification_conf = (
                        np.mean(classification_confidences) if classification_confidences else 0.0
                    )

            # フレーム信頼度（検出と分類の調和平均）
            if detection_conf + classification_conf > 0:
                frame_confidence = (
                    2
                    * detection_conf
                    * classification_conf
                    / (detection_conf + classification_conf)
                )
            else:
                frame_confidence = 0.0

            return max(0.0, min(1.0, frame_confidence))

        except Exception as e:
            self.logger.error(f"Frame confidence calculation failed: {e}")
            return 0.0

    def export_confidence_report(self, confidence_score: ConfidenceScore, output_path: str):
        """信頼度レポートをエクスポート"""
        try:
            import json
            import time

            report_data = {
                "confidence_summary": {
                    "overall_score": confidence_score.overall_score,
                    "level": confidence_score.level.value,
                    "component_scores": {
                        "detection": confidence_score.detection_score,
                        "classification": confidence_score.classification_score,
                        "consistency": confidence_score.consistency_score,
                        "temporal": confidence_score.temporal_score,
                    },
                },
                "factors": confidence_score.factors,
                "recommendations": confidence_score.recommendations,
                "calculation_weights": self.weights,
                "thresholds": self.thresholds,
                "export_timestamp": time.time(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Confidence report exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export confidence report: {e}")
