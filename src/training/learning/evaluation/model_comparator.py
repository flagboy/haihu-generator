"""
モデル比較専用クラス
"""

from datetime import datetime
from typing import Any

import numpy as np

from ....utils.logger import LoggerMixin
from .evaluator_base import EvaluationResult


class ModelComparator(LoggerMixin):
    """モデル比較専用クラス"""

    def __init__(self):
        """初期化"""
        self.logger.info("ModelComparator初期化完了")

    def compare_models(self, results: dict[str, EvaluationResult]) -> dict[str, Any]:
        """
        複数モデルを比較

        Args:
            results: モデル名 -> 評価結果の辞書

        Returns:
            比較結果
        """
        if not results:
            return {}

        comparison = {
            "model_count": len(results),
            "comparison_date": datetime.now().isoformat(),
            "overall_comparison": self._compare_overall_metrics(results),
            "per_class_comparison": self._compare_per_class_metrics(results),
            "rankings": self._calculate_rankings(results),
            "statistical_analysis": self._statistical_analysis(results),
            "recommendations": self._generate_recommendations(results),
        }

        return comparison

    def _compare_overall_metrics(
        self, results: dict[str, EvaluationResult]
    ) -> dict[str, dict[str, float]]:
        """
        全体的なメトリクスを比較

        Args:
            results: モデル評価結果

        Returns:
            メトリクス比較結果
        """
        metrics_comparison = {
            "accuracy": {},
            "precision_weighted": {},
            "recall_weighted": {},
            "f1_weighted": {},
            "evaluation_time": {},
        }

        for model_name, result in results.items():
            metrics_comparison["accuracy"][model_name] = result.accuracy
            metrics_comparison["precision_weighted"][model_name] = result.precision.get(
                "weighted_avg", 0
            )
            metrics_comparison["recall_weighted"][model_name] = result.recall.get("weighted_avg", 0)
            metrics_comparison["f1_weighted"][model_name] = result.f1_score.get("weighted_avg", 0)
            metrics_comparison["evaluation_time"][model_name] = result.evaluation_time

        return metrics_comparison

    def _compare_per_class_metrics(
        self, results: dict[str, EvaluationResult]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        クラスごとのメトリクスを比較

        Args:
            results: モデル評価結果

        Returns:
            クラスごとの比較結果
        """
        # 全クラスを収集
        all_classes = set()
        for result in results.values():
            all_classes.update(
                [cls for cls in result.per_class_accuracy if not cls.endswith("_avg")]
            )

        per_class_comparison = {}

        for class_name in sorted(all_classes):
            per_class_comparison[class_name] = {
                "accuracy": {},
                "precision": {},
                "recall": {},
                "f1_score": {},
            }

            for model_name, result in results.items():
                per_class_comparison[class_name]["accuracy"][model_name] = (
                    result.per_class_accuracy.get(class_name, 0)
                )
                per_class_comparison[class_name]["precision"][model_name] = result.precision.get(
                    class_name, 0
                )
                per_class_comparison[class_name]["recall"][model_name] = result.recall.get(
                    class_name, 0
                )
                per_class_comparison[class_name]["f1_score"][model_name] = result.f1_score.get(
                    class_name, 0
                )

        return per_class_comparison

    def _calculate_rankings(
        self, results: dict[str, EvaluationResult]
    ) -> dict[str, list[tuple[str, float]]]:
        """
        各メトリクスでのランキングを計算

        Args:
            results: モデル評価結果

        Returns:
            ランキング結果
        """
        rankings = {
            "accuracy": [],
            "precision_weighted": [],
            "recall_weighted": [],
            "f1_weighted": [],
            "speed": [],  # 評価時間（昇順）
        }

        # 精度ランキング
        accuracy_scores = [(name, result.accuracy) for name, result in results.items()]
        rankings["accuracy"] = sorted(accuracy_scores, key=lambda x: x[1], reverse=True)

        # 適合率ランキング
        precision_scores = [
            (name, result.precision.get("weighted_avg", 0)) for name, result in results.items()
        ]
        rankings["precision_weighted"] = sorted(precision_scores, key=lambda x: x[1], reverse=True)

        # 再現率ランキング
        recall_scores = [
            (name, result.recall.get("weighted_avg", 0)) for name, result in results.items()
        ]
        rankings["recall_weighted"] = sorted(recall_scores, key=lambda x: x[1], reverse=True)

        # F1スコアランキング
        f1_scores = [
            (name, result.f1_score.get("weighted_avg", 0)) for name, result in results.items()
        ]
        rankings["f1_weighted"] = sorted(f1_scores, key=lambda x: x[1], reverse=True)

        # 速度ランキング（昇順）
        speed_scores = [(name, result.evaluation_time) for name, result in results.items()]
        rankings["speed"] = sorted(speed_scores, key=lambda x: x[1])

        return rankings

    def _statistical_analysis(self, results: dict[str, EvaluationResult]) -> dict[str, Any]:
        """
        統計的分析を実行

        Args:
            results: モデル評価結果

        Returns:
            統計分析結果
        """
        # 各メトリクスの値を収集
        accuracies = [r.accuracy for r in results.values()]
        precisions = [r.precision.get("weighted_avg", 0) for r in results.values()]
        recalls = [r.recall.get("weighted_avg", 0) for r in results.values()]
        f1_scores = [r.f1_score.get("weighted_avg", 0) for r in results.values()]
        times = [r.evaluation_time for r in results.values()]

        analysis = {
            "accuracy_stats": self._calculate_stats(accuracies),
            "precision_stats": self._calculate_stats(precisions),
            "recall_stats": self._calculate_stats(recalls),
            "f1_stats": self._calculate_stats(f1_scores),
            "time_stats": self._calculate_stats(times),
        }

        # 相関分析
        if len(results) > 2:
            analysis["correlations"] = {
                "accuracy_vs_time": np.corrcoef(accuracies, times)[0, 1],
                "f1_vs_time": np.corrcoef(f1_scores, times)[0, 1],
            }

        return analysis

    def _calculate_stats(self, values: list[float]) -> dict[str, float]:
        """
        統計値を計算

        Args:
            values: 値のリスト

        Returns:
            統計値
        """
        if not values:
            return {}

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }

    def _generate_recommendations(self, results: dict[str, EvaluationResult]) -> list[str]:
        """
        推奨事項を生成

        Args:
            results: モデル評価結果

        Returns:
            推奨事項リスト
        """
        recommendations = []

        rankings = self._calculate_rankings(results)

        # 最高精度モデル
        best_accuracy = rankings["accuracy"][0]
        recommendations.append(f"最高精度: {best_accuracy[0]} (精度: {best_accuracy[1]:.4f})")

        # 最高F1スコアモデル
        best_f1 = rankings["f1_weighted"][0]
        if best_f1[0] != best_accuracy[0]:
            recommendations.append(f"最高F1スコア: {best_f1[0]} (F1: {best_f1[1]:.4f})")

        # 最速モデル
        fastest = rankings["speed"][0]
        recommendations.append(f"最速モデル: {fastest[0]} (評価時間: {fastest[1]:.2f}秒)")

        # バランス型モデル（精度と速度のバランス）
        # 正規化してスコアを計算
        accuracy_scores = {name: result.accuracy for name, result in results.items()}
        time_scores = {name: result.evaluation_time for name, result in results.items()}

        max_acc = max(accuracy_scores.values())
        min_time = min(time_scores.values())
        max_time = max(time_scores.values())

        balance_scores = {}
        for name in results:
            # 精度スコア（0-1）
            acc_score = accuracy_scores[name] / max_acc if max_acc > 0 else 0
            # 時間スコア（0-1、逆順）
            time_score = (
                1 - ((time_scores[name] - min_time) / (max_time - min_time))
                if max_time > min_time
                else 1
            )
            # バランススコア
            balance_scores[name] = 0.7 * acc_score + 0.3 * time_score

        best_balanced = max(balance_scores.items(), key=lambda x: x[1])
        recommendations.append(
            f"バランス型推奨: {best_balanced[0]} "
            f"(精度: {accuracy_scores[best_balanced[0]]:.4f}, "
            f"時間: {time_scores[best_balanced[0]]:.2f}秒)"
        )

        # 改善提案
        worst_accuracy = rankings["accuracy"][-1]
        if worst_accuracy[1] < 0.7:
            recommendations.append(
                f"改善推奨: {worst_accuracy[0]}は精度が低い({worst_accuracy[1]:.4f})ため、"
                f"追加学習を検討してください"
            )

        return recommendations

    def find_best_model(
        self, results: dict[str, EvaluationResult], criterion: str = "f1_weighted"
    ) -> tuple[str, EvaluationResult]:
        """
        指定された基準で最良のモデルを見つける

        Args:
            results: モデル評価結果
            criterion: 評価基準

        Returns:
            (モデル名, 評価結果)のタプル
        """
        if not results:
            raise ValueError("No results to compare")

        if criterion == "accuracy":
            best = max(results.items(), key=lambda x: x[1].accuracy)
        elif criterion == "precision_weighted":
            best = max(results.items(), key=lambda x: x[1].precision.get("weighted_avg", 0))
        elif criterion == "recall_weighted":
            best = max(results.items(), key=lambda x: x[1].recall.get("weighted_avg", 0))
        elif criterion == "f1_weighted":
            best = max(results.items(), key=lambda x: x[1].f1_score.get("weighted_avg", 0))
        elif criterion == "speed":
            best = min(results.items(), key=lambda x: x[1].evaluation_time)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        return best
