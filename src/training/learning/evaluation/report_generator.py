"""
レポート生成専用クラス
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ....utils.logger import LoggerMixin
from .evaluator_base import EvaluationResult


class ReportGenerator(LoggerMixin):
    """レポート生成専用クラス"""

    def __init__(self, output_dir: Path | None = None):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir or Path("data/training/evaluation/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("ReportGenerator初期化完了")

    def generate_evaluation_report(
        self,
        evaluation_result: EvaluationResult,
        model_info: dict[str, Any],
        dataset_info: dict[str, Any],
        save_path: Path | None = None,
    ) -> str:
        """
        評価レポートを生成

        Args:
            evaluation_result: 評価結果
            model_info: モデル情報
            dataset_info: データセット情報
            save_path: 保存パス

        Returns:
            レポート文字列
        """
        report_lines = []

        # ヘッダー
        report_lines.append("=" * 80)
        report_lines.append("モデル評価レポート")
        report_lines.append("=" * 80)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # モデル情報
        report_lines.append("## モデル情報")
        report_lines.append("-" * 40)
        for key, value in model_info.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")

        # データセット情報
        report_lines.append("## データセット情報")
        report_lines.append("-" * 40)
        for key, value in dataset_info.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")

        # 全体的なメトリクス
        report_lines.append("## 全体的なパフォーマンス")
        report_lines.append("-" * 40)
        report_lines.append(f"精度 (Accuracy): {evaluation_result.accuracy:.4f}")
        report_lines.append(
            f"適合率 (Precision) - 重み付き平均: {evaluation_result.precision.get('weighted_avg', 0):.4f}"
        )
        report_lines.append(
            f"再現率 (Recall) - 重み付き平均: {evaluation_result.recall.get('weighted_avg', 0):.4f}"
        )
        report_lines.append(
            f"F1スコア - 重み付き平均: {evaluation_result.f1_score.get('weighted_avg', 0):.4f}"
        )
        report_lines.append("")

        # 評価統計
        report_lines.append("## 評価統計")
        report_lines.append("-" * 40)
        report_lines.append(f"総サンプル数: {evaluation_result.total_samples:,}")
        report_lines.append(f"正解予測数: {evaluation_result.correct_predictions:,}")
        report_lines.append(f"評価時間: {evaluation_result.evaluation_time:.2f}秒")
        report_lines.append("")

        # 信頼度統計
        if evaluation_result.confidence_scores:
            report_lines.append("## 信頼度統計")
            report_lines.append("-" * 40)
            for key, value in evaluation_result.confidence_scores.items():
                report_lines.append(f"{key}: {value:.4f}")
            report_lines.append("")

        # クラスごとの詳細
        report_lines.append("## クラスごとのパフォーマンス")
        report_lines.append("-" * 40)

        # ヘッダー
        report_lines.append(
            f"{'クラス':<15} {'精度':<10} {'適合率':<10} {'再現率':<10} {'F1スコア':<10} {'サンプル数':<10}"
        )
        report_lines.append("-" * 65)

        # 各クラスのメトリクス
        for class_name in sorted(evaluation_result.per_class_accuracy.keys()):
            if class_name.endswith("_avg"):
                continue

            accuracy = evaluation_result.per_class_accuracy.get(class_name, 0)
            precision = evaluation_result.precision.get(class_name, 0)
            recall = evaluation_result.recall.get(class_name, 0)
            f1 = evaluation_result.f1_score.get(class_name, 0)
            samples = evaluation_result.class_distribution.get(class_name, 0)

            report_lines.append(
                f"{class_name:<15} {accuracy:<10.4f} {precision:<10.4f} "
                f"{recall:<10.4f} {f1:<10.4f} {samples:<10}"
            )

        report_lines.append("")

        # 追加メトリクス
        if evaluation_result.additional_metrics:
            report_lines.append("## 追加メトリクス")
            report_lines.append("-" * 40)
            for key, value in evaluation_result.additional_metrics.items():
                report_lines.append(f"{key}: {value}")
            report_lines.append("")

        # レポートを結合
        report = "\n".join(report_lines)

        # 保存
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info(f"評価レポートを保存: {save_path}")

        return report

    def generate_comparison_report(
        self, results: dict[str, EvaluationResult], save_path: Path | None = None
    ) -> str:
        """
        複数モデルの比較レポートを生成

        Args:
            results: モデル名 -> 評価結果の辞書
            save_path: 保存パス

        Returns:
            比較レポート文字列
        """
        report_lines = []

        # ヘッダー
        report_lines.append("=" * 100)
        report_lines.append("モデル比較レポート")
        report_lines.append("=" * 100)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"比較モデル数: {len(results)}")
        report_lines.append("")

        # 全体的なメトリクス比較
        report_lines.append("## 全体的なパフォーマンス比較")
        report_lines.append("-" * 80)

        # ヘッダー
        report_lines.append(
            f"{'モデル名':<20} {'精度':<10} {'適合率':<10} {'再現率':<10} {'F1スコア':<10} {'評価時間':<10}"
        )
        report_lines.append("-" * 70)

        # 各モデルのメトリクス
        for model_name, result in results.items():
            accuracy = result.accuracy
            precision = result.precision.get("weighted_avg", 0)
            recall = result.recall.get("weighted_avg", 0)
            f1 = result.f1_score.get("weighted_avg", 0)
            time = result.evaluation_time

            report_lines.append(
                f"{model_name:<20} {accuracy:<10.4f} {precision:<10.4f} "
                f"{recall:<10.4f} {f1:<10.4f} {time:<10.2f}"
            )

        report_lines.append("")

        # ベストモデルの特定
        report_lines.append("## ベストモデル")
        report_lines.append("-" * 40)

        # 各メトリクスでのベストモデル
        best_accuracy = max(results.items(), key=lambda x: x[1].accuracy)
        best_f1 = max(results.items(), key=lambda x: x[1].f1_score.get("weighted_avg", 0))
        fastest = min(results.items(), key=lambda x: x[1].evaluation_time)

        report_lines.append(f"最高精度: {best_accuracy[0]} ({best_accuracy[1].accuracy:.4f})")
        report_lines.append(
            f"最高F1スコア: {best_f1[0]} ({best_f1[1].f1_score.get('weighted_avg', 0):.4f})"
        )
        report_lines.append(f"最速評価: {fastest[0]} ({fastest[1].evaluation_time:.2f}秒)")
        report_lines.append("")

        # レポートを結合
        report = "\n".join(report_lines)

        # 保存
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            self.logger.info(f"比較レポートを保存: {save_path}")

        return report

    def generate_json_report(
        self,
        evaluation_result: EvaluationResult,
        model_info: dict[str, Any],
        dataset_info: dict[str, Any],
        save_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        JSON形式のレポートを生成

        Args:
            evaluation_result: 評価結果
            model_info: モデル情報
            dataset_info: データセット情報
            save_path: 保存パス

        Returns:
            レポート辞書
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "model_info": model_info,
            "dataset_info": dataset_info,
            "overall_metrics": {
                "accuracy": evaluation_result.accuracy,
                "precision_weighted": evaluation_result.precision.get("weighted_avg", 0),
                "recall_weighted": evaluation_result.recall.get("weighted_avg", 0),
                "f1_weighted": evaluation_result.f1_score.get("weighted_avg", 0),
                "precision_macro": evaluation_result.precision.get("macro_avg", 0),
                "recall_macro": evaluation_result.recall.get("macro_avg", 0),
                "f1_macro": evaluation_result.f1_score.get("macro_avg", 0),
            },
            "per_class_metrics": {
                class_name: {
                    "accuracy": evaluation_result.per_class_accuracy.get(class_name, 0),
                    "precision": evaluation_result.precision.get(class_name, 0),
                    "recall": evaluation_result.recall.get(class_name, 0),
                    "f1_score": evaluation_result.f1_score.get(class_name, 0),
                    "support": evaluation_result.class_distribution.get(class_name, 0),
                }
                for class_name in evaluation_result.per_class_accuracy
                if not class_name.endswith("_avg")
            },
            "evaluation_stats": {
                "total_samples": evaluation_result.total_samples,
                "correct_predictions": evaluation_result.correct_predictions,
                "evaluation_time": evaluation_result.evaluation_time,
            },
            "confidence_scores": evaluation_result.confidence_scores,
            "additional_metrics": evaluation_result.additional_metrics,
        }

        # 保存
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"JSONレポートを保存: {save_path}")

        return report
