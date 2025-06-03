"""
可視化機能専用クラス
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ....utils.logger import LoggerMixin


class VisualizationGenerator(LoggerMixin):
    """可視化機能専用クラス"""

    def __init__(self, output_dir: Path | None = None):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir or Path("data/training/evaluation/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # matplotlib設定
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

        # 日本語フォント設定
        try:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [
                "Hiragino Sans",
                "Yu Gothic",
                "Meirio",
                "Takao",
                "IPAexGothic",
                "IPAPGothic",
                "VL PGothic",
                "Noto Sans CJK JP",
            ]
        except Exception:
            self.logger.warning("日本語フォントの設定に失敗しました")

        self.logger.info("VisualizationGenerator初期化完了")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: list[str] | None = None,
        title: str = "Confusion Matrix",
        save_path: Path | None = None,
        normalize: bool = False,
        cmap: str = "Blues",
    ) -> None:
        """
        混同行列をプロット

        Args:
            cm: 混同行列
            class_names: クラス名
            title: タイトル
            save_path: 保存パス
            normalize: 正規化するか
            cmap: カラーマップ
        """
        plt.figure(figsize=(12, 10))

        # 正規化
        if normalize:
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
        else:
            cm_normalized = cm

        # クラス名設定
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        # ヒートマップ作成
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={"label": "Normalized Count" if normalize else "Count"},
        )

        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"混同行列を保存: {save_path}")

        plt.close()

    def plot_per_class_metrics(
        self,
        metrics: dict[str, float],
        metric_name: str = "Accuracy",
        title: str | None = None,
        save_path: Path | None = None,
        color: str = "skyblue",
    ) -> None:
        """
        クラスごとのメトリクスをプロット

        Args:
            metrics: クラスごとのメトリクス
            metric_name: メトリクス名
            title: タイトル
            save_path: 保存パス
            color: バーの色
        """
        # データ準備
        classes = list(metrics.keys())
        values = list(metrics.values())

        # マクロ/重み付き平均を除外（必要に応じて）
        if "macro_avg" in classes:
            avg_indices = [i for i, c in enumerate(classes) if c.endswith("_avg")]
            for idx in reversed(avg_indices):
                classes.pop(idx)
                values.pop(idx)

        # ソート
        sorted_data = sorted(zip(classes, values, strict=False), key=lambda x: x[1], reverse=True)
        classes, values = zip(*sorted_data, strict=False)

        # プロット
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(classes)), values, color=color, edgecolor="black", alpha=0.7)

        # 値をバーの上に表示
        for bar, value in zip(bars, values, strict=False):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xlabel("Class")
        plt.ylabel(metric_name)
        plt.title(title or f"{metric_name} by Class")
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
        plt.ylim(0, max(values) * 1.1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"クラスごとのメトリクスを保存: {save_path}")

        plt.close()

    def plot_metrics_comparison(
        self,
        metrics_dict: dict[str, dict[str, float]],
        title: str = "Metrics Comparison",
        save_path: Path | None = None,
    ) -> None:
        """
        複数のメトリクスを比較プロット

        Args:
            metrics_dict: メトリクスの辞書 (metric_name -> class -> value)
            title: タイトル
            save_path: 保存パス
        """
        # データ準備
        metric_names = list(metrics_dict.keys())
        if not metric_names:
            return

        # クラス名を取得（最初のメトリクスから）
        class_names = list(metrics_dict[metric_names[0]].keys())
        # 平均値を除外
        class_names = [c for c in class_names if not c.endswith("_avg")]

        # バーの設定
        n_metrics = len(metric_names)
        n_classes = len(class_names)
        bar_width = 0.8 / n_metrics
        x = np.arange(n_classes)

        # プロット
        plt.figure(figsize=(14, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))

        for i, metric_name in enumerate(metric_names):
            values = [metrics_dict[metric_name].get(cls, 0) for cls in class_names]
            offset = (i - n_metrics / 2) * bar_width + bar_width / 2
            plt.bar(x + offset, values, bar_width, label=metric_name, color=colors[i])

        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.title(title)
        plt.xticks(x, class_names, rotation=45, ha="right")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"メトリクス比較を保存: {save_path}")

        plt.close()

    def plot_training_history(
        self,
        history: dict[str, list[float]],
        title: str = "Training History",
        save_path: Path | None = None,
    ) -> None:
        """
        学習履歴をプロット

        Args:
            history: 学習履歴 (metric_name -> values)
            title: タイトル
            save_path: 保存パス
        """
        # 損失とメトリクスを分離
        loss_keys = [k for k in history if "loss" in k.lower()]
        metric_keys = [k for k in history if k not in loss_keys]

        # サブプロット数を決定
        n_plots = 1 if not metric_keys else 2

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
        if n_plots == 1:
            axes = [axes]

        # 損失のプロット
        ax = axes[0]
        for key in loss_keys:
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], label=key, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # メトリクスのプロット
        if metric_keys and n_plots > 1:
            ax = axes[1]
            for key in metric_keys:
                epochs = range(1, len(history[key]) + 1)
                ax.plot(epochs, history[key], label=key, marker="o", markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.set_title("Metrics Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"学習履歴を保存: {save_path}")

        plt.close()

    def create_evaluation_summary_plot(
        self, evaluation_result: Any, save_path: Path | None = None
    ) -> None:
        """
        評価結果のサマリープロットを作成

        Args:
            evaluation_result: 評価結果
            save_path: 保存パス
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Evaluation Summary", fontsize=16)

        # 1. 全体的なメトリクス
        ax = axes[0, 0]
        metrics = {
            "Accuracy": evaluation_result.accuracy,
            "Precision": evaluation_result.precision.get("weighted_avg", 0),
            "Recall": evaluation_result.recall.get("weighted_avg", 0),
            "F1-Score": evaluation_result.f1_score.get("weighted_avg", 0),
        }
        bars = ax.bar(metrics.keys(), metrics.values(), color=["green", "blue", "orange", "red"])
        ax.set_ylabel("Score")
        ax.set_title("Overall Metrics")
        ax.set_ylim(0, 1.1)

        # 値を表示
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. クラス分布
        ax = axes[0, 1]
        if evaluation_result.class_distribution:
            classes = list(evaluation_result.class_distribution.keys())
            counts = list(evaluation_result.class_distribution.values())
            ax.pie(counts, labels=classes, autopct="%1.1f%%", startangle=90)
            ax.set_title("Class Distribution")

        # 3. Top-5 Best/Worst Classes
        ax = axes[1, 0]
        if evaluation_result.per_class_accuracy:
            sorted_classes = sorted(
                evaluation_result.per_class_accuracy.items(), key=lambda x: x[1], reverse=True
            )
            top_5 = sorted_classes[:5]
            bottom_5 = sorted_classes[-5:]

            classes = [x[0] for x in (top_5 + bottom_5)]
            accuracies = [x[1] for x in (top_5 + bottom_5)]
            colors = ["green"] * 5 + ["red"] * 5

            bars = ax.bar(range(len(classes)), accuracies, color=colors)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title("Best and Worst Performing Classes")
            ax.set_ylim(0, 1.1)

        # 4. 評価統計
        ax = axes[1, 1]
        stats_text = f"""
Total Samples: {evaluation_result.total_samples:,}
Correct Predictions: {evaluation_result.correct_predictions:,}
Evaluation Time: {evaluation_result.evaluation_time:.2f}s

Confidence Stats:
Mean: {evaluation_result.confidence_scores.get("mean_confidence", 0):.3f}
Std: {evaluation_result.confidence_scores.get("std_confidence", 0):.3f}
        """
        ax.text(
            0.1,
            0.5,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="center",
            fontsize=12,
            family="monospace",
        )
        ax.set_title("Evaluation Statistics")
        ax.axis("off")

        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"評価サマリーを保存: {save_path}")

        plt.close()

    def plot_learning_curves(
        self, history: list[dict[str, float]], save_path: Path | None = None
    ) -> None:
        """
        学習曲線をプロット

        Args:
            history: 学習履歴のリスト
            save_path: 保存パス
        """
        if not history:
            return

        # データを抽出
        epochs = [h.get("epoch", i) for i, h in enumerate(history)]
        train_loss = [h.get("train_loss", 0) for h in history]
        val_loss = [h.get("val_loss", 0) for h in history]
        train_acc = [h.get("train_accuracy", 0) for h in history]
        val_acc = [h.get("val_accuracy", 0) for h in history]

        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 損失曲線
        ax1.plot(epochs, train_loss, "b-", label="Train Loss")
        ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 精度曲線
        ax2.plot(epochs, train_acc, "b-", label="Train Accuracy")
        ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"学習曲線を保存: {save_path}")

        plt.close()
