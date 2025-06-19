"""
学習進捗の可視化モジュール

リアルタイムでの学習進捗モニタリングと可視化
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from ...utils.logger import LoggerMixin


class TrainingVisualizer(LoggerMixin):
    """学習進捗の可視化クラス"""

    def __init__(self, output_dir: str = "training_plots"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def plot_training_history(
        self,
        history: list[dict[str, Any]],
        session_id: str,
        save: bool = True,
    ) -> Figure:
        """
        学習履歴をプロット

        Args:
            history: 学習履歴
            session_id: セッションID
            save: ファイルに保存するか

        Returns:
            Matplotlibの図
        """
        if not history:
            self.logger.warning("履歴データが空です")
            return None

        # データを抽出
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        train_accuracies = [h.get("train_accuracy", 0) for h in history]
        val_accuracies = [h.get("val_accuracy", 0) for h in history]
        learning_rates = [h["learning_rate"] for h in history]

        # 図を作成
        fig = plt.figure(figsize=(15, 10))

        # 損失のプロット
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(epochs, train_losses, label="訓練損失", marker="o", markersize=4)
        ax1.plot(epochs, val_losses, label="検証損失", marker="s", markersize=4)
        ax1.set_xlabel("エポック")
        ax1.set_ylabel("損失")
        ax1.set_title("損失の推移")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 精度のプロット（分類タスクの場合）
        ax2 = plt.subplot(2, 2, 2)
        if any(acc > 0 for acc in train_accuracies):
            ax2.plot(epochs, train_accuracies, label="訓練精度", marker="o", markersize=4)
            ax2.plot(epochs, val_accuracies, label="検証精度", marker="s", markersize=4)
            ax2.set_xlabel("エポック")
            ax2.set_ylabel("精度")
            ax2.set_title("精度の推移")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)
        else:
            ax2.text(0.5, 0.5, "精度データなし", ha="center", va="center", fontsize=16)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

        # 学習率のプロット
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(epochs, learning_rates, label="学習率", marker="^", markersize=4, color="green")
        ax3.set_xlabel("エポック")
        ax3.set_ylabel("学習率")
        ax3.set_title("学習率の推移")
        ax3.set_yscale("log")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 統計情報
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis("off")

        # 最良の結果を取得
        best_val_loss_idx = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_loss_idx]
        best_val_loss_epoch = epochs[best_val_loss_idx]

        stats_text = f"セッションID: {session_id}\n\n"
        stats_text += f"総エポック数: {len(epochs)}\n"
        stats_text += f"最良検証損失: {best_val_loss:.4f} (エポック {best_val_loss_epoch})\n"

        if any(acc > 0 for acc in val_accuracies):
            best_val_acc_idx = np.argmax(val_accuracies)
            best_val_acc = val_accuracies[best_val_acc_idx]
            best_val_acc_epoch = epochs[best_val_acc_idx]
            stats_text += f"最良検証精度: {best_val_acc:.4f} (エポック {best_val_acc_epoch})\n"

        stats_text += f"\n最終学習率: {learning_rates[-1]:.6f}\n"
        stats_text += f"最終訓練損失: {train_losses[-1]:.4f}\n"
        stats_text += f"最終検証損失: {val_losses[-1]:.4f}"

        ax4.text(
            0.1,
            0.9,
            stats_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        # 保存
        if save:
            output_path = self.output_dir / f"training_history_{session_id}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            self.logger.info(f"学習履歴プロットを保存: {output_path}")

        return fig

    def plot_live_metrics(
        self,
        current_metrics: dict[str, float],
        history: list[dict[str, Any]],
        session_id: str,
        window_size: int = 50,
    ) -> Figure:
        """
        リアルタイムメトリクスをプロット

        Args:
            current_metrics: 現在のメトリクス
            history: 過去の履歴
            session_id: セッションID
            window_size: 表示するエポック数のウィンドウ

        Returns:
            Matplotlibの図
        """
        # 最新のwindow_size分のデータを使用
        recent_history = history[-window_size:] if len(history) > window_size else history

        if not recent_history:
            return None

        # 図を作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 損失の移動平均
        epochs = [h["epoch"] for h in recent_history]
        val_losses = [h["val_loss"] for h in recent_history]

        # 移動平均を計算
        ma_window = min(5, len(val_losses))
        if len(val_losses) >= ma_window:
            ma_losses = np.convolve(val_losses, np.ones(ma_window) / ma_window, mode="valid")
            ma_epochs = epochs[ma_window - 1 :]
        else:
            ma_losses = val_losses
            ma_epochs = epochs

        ax1.plot(epochs, val_losses, "o-", alpha=0.5, label="検証損失")
        ax1.plot(ma_epochs, ma_losses, "-", linewidth=2, label=f"移動平均({ma_window})")

        # 現在の値を強調
        if current_metrics:
            current_epoch = history[-1]["epoch"] if history else 0
            ax1.plot(
                current_epoch, current_metrics.get("loss", 0), "ro", markersize=10, label="現在"
            )

        ax1.set_xlabel("エポック")
        ax1.set_ylabel("損失")
        ax1.set_title("検証損失の推移（リアルタイム）")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 学習率と損失の関係
        learning_rates = [h["learning_rate"] for h in recent_history]
        ax2.scatter(learning_rates, val_losses, alpha=0.6)
        ax2.set_xlabel("学習率")
        ax2.set_ylabel("検証損失")
        ax2.set_title("学習率 vs 検証損失")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_training_report(
        self,
        session_id: str,
        config: Any,
        final_metrics: dict[str, Any],
        history: list[dict[str, Any]],
        optimal_batch_size: int | None = None,
    ):
        """
        学習レポートを保存

        Args:
            session_id: セッションID
            config: 訓練設定
            final_metrics: 最終メトリクス
            history: 学習履歴
            optimal_batch_size: 最適化されたバッチサイズ
        """
        report = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "model_type": getattr(config, "model_type", "unknown"),
                "epochs": getattr(config, "epochs", 0),
                "batch_size": getattr(config, "batch_size", 0),
                "optimal_batch_size": optimal_batch_size,
                "learning_rate": getattr(config, "learning_rate", 0),
                "optimizer": getattr(config, "optimizer_type", "unknown"),
                "mixed_precision": getattr(config, "mixed_precision", False),
                "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps", 1),
            },
            "final_metrics": final_metrics,
            "training_summary": self._generate_training_summary(history),
        }

        # JSONで保存
        report_path = self.output_dir / f"training_report_{session_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"学習レポートを保存: {report_path}")

        # テキストレポートも生成
        self._save_text_report(session_id, report)

    def _generate_training_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """学習サマリーを生成"""
        if not history:
            return {}

        val_losses = [h["val_loss"] for h in history]
        val_accuracies = [h.get("val_accuracy", 0) for h in history]

        summary = {
            "total_epochs": len(history),
            "best_val_loss": min(val_losses),
            "best_val_loss_epoch": int(np.argmin(val_losses)),
            "final_val_loss": val_losses[-1],
            "loss_improvement": (val_losses[0] - min(val_losses)) / val_losses[0] * 100,
        }

        if any(acc > 0 for acc in val_accuracies):
            summary.update(
                {
                    "best_val_accuracy": max(val_accuracies),
                    "best_val_accuracy_epoch": int(np.argmax(val_accuracies)),
                    "final_val_accuracy": val_accuracies[-1],
                    "accuracy_improvement": max(val_accuracies) - val_accuracies[0],
                }
            )

        return summary

    def _save_text_report(self, session_id: str, report: dict[str, Any]):
        """テキスト形式のレポートを保存"""
        text_path = self.output_dir / f"training_report_{session_id}.txt"

        with open(text_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("麻雀牌検出モデル 学習レポート\n")
            f.write(f"セッションID: {session_id}\n")
            f.write(f"生成日時: {report['timestamp']}\n")
            f.write("=" * 60 + "\n\n")

            f.write("【訓練設定】\n")
            config = report["configuration"]
            f.write(f"  - モデルタイプ: {config['model_type']}\n")
            f.write(f"  - エポック数: {config['epochs']}\n")
            f.write(f"  - バッチサイズ: {config['batch_size']}")
            if config["optimal_batch_size"]:
                f.write(f" → {config['optimal_batch_size']} (最適化)")
            f.write("\n")
            f.write(f"  - 学習率: {config['learning_rate']}\n")
            f.write(f"  - 最適化器: {config['optimizer']}\n")
            f.write(f"  - 混合精度訓練: {'有効' if config['mixed_precision'] else '無効'}\n")
            f.write(f"  - 勾配累積: {config['gradient_accumulation_steps']}ステップ\n\n")

            f.write("【最終結果】\n")
            metrics = report["final_metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  - {key}: {value:.4f}\n")
                else:
                    f.write(f"  - {key}: {value}\n")
            f.write("\n")

            f.write("【訓練サマリー】\n")
            summary = report["training_summary"]
            if summary:
                f.write(f"  - 総エポック数: {summary.get('total_epochs', 0)}\n")
                f.write(f"  - 最良検証損失: {summary.get('best_val_loss', 0):.4f} ")
                f.write(f"(エポック {summary.get('best_val_loss_epoch', 0)})\n")
                f.write(f"  - 損失改善率: {summary.get('loss_improvement', 0):.1f}%\n")

                if "best_val_accuracy" in summary:
                    f.write(f"  - 最良検証精度: {summary['best_val_accuracy']:.4f} ")
                    f.write(f"(エポック {summary['best_val_accuracy_epoch']})\n")
                    f.write(f"  - 精度向上: +{summary['accuracy_improvement']:.4f}\n")

        self.logger.info(f"テキストレポートを保存: {text_path}")
