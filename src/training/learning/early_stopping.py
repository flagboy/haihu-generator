"""
早期停止（Early Stopping）機能の実装

過学習を防ぐための高度な早期停止機能
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...utils.logger import LoggerMixin


class MetricMode(Enum):
    """メトリクスの最適化方向"""

    MIN = "min"  # 損失など、小さいほど良い
    MAX = "max"  # 精度など、大きいほど良い


@dataclass
class EarlyStoppingConfig:
    """早期停止の設定"""

    patience: int = 10  # 改善が見られない最大エポック数
    min_delta: float = 0.0001  # 改善と見なす最小変化量
    mode: MetricMode = MetricMode.MIN  # メトリクスの最適化方向
    restore_best_weights: bool = True  # ベストウェイトを復元するか
    baseline: float | None = None  # ベースラインスコア
    warmup_epochs: int = 0  # ウォームアップエポック数


class EarlyStopping(LoggerMixin):
    """早期停止クラス"""

    def __init__(self, config: EarlyStoppingConfig):
        """
        初期化

        Args:
            config: 早期停止の設定
        """
        super().__init__()
        self.config = config
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.epoch_count = 0
        self.stopped_epoch = 0
        self.early_stopped = False

        # ベースラインの設定
        if config.baseline is not None:
            self.best_score = config.baseline

    def __call__(self, current_score: float, model: Any = None) -> bool:
        """
        早期停止の判定

        Args:
            current_score: 現在のスコア
            model: モデル（ベストウェイトの保存用）

        Returns:
            停止すべきかどうか
        """
        self.epoch_count += 1

        # ウォームアップ期間中は停止しない
        if self.epoch_count <= self.config.warmup_epochs:
            self.logger.debug(
                f"ウォームアップ期間中: エポック {self.epoch_count}/{self.config.warmup_epochs}"
            )
            return False

        # 初回または改善があった場合
        if self.best_score is None or self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = self.epoch_count
            self.counter = 0

            # ベストウェイトを保存
            if model is not None and self.config.restore_best_weights:
                import copy

                self.best_weights = copy.deepcopy(model.state_dict())
                self.logger.info(
                    f"改善を検出: {self._format_score(current_score)} (エポック {self.epoch_count})"
                )
        else:
            self.counter += 1
            self.logger.debug(
                f"改善なし: カウンター {self.counter}/{self.config.patience} "
                f"(現在: {self._format_score(current_score)}, "
                f"ベスト: {self._format_score(self.best_score)})"
            )

            # 忍耐限界に達した場合
            if self.counter >= self.config.patience:
                self.stopped_epoch = self.epoch_count
                self.early_stopped = True
                self.logger.info(
                    f"早期停止: {self.config.patience}エポック改善なし "
                    f"(ベストスコア: {self._format_score(self.best_score)} "
                    f"@ エポック {self.best_epoch})"
                )
                return True

        return False

    def _is_improvement(self, current_score: float) -> bool:
        """
        スコアが改善したかを判定

        Args:
            current_score: 現在のスコア

        Returns:
            改善したかどうか
        """
        if self.best_score is None:
            return True

        if self.config.mode == MetricMode.MIN:
            return current_score < self.best_score - self.config.min_delta
        else:
            return current_score > self.best_score + self.config.min_delta

    def _format_score(self, score: float) -> str:
        """スコアをフォーマット"""
        return f"{score:.6f}"

    def restore_best_weights(self, model: Any) -> bool:
        """
        ベストウェイトを復元

        Args:
            model: モデル

        Returns:
            復元成功かどうか
        """
        if self.best_weights is not None and self.config.restore_best_weights:
            model.load_state_dict(self.best_weights)
            self.logger.info(
                f"ベストウェイトを復元 (エポック {self.best_epoch}, "
                f"スコア: {self._format_score(self.best_score)})"
            )
            return True
        return False

    def get_status(self) -> dict[str, Any]:
        """早期停止の状態を取得"""
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "epoch_count": self.epoch_count,
            "early_stopped": self.early_stopped,
            "stopped_epoch": self.stopped_epoch,
        }


class MultiMetricEarlyStopping(LoggerMixin):
    """複数メトリクスに基づく早期停止"""

    def __init__(
        self,
        metrics_config: dict[str, EarlyStoppingConfig],
        aggregation: str = "all",  # "all" or "any"
    ):
        """
        初期化

        Args:
            metrics_config: メトリクス名とその設定のマッピング
            aggregation: 集約方法（"all": 全て停止、"any": いずれか停止）
        """
        super().__init__()
        self.stoppers = {name: EarlyStopping(config) for name, config in metrics_config.items()}
        self.aggregation = aggregation

    def __call__(self, metrics: dict[str, float], model: Any = None) -> bool:
        """
        早期停止の判定

        Args:
            metrics: メトリクス辞書
            model: モデル

        Returns:
            停止すべきかどうか
        """
        stop_decisions = []

        for name, stopper in self.stoppers.items():
            if name in metrics:
                should_stop = stopper(metrics[name], model)
                stop_decisions.append(should_stop)

                if should_stop:
                    self.logger.info(f"メトリクス '{name}' が早期停止条件を満たしました")

        if not stop_decisions:
            return False

        if self.aggregation == "all":
            return all(stop_decisions)
        else:  # "any"
            return any(stop_decisions)

    def restore_best_weights(self, model: Any, metric_name: str | None = None) -> bool:
        """
        ベストウェイトを復元

        Args:
            model: モデル
            metric_name: 特定のメトリクスのウェイトを使用（Noneの場合は最初のもの）

        Returns:
            復元成功かどうか
        """
        if metric_name and metric_name in self.stoppers:
            return self.stoppers[metric_name].restore_best_weights(model)

        # 最初の有効なウェイトを復元
        return any(stopper.restore_best_weights(model) for stopper in self.stoppers.values())

    def get_status(self) -> dict[str, dict[str, Any]]:
        """全ての早期停止の状態を取得"""
        return {name: stopper.get_status() for name, stopper in self.stoppers.items()}


class AdaptiveEarlyStopping(EarlyStopping):
    """適応的早期停止（学習率に応じて忍耐度を調整）"""

    def __init__(
        self,
        config: EarlyStoppingConfig,
        initial_lr: float,
        lr_patience_factor: float = 2.0,
    ):
        """
        初期化

        Args:
            config: 早期停止の設定
            initial_lr: 初期学習率
            lr_patience_factor: 学習率低下時の忍耐度増加係数
        """
        super().__init__(config)
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_patience_factor = lr_patience_factor
        self.base_patience = config.patience

    def update_learning_rate(self, new_lr: float):
        """
        学習率を更新し、忍耐度を調整

        Args:
            new_lr: 新しい学習率
        """
        if new_lr < self.current_lr:
            # 学習率が下がったら忍耐度を増やす
            lr_reduction_ratio = new_lr / self.initial_lr
            patience_multiplier = 1 + (1 - lr_reduction_ratio) * (self.lr_patience_factor - 1)
            self.config.patience = int(self.base_patience * patience_multiplier)

            self.logger.info(
                f"学習率低下を検出: {self.current_lr:.6f} → {new_lr:.6f}, "
                f"忍耐度を調整: {self.base_patience} → {self.config.patience}"
            )

        self.current_lr = new_lr
