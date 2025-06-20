"""
訓練コールバック

訓練中のイベント処理を抽象化し、
拡張可能なコールバックシステムを提供
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from ....utils.logger import LoggerMixin


class TrainingCallback(ABC):
    """訓練コールバックの基底クラス"""

    @abstractmethod
    def on_train_begin(self, **kwargs):
        """訓練開始時"""
        pass

    @abstractmethod
    def on_train_end(self, **kwargs):
        """訓練終了時"""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時"""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時"""
        pass

    @abstractmethod
    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時"""
        pass

    @abstractmethod
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時"""
        pass


class CallbackManager:
    """コールバック管理クラス"""

    def __init__(self, callbacks: list[TrainingCallback] | None = None):
        """
        初期化

        Args:
            callbacks: コールバックのリスト
        """
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainingCallback):
        """コールバックを追加"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainingCallback):
        """コールバックを削除"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_train_begin(self, **kwargs):
        """訓練開始時の処理"""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        """訓練終了時の処理"""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時の処理"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時の処理"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, **kwargs)

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時の処理"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, **kwargs)

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時の処理"""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, **kwargs)


class ProgressBarCallback(TrainingCallback, LoggerMixin):
    """プログレスバー表示コールバック"""

    def __init__(self, total_epochs: int, total_batches: int):
        """
        初期化

        Args:
            total_epochs: 総エポック数
            total_batches: エポックあたりのバッチ数
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.current_epoch = 0
        self.current_batch = 0

    def on_train_begin(self, **kwargs):
        """訓練開始時"""
        self.logger.info(f"訓練開始: {self.total_epochs}エポック")

    def on_train_end(self, **kwargs):
        """訓練終了時"""
        self.logger.info("訓練完了")

    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時"""
        self.current_epoch = epoch
        self.current_batch = 0
        self.logger.info(f"エポック {epoch + 1}/{self.total_epochs} 開始")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"エポック {epoch + 1}/{self.total_epochs} 完了: {metrics_str}")

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時"""
        self.current_batch = batch_idx

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時"""
        if (batch_idx + 1) % 10 == 0:  # 10バッチごとに表示
            progress = (batch_idx + 1) / self.total_batches * 100
            self.logger.info(
                f"  バッチ {batch_idx + 1}/{self.total_batches} "
                f"({progress:.1f}%) - loss: {loss:.4f}"
            )


class TensorBoardCallback(TrainingCallback, LoggerMixin):
    """TensorBoard用コールバック"""

    def __init__(self, log_dir: str = "logs/tensorboard"):
        """
        初期化

        Args:
            log_dir: ログディレクトリ
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.log_dir))
            self.logger.info(f"TensorBoardログ出力先: {self.log_dir}")
        except ImportError:
            self.logger.warning("TensorBoardが利用できません")

    def on_train_begin(self, **kwargs):
        """訓練開始時"""
        if self.writer and "config" in kwargs:
            config = kwargs["config"]
            hparams = {
                "learning_rate": getattr(config, "learning_rate", 0.001),
                "batch_size": getattr(config, "batch_size", 32),
                "epochs": getattr(config, "epochs", 100),
                "optimizer": getattr(config, "optimizer_type", "adam"),
            }
            self.writer.add_hparams(hparams, {})

    def on_train_end(self, **kwargs):
        """訓練終了時"""
        if self.writer:
            self.writer.close()

    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時"""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時"""
        if self.writer:
            # メトリクスを記録
            for key, value in metrics.items():
                if "train" in key:
                    self.writer.add_scalar(f"Train/{key}", value, epoch)
                elif "val" in key:
                    self.writer.add_scalar(f"Validation/{key}", value, epoch)
                else:
                    self.writer.add_scalar(f"Metrics/{key}", value, epoch)

            # 学習率を記録
            if "optimizer" in kwargs:
                optimizer = kwargs["optimizer"]
                for i, param_group in enumerate(optimizer.param_groups):
                    self.writer.add_scalar(f"Learning_Rate/group_{i}", param_group["lr"], epoch)

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時"""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時"""
        if self.writer and "global_step" in kwargs:
            global_step = kwargs["global_step"]
            self.writer.add_scalar("Loss/batch", loss, global_step)


class ModelCheckpointCallback(TrainingCallback, LoggerMixin):
    """モデルチェックポイント保存コールバック"""

    def __init__(
        self,
        checkpoint_manager,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: int = 1,
    ):
        """
        初期化

        Args:
            checkpoint_manager: チェックポイント管理器
            monitor: モニターするメトリクス
            mode: "min" または "max"
            save_best_only: ベストモデルのみ保存するか
            save_freq: 保存頻度（エポック）
        """
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.session_id = None

    def on_train_begin(self, **kwargs):
        """訓練開始時"""
        self.session_id = kwargs.get("session_id", "default")

    def on_train_end(self, **kwargs):
        """訓練終了時"""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時"""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時"""
        if self.monitor in metrics:
            current_value = metrics[self.monitor]

            # ベストモデルの判定
            is_best = False
            if (
                self.mode == "min"
                and current_value < self.best_value
                or self.mode == "max"
                and current_value > self.best_value
            ):
                is_best = True
                self.best_value = current_value

            # 保存条件の判定
            should_save = False
            if is_best or not self.save_best_only and epoch % self.save_freq == 0:
                should_save = True

            # チェックポイント保存
            if should_save and "model" in kwargs and "optimizer" in kwargs:
                self.checkpoint_manager.save_checkpoint(
                    model=kwargs["model"],
                    optimizer=kwargs["optimizer"],
                    epoch=epoch,
                    metrics=metrics,
                    config=kwargs.get("config"),
                    session_id=self.session_id,
                    is_best=is_best,
                )

                if is_best:
                    self.logger.info(f"ベストモデル更新: {self.monitor}={current_value:.4f}")

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時"""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時"""
        pass


class TrainingHistoryCallback(TrainingCallback):
    """訓練履歴記録コールバック"""

    def __init__(self, history_file: str | None = None):
        """
        初期化

        Args:
            history_file: 履歴保存ファイル
        """
        self.history = {
            "epochs": [],
            "metrics": {},
            "start_time": None,
            "end_time": None,
            "config": {},
        }
        self.history_file = history_file
        self.current_epoch_data = {}

    def on_train_begin(self, **kwargs):
        """訓練開始時"""
        self.history["start_time"] = datetime.now().isoformat()
        if "config" in kwargs:
            config = kwargs["config"]
            if hasattr(config, "__dict__"):
                self.history["config"] = config.__dict__

    def on_train_end(self, **kwargs):
        """訓練終了時"""
        self.history["end_time"] = datetime.now().isoformat()
        self._save_history()

    def on_epoch_begin(self, epoch: int, **kwargs):
        """エポック開始時"""
        self.current_epoch_data = {
            "epoch": epoch,
            "start_time": datetime.now().isoformat(),
        }

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs):
        """エポック終了時"""
        self.current_epoch_data["end_time"] = datetime.now().isoformat()
        self.current_epoch_data["metrics"] = metrics

        # 履歴に追加
        self.history["epochs"].append(self.current_epoch_data)

        # メトリクスごとの履歴を更新
        for key, value in metrics.items():
            if key not in self.history["metrics"]:
                self.history["metrics"][key] = []
            self.history["metrics"][key].append(value)

        # 定期的に保存
        if epoch % 5 == 0:
            self._save_history()

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """バッチ開始時"""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """バッチ終了時"""
        # バッチごとの損失を記録（オプション）
        if "batch_losses" not in self.current_epoch_data:
            self.current_epoch_data["batch_losses"] = []
        self.current_epoch_data["batch_losses"].append(loss)

    def _save_history(self):
        """履歴を保存"""
        if self.history_file:
            try:
                with open(self.history_file, "w") as f:
                    json.dump(self.history, f, indent=2)
            except Exception:
                pass  # エラーは無視

    def get_history(self) -> dict[str, Any]:
        """履歴を取得"""
        return self.history
