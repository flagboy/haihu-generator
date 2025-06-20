"""
リファクタリングされたモデル訓練システム

責務が分離され、単一責任の原則に従った設計
"""

from datetime import datetime
from pathlib import Path
from typing import Any

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None

from ...utils.config import ConfigManager
from ...utils.device_utils import get_available_device
from ...utils.logger import LoggerMixin
from ..annotation_data import AnnotationData
from .batch_size_optimizer import BatchSizeOptimizer, GradientAccumulator
from .components import (
    CallbackManager,
    CheckpointManager,
    DataLoaderFactory,
    MetricsCalculator,
    ModelCheckpointCallback,
    ProgressBarCallback,
    TensorBoardCallback,
    TrainingHistoryCallback,
)
from .early_stopping import AdaptiveEarlyStopping, EarlyStoppingConfig, MetricMode
from .learning_scheduler import LearningScheduler


class RefactoredModelTrainer(LoggerMixin):
    """リファクタリングされたモデル訓練クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # コンポーネントの初期化
        self.device = self._setup_device()
        self.data_loader_factory = DataLoaderFactory(self.device)
        self.checkpoint_manager = CheckpointManager(
            self.config.get("training", {}).get("training_root", "data/training") + "/checkpoints"
        )

        # 学習スケジューラー
        self.scheduler = LearningScheduler(config_manager)

        # 訓練状態管理
        self.training_sessions: dict[str, dict[str, Any]] = {}
        self.stop_flags: dict[str, bool] = {}

    def _setup_device(self) -> Any:
        """デバイス設定"""
        device_config = self.config.get("ai", {}).get("training", {}).get("device", "auto")
        device = get_available_device(preferred_device=device_config)
        if device is None:
            device = torch.device("cpu") if TORCH_AVAILABLE else None
            self.logger.warning("PyTorchが利用できないため、CPUモードで動作します")
        return device

    def train_model(
        self,
        model: nn.Module,
        train_data: AnnotationData,
        val_data: AnnotationData,
        config: Any,
        session_id: str,
    ) -> dict[str, Any]:
        """
        モデルを訓練

        Args:
            model: 訓練するモデル
            train_data: 訓練データ
            val_data: 検証データ
            config: 訓練設定
            session_id: セッションID

        Returns:
            訓練結果
        """
        self.logger.info(f"モデル訓練開始: {config.model_type}, セッション: {session_id}")

        # 訓練状態を初期化
        self._initialize_session(session_id)

        try:
            # デバイスに移動
            model = model.to(self.device)

            # データローダーを作成
            train_loader, val_loader = self._create_data_loaders(train_data, val_data, config)

            # 訓練コンポーネントを設定
            optimizer = self._create_optimizer(model, config)
            lr_scheduler = self._create_lr_scheduler(optimizer, config)
            criterion = self._create_criterion(config)

            # メトリクス計算器
            num_classes = config.get("num_classes", 37)
            train_metrics = MetricsCalculator(num_classes, config.model_type)
            val_metrics = MetricsCalculator(num_classes, config.model_type)

            # コールバックを設定
            callbacks = self._setup_callbacks(
                model, optimizer, config, session_id, len(train_loader), config.epochs
            )

            # 勾配累積の設定
            gradient_accumulator = self._setup_gradient_accumulation(model, optimizer, config)

            # 混合精度訓練の設定
            scaler, use_autocast = self._setup_mixed_precision(config)

            # 早期停止の設定
            early_stopping = self._setup_early_stopping(config, optimizer)

            # 訓練開始
            callbacks.on_train_begin(
                model=model, optimizer=optimizer, config=config, session_id=session_id
            )

            # 訓練ループ
            for epoch in range(config.epochs):
                if self.stop_flags.get(session_id, False):
                    self.logger.info(f"訓練停止要求: セッション {session_id}")
                    break

                self.training_sessions[session_id]["current_epoch"] = epoch

                # エポック開始
                callbacks.on_epoch_begin(epoch)

                # 訓練フェーズ
                train_epoch_metrics = self._train_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    metrics_calculator=train_metrics,
                    callbacks=callbacks,
                    epoch=epoch,
                    gradient_accumulator=gradient_accumulator,
                    scaler=scaler,
                    use_autocast=use_autocast,
                )

                # 検証フェーズ
                val_epoch_metrics = self._validate_epoch(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    metrics_calculator=val_metrics,
                    epoch=epoch,
                )

                # 全体のメトリクスを結合
                epoch_metrics = {
                    **{f"train_{k}": v for k, v in train_epoch_metrics.items()},
                    **{f"val_{k}": v for k, v in val_epoch_metrics.items()},
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                # エポック終了
                callbacks.on_epoch_end(
                    epoch=epoch,
                    metrics=epoch_metrics,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                )

                # 学習率スケジューラー更新
                if lr_scheduler:
                    self._update_lr_scheduler(lr_scheduler, val_epoch_metrics["loss"])

                # 履歴を記録
                self._update_training_history(session_id, epoch, epoch_metrics)

                # 早期停止チェック
                if early_stopping and self._check_early_stopping(
                    early_stopping, val_epoch_metrics, config, epoch, model
                ):
                    break

                # ログ出力
                self._log_epoch_summary(epoch, train_epoch_metrics, val_epoch_metrics)

            # 訓練終了
            callbacks.on_train_end()
            self.training_sessions[session_id]["status"] = "completed"

            # 最終結果を返す
            return self._prepare_final_results(session_id, config)

        except Exception as e:
            self.training_sessions[session_id]["status"] = "failed"
            self.logger.error(f"モデル訓練に失敗: {e}")
            raise

        finally:
            # クリーンアップ
            if session_id in self.stop_flags:
                del self.stop_flags[session_id]

    def _initialize_session(self, session_id: str):
        """セッションを初期化"""
        self.training_sessions[session_id] = {
            "start_time": datetime.now(),
            "current_epoch": 0,
            "best_loss": float("inf"),
            "best_accuracy": 0.0,
            "training_history": [],
            "status": "running",
        }
        self.stop_flags[session_id] = False

    def _create_data_loaders(
        self, train_data: AnnotationData, val_data: AnnotationData, config: Any
    ) -> tuple[DataLoader, DataLoader]:
        """データローダーを作成"""
        # バッチサイズ最適化
        optimal_batch_size = self._optimize_batch_size(train_data, config)

        # データローダーを作成
        train_loader = self.data_loader_factory.create_dataloader(
            train_data, config, is_training=True, batch_size=optimal_batch_size
        )
        val_loader = self.data_loader_factory.create_dataloader(
            val_data, config, is_training=False, batch_size=optimal_batch_size
        )

        return train_loader, val_loader

    def _optimize_batch_size(self, train_data: AnnotationData, config: Any) -> int:
        """バッチサイズを最適化"""
        if not hasattr(config, "optimize_batch_size") or not config.optimize_batch_size:
            return config.batch_size

        self.logger.info("バッチサイズ最適化を実行中...")

        # ダミーモデルを作成（実際の実装では適切なモデルを使用）
        dummy_model = nn.Linear(10, 10).to(self.device)

        optimizer = BatchSizeOptimizer(
            model=dummy_model,
            device=self.device,
            initial_batch_size=config.batch_size,
            max_batch_size=config.get("max_batch_size", 256),
            memory_fraction=config.get("gpu_memory_fraction", 0.9),
        )

        def dataloader_factory(batch_size: int) -> DataLoader:
            return self.data_loader_factory.create_dataloader(
                train_data, config, is_training=True, batch_size=batch_size
            )

        criterion = self._create_criterion(config)
        optimal_batch_size = optimizer.find_optimal_batch_size(
            dataloader_factory=dataloader_factory,
            loss_fn=criterion,
            mixed_precision=config.get("mixed_precision", True),
        )

        self.logger.info(f"最適バッチサイズ: {optimal_batch_size}")
        return optimal_batch_size

    def _setup_callbacks(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        config: Any,
        session_id: str,
        batches_per_epoch: int,
        total_epochs: int,
    ) -> CallbackManager:
        """コールバックを設定"""
        callbacks = []

        # プログレスバー
        callbacks.append(ProgressBarCallback(total_epochs, batches_per_epoch))

        # モデルチェックポイント
        monitor = "val_accuracy" if config.model_type == "classification" else "val_loss"
        mode = "max" if config.model_type == "classification" else "min"
        callbacks.append(
            ModelCheckpointCallback(
                self.checkpoint_manager,
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_freq=10,
            )
        )

        # TensorBoard
        if config.get("use_tensorboard", True):
            callbacks.append(TensorBoardCallback())

        # 訓練履歴
        history_file = (
            Path(self.config.get("training", {}).get("training_root", "data/training"))
            / "history"
            / f"{session_id}_history.json"
        )
        history_file.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(TrainingHistoryCallback(str(history_file)))

        return CallbackManager(callbacks)

    def _setup_gradient_accumulation(
        self, model: nn.Module, optimizer: optim.Optimizer, config: Any
    ) -> GradientAccumulator | None:
        """勾配累積を設定"""
        accumulation_steps = config.get("gradient_accumulation_steps", 1)

        if accumulation_steps > 1:
            gradient_accumulator = GradientAccumulator(
                model=model,
                optimizer=optimizer,
                accumulation_steps=accumulation_steps,
                max_grad_norm=config.get("max_grad_norm", 1.0),
            )
            self.logger.info(f"勾配累積を使用: {accumulation_steps}ステップ")
            return gradient_accumulator

        return None

    def _setup_mixed_precision(self, config: Any) -> tuple[Any, bool]:
        """混合精度訓練を設定"""
        scaler = None
        use_autocast = False

        if config.get("mixed_precision", False):
            if self.device.type == "cuda":
                scaler = torch.cuda.amp.GradScaler()
                use_autocast = True
                self.logger.info("CUDA混合精度訓練を有効化")
            elif self.device.type == "mps":
                use_autocast = True
                self.logger.info("MPS混合精度訓練を有効化（自動キャスト）")

        return scaler, use_autocast

    def _setup_early_stopping(
        self, config: Any, optimizer: optim.Optimizer
    ) -> AdaptiveEarlyStopping | None:
        """早期停止を設定"""
        if not config.get("use_early_stopping", True):
            return None

        metric_mode = MetricMode.MAX if config.model_type == "classification" else MetricMode.MIN

        early_stopping_config = EarlyStoppingConfig(
            patience=config.get("early_stopping_patience", 10),
            min_delta=config.get("early_stopping_min_delta", 0.0001),
            mode=metric_mode,
            restore_best_weights=True,
            warmup_epochs=config.get("warmup_epochs", 0),
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        early_stopping = AdaptiveEarlyStopping(
            config=early_stopping_config,
            initial_lr=initial_lr,
            lr_patience_factor=config.get("lr_patience_factor", 2.0),
        )

        self.logger.info(
            f"早期停止を有効化: 忍耐度={early_stopping_config.patience}, モード={metric_mode.value}"
        )

        return early_stopping

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        metrics_calculator: MetricsCalculator,
        callbacks: CallbackManager,
        epoch: int,
        gradient_accumulator: GradientAccumulator | None = None,
        scaler: Any = None,
        use_autocast: bool = False,
    ) -> dict[str, float]:
        """1エポックの訓練"""
        model.train()
        metrics_calculator.reset()

        global_step = epoch * len(dataloader)

        for batch_idx, (data, target) in enumerate(dataloader):
            callbacks.on_batch_begin(batch_idx)

            data, target = data.to(self.device), target.to(self.device)

            # 勾配計算
            loss = self._compute_loss(
                model,
                data,
                target,
                criterion,
                optimizer,
                gradient_accumulator,
                scaler,
                use_autocast,
            )

            # メトリクス更新
            with torch.no_grad():
                outputs = model(data)
                metrics_calculator.update(loss.item(), outputs, target, data.size(0))

            # コールバック
            callbacks.on_batch_end(batch_idx, loss.item(), global_step=global_step + batch_idx)

        return metrics_calculator.compute_epoch_metrics()

    def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        metrics_calculator: MetricsCalculator,
        epoch: int,
    ) -> dict[str, float]:
        """1エポックの検証"""
        model.eval()
        metrics_calculator.reset()

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = model(data)
                loss = criterion(outputs, target)

                metrics_calculator.update(loss.item(), outputs, target, data.size(0))

        return metrics_calculator.compute_epoch_metrics()

    def _compute_loss(
        self,
        model: nn.Module,
        data: Any,
        target: Any,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        gradient_accumulator: GradientAccumulator | None,
        scaler: Any,
        use_autocast: bool,
    ) -> Any:
        """損失を計算して逆伝播"""
        if gradient_accumulator is None:
            optimizer.zero_grad()

        # 混合精度訓練の文脈
        autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device_type, enabled=use_autocast):
            outputs = model(data)
            loss = criterion(outputs, target)

        # 勾配計算
        if gradient_accumulator is not None:
            gradient_accumulator.step(loss, scaler)
        else:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        return loss

    def _create_optimizer(self, model: nn.Module, config: Any) -> optim.Optimizer:
        """最適化器を作成"""
        optimizer_type = getattr(config, "optimizer_type", "adam").lower()
        lr = config.learning_rate

        if optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "adamw":
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            return optim.Adam(model.parameters(), lr=lr)

    def _create_lr_scheduler(self, optimizer: optim.Optimizer, config: Any) -> Any:
        """学習率スケジューラーを作成"""
        scheduler_type = getattr(config, "lr_scheduler_type", "plateau").lower()

        if scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        else:
            return None

    def _create_criterion(self, config: Any) -> nn.Module:
        """損失関数を作成"""
        if config.model_type == "detection":
            return nn.MSELoss()
        else:
            return nn.CrossEntropyLoss()

    def _update_lr_scheduler(self, lr_scheduler: Any, val_loss: float):
        """学習率スケジューラーを更新"""
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    def _update_training_history(self, session_id: str, epoch: int, metrics: dict[str, float]):
        """訓練履歴を更新"""
        epoch_history = {
            "epoch": epoch,
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.training_sessions[session_id]["training_history"].append(epoch_history)

        # ベストスコアを更新
        if (
            "val_loss" in metrics
            and metrics["val_loss"] < self.training_sessions[session_id]["best_loss"]
        ):
            self.training_sessions[session_id]["best_loss"] = metrics["val_loss"]

        if (
            "val_accuracy" in metrics
            and metrics["val_accuracy"] > self.training_sessions[session_id]["best_accuracy"]
        ):
            self.training_sessions[session_id]["best_accuracy"] = metrics["val_accuracy"]

    def _check_early_stopping(
        self,
        early_stopping: AdaptiveEarlyStopping,
        val_metrics: dict[str, float],
        config: Any,
        epoch: int,
        model: nn.Module,
    ) -> bool:
        """早期停止をチェック"""
        monitor_value = (
            val_metrics.get("accuracy", 0.0)
            if config.model_type == "classification"
            else val_metrics["loss"]
        )

        if early_stopping(monitor_value, model):
            self.logger.info(f"早期停止条件を満たしました (エポック {epoch})")
            early_stopping.restore_best_weights(model)
            return True

        return False

    def _log_epoch_summary(
        self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
    ):
        """エポックのサマリーをログ出力"""
        self.logger.info(
            f"エポック {epoch}: "
            f"訓練損失={train_metrics['loss']:.4f}, "
            f"検証損失={val_metrics['loss']:.4f}, "
            f"検証精度={val_metrics.get('accuracy', 0.0):.4f}"
        )

    def _prepare_final_results(self, session_id: str, config: Any) -> dict[str, Any]:
        """最終結果を準備"""
        session = self.training_sessions[session_id]

        # ベストモデルのパスを取得
        best_model_path = self.checkpoint_manager.find_best_checkpoint(session_id)

        final_metrics = {
            "best_val_loss": session["best_loss"],
            "best_val_accuracy": session["best_accuracy"],
            "total_epochs": len(session["training_history"]),
            "training_time": (datetime.now() - session["start_time"]).total_seconds(),
        }

        return {
            "best_model_path": best_model_path,
            "final_metrics": final_metrics,
            "training_history": session["training_history"],
        }

    def stop_training(self, session_id: str):
        """訓練を停止"""
        self.stop_flags[session_id] = True
        if session_id in self.training_sessions:
            self.training_sessions[session_id]["status"] = "stopping"

    def get_training_progress(self, session_id: str) -> dict[str, Any] | None:
        """訓練進捗を取得"""
        return self.training_sessions.get(session_id)
