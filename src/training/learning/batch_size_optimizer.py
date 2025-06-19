"""
バッチサイズ最適化ユーティリティ

GPUメモリ使用量に基づいて最適なバッチサイズを動的に決定
"""

import gc
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from ...utils.logger import LoggerMixin


class BatchSizeOptimizer(LoggerMixin):
    """バッチサイズ最適化器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        memory_fraction: float = 0.9,
        num_trials: int = 5,
    ):
        """
        初期化

        Args:
            model: 訓練対象のモデル
            device: 使用デバイス
            initial_batch_size: 初期バッチサイズ
            max_batch_size: 最大バッチサイズ
            memory_fraction: 使用可能なGPUメモリの割合
            num_trials: 各バッチサイズの試行回数
        """
        super().__init__()
        self.model = model
        self.device = device
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_fraction = memory_fraction
        self.num_trials = num_trials

        # デバイス情報を取得
        if device.type == "cuda":
            self.gpu_properties = torch.cuda.get_device_properties(device)
            self.total_memory = self.gpu_properties.total_memory
            self.logger.info(
                f"CUDA GPU検出: {self.gpu_properties.name} (メモリ: {self.total_memory / 1024**3:.1f}GB)"
            )
        elif device.type == "mps":
            # MPSの場合、メモリ情報の取得は現在サポートされていない
            self.total_memory = 0
            self.logger.info("Apple Silicon MPS検出: メモリ最適化は制限されます")
        else:
            self.total_memory = 0
            self.logger.info("CPU使用: バッチサイズ最適化は制限されます")

    def find_optimal_batch_size(
        self,
        dataloader_factory: Callable[[int], DataLoader],
        loss_fn: nn.Module,
        mixed_precision: bool = True,
    ) -> int:
        """
        最適なバッチサイズを見つける

        Args:
            dataloader_factory: バッチサイズを受け取ってDataLoaderを返す関数
            loss_fn: 損失関数
            mixed_precision: 混合精度訓練を使用するか

        Returns:
            最適なバッチサイズ
        """
        if self.device.type not in ["cuda", "mps"]:
            self.logger.info("CPU使用のため、デフォルトバッチサイズを返します")
            return min(self.initial_batch_size, 8)

        if self.device.type == "mps":
            # MPSの場合は控えめなバッチサイズから開始
            self.logger.info("MPS使用のため、制限されたバッチサイズ最適化を実行します")
            return min(self.initial_batch_size, 16)

        optimal_batch_size = self.initial_batch_size
        max_throughput = 0
        batch_size_results = {}

        # バイナリサーチで効率的に探索
        low, high = 1, self.max_batch_size
        tested_sizes = set()

        while low <= high:
            batch_size = (low + high) // 2

            # 2の累乗に丸める（効率的なメモリアライメントのため）
            batch_size = 2 ** int(np.log2(batch_size))

            if batch_size in tested_sizes:
                break
            tested_sizes.add(batch_size)

            self.logger.info(f"バッチサイズ {batch_size} をテスト中...")

            try:
                # メモリをクリア
                self._clear_gpu_memory()

                # データローダーを作成
                dataloader = dataloader_factory(batch_size)

                # スループットを測定
                throughput = self._measure_throughput(dataloader, loss_fn, mixed_precision)

                batch_size_results[batch_size] = {
                    "throughput": throughput,
                    "memory_used": self._get_gpu_memory_used(),
                }

                self.logger.info(
                    f"バッチサイズ {batch_size}: "
                    f"スループット={throughput:.1f} samples/sec, "
                    f"メモリ使用量={batch_size_results[batch_size]['memory_used'] / 1024**3:.1f}GB"
                )

                if throughput > max_throughput:
                    max_throughput = throughput
                    optimal_batch_size = batch_size

                # バイナリサーチの範囲を更新
                low = batch_size * 2

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"バッチサイズ {batch_size} でOOM発生")
                    high = batch_size // 2
                else:
                    self.logger.error(f"エラー発生: {e}")
                    break
            finally:
                self._clear_gpu_memory()

        # 結果をログ出力
        self._log_optimization_results(batch_size_results, optimal_batch_size)

        return optimal_batch_size

    def _measure_throughput(
        self,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        mixed_precision: bool,
    ) -> float:
        """
        スループットを測定

        Args:
            dataloader: データローダー
            loss_fn: 損失関数
            mixed_precision: 混合精度訓練を使用するか

        Returns:
            スループット (samples/sec)
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        total_samples = 0
        total_time = 0

        # GradScalerを混合精度訓練用に準備
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        for _trial in range(self.num_trials):
            batch_time = 0
            batch_samples = 0

            for i, batch in enumerate(dataloader):
                if i >= 10:  # 最初の10バッチのみ測定
                    break

                # データをデバイスに転送
                if isinstance(batch, dict):
                    inputs = batch["image"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                batch_size = inputs.size(0)

                # 順伝播と逆伝播の時間を測定
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

                optimizer.zero_grad()

                if mixed_precision and scaler is not None:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = loss_fn(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                batch_time += end_time - start_time
                batch_samples += batch_size

            if batch_samples > 0:
                total_time += batch_time
                total_samples += batch_samples

        return total_samples / total_time if total_time > 0 else 0

    def _clear_gpu_memory(self):
        """GPUメモリをクリア"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def _get_gpu_memory_used(self) -> int:
        """使用中のGPUメモリを取得"""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        return 0

    def _log_optimization_results(
        self, results: dict[int, dict[str, float]], optimal_batch_size: int
    ):
        """最適化結果をログ出力"""
        self.logger.info("\n=== バッチサイズ最適化結果 ===")
        for batch_size in sorted(results.keys()):
            result = results[batch_size]
            is_optimal = batch_size == optimal_batch_size
            marker = " ★" if is_optimal else ""
            self.logger.info(
                f"バッチサイズ {batch_size:4d}: "
                f"スループット={result['throughput']:7.1f} samples/sec, "
                f"メモリ={result['memory_used'] / 1024**3:5.2f}GB{marker}"
            )
        self.logger.info(f"\n最適バッチサイズ: {optimal_batch_size}")


class GradientAccumulator:
    """勾配累積によるバッチサイズ拡張"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
    ):
        """
        初期化

        Args:
            model: モデル
            optimizer: オプティマイザー
            accumulation_steps: 勾配累積ステップ数
            max_grad_norm: 勾配クリッピングの最大ノルム
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._step_count = 0

    def step(self, loss: torch.Tensor, mixed_precision_scaler=None):
        """
        勾配累積ステップを実行

        Args:
            loss: 損失値
            mixed_precision_scaler: 混合精度訓練用のスケーラー
        """
        # 累積のために損失を正規化
        loss = loss / self.accumulation_steps

        if mixed_precision_scaler is not None:
            mixed_precision_scaler.scale(loss).backward()
        else:
            loss.backward()

        self._step_count += 1

        # 累積ステップ数に達したら更新
        if self._step_count % self.accumulation_steps == 0:
            if mixed_precision_scaler is not None:
                mixed_precision_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                mixed_precision_scaler.step(self.optimizer)
                mixed_precision_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()

    @property
    def effective_batch_size(self) -> int:
        """実効バッチサイズを取得"""
        return self.accumulation_steps
