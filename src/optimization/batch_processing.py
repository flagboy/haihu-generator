"""
バッチ処理最適化モジュール
効率的なバッチ処理とメモリ管理を提供
"""

import gc
import multiprocessing as mp
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.device_utils import get_available_device, get_device_memory_info
from src.utils.logger import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class OptimizedBatchProcessor:
    """最適化されたバッチ処理クラス"""

    def __init__(
        self,
        batch_size: int | None = None,
        auto_optimize: bool = True,
        memory_fraction: float = 0.85,
        num_workers: int | None = None,
        device: str | None = None,
    ):
        """
        Args:
            batch_size: バッチサイズ（Noneの場合は自動最適化）
            auto_optimize: 自動最適化を有効にするか
            memory_fraction: 使用するメモリの割合（0.0-1.0）
            num_workers: ワーカー数（Noneの場合は自動設定）
            device: 使用デバイス（Noneの場合は自動選択）
        """
        self.device = device or get_available_device()
        self.auto_optimize = auto_optimize
        self.memory_fraction = memory_fraction

        # バッチサイズの設定
        if batch_size is None and auto_optimize:
            self.batch_size = self._calculate_optimal_batch_size()
        else:
            self.batch_size = batch_size or 32

        # ワーカー数の設定
        if num_workers is None:
            self.num_workers = self._get_optimal_workers()
        else:
            self.num_workers = num_workers

        # 非同期転送の設定
        self.non_blocking = torch.cuda.is_available()

        logger.info(
            f"OptimizedBatchProcessor初期化: "
            f"device={self.device}, batch_size={self.batch_size}, "
            f"num_workers={self.num_workers}"
        )

    def _calculate_optimal_batch_size(self) -> int:
        """最適なバッチサイズを計算"""
        try:
            memory_info = get_device_memory_info(self.device)
            if not memory_info or memory_info.get("free", 0) == 0:
                return 32  # デフォルト値

            # 利用可能なメモリから推定
            available_memory = (
                memory_info["free"] * self.memory_fraction * 1024 * 1024 * 1024
            )  # GBからバイトに変換
            # 1サンプルあたり約100MBと仮定（画像処理の場合）
            estimated_sample_size = 100 * 1024 * 1024  # 100MB
            optimal_size = int(available_memory / estimated_sample_size)

            # 範囲を制限
            optimal_size = max(8, min(128, optimal_size))
            # 8の倍数に調整（効率的なGPU処理のため）
            optimal_size = (optimal_size // 8) * 8

            logger.info(f"最適バッチサイズを計算: {optimal_size}")
            return optimal_size

        except Exception as e:
            logger.warning(f"バッチサイズ計算エラー: {e}")
            return 32

    def _get_optimal_workers(self) -> int:
        """最適なワーカー数を取得"""
        cpu_count = mp.cpu_count()

        if torch.cuda.is_available():
            # GPU使用時は少なめのワーカー
            return min(4, cpu_count // 2)
        else:
            # CPU使用時は多めのワーカー
            return min(8, cpu_count - 1)

    def create_dataloader(
        self,
        dataset: Any,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs: Any,
    ) -> DataLoader:
        """最適化されたDataLoaderを作成"""
        # pin_memoryの設定（CUDAのみ有効）
        pin_memory = (
            torch.cuda.is_available()
            and (self.device.type if hasattr(self.device, "type") else str(self.device)) != "cpu"
        )

        # persistent_workersの設定（ワーカーが複数の場合のみ）
        persistent_workers = self.num_workers > 0

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=drop_last,
            **kwargs,
        )

        return dataloader

    def transfer_to_device(self, data: Any, non_blocking: bool | None = None) -> Any:
        """データをデバイスに転送（非同期対応）"""
        if non_blocking is None:
            non_blocking = self.non_blocking

        if isinstance(data, dict):
            return {k: self.transfer_to_device(v, non_blocking) for k, v in data.items()}
        elif isinstance(data, list | tuple):
            return type(data)(self.transfer_to_device(item, non_blocking) for item in data)
        elif hasattr(data, "to"):
            return data.to(self.device, non_blocking=non_blocking)
        else:
            return data

    def process_in_batches(
        self,
        data: list[T],
        process_fn: callable,
        show_progress: bool = True,
    ) -> list[Any]:
        """データをバッチ単位で処理"""
        results = []
        total_batches = len(data) // self.batch_size + (1 if len(data) % self.batch_size > 0 else 0)

        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]

            if show_progress:
                current_batch = i // self.batch_size + 1
                logger.info(f"バッチ処理中: {current_batch}/{total_batches}")

            # メモリ最適化
            if i > 0 and i % (self.batch_size * 10) == 0:
                self.optimize_memory()

            # バッチ処理
            batch_results = process_fn(batch)
            results.extend(batch_results)

        return results

    def stream_batches(
        self, data: list[T], shuffle: bool = False
    ) -> Generator[list[T], None, None]:
        """バッチをストリーミングで生成（メモリ効率的）"""
        indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield [data[idx] for idx in batch_indices]

    def optimize_memory(self) -> None:
        """メモリを最適化"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_memory_stats(self) -> dict[str, Any]:
        """メモリ使用状況を取得"""
        return get_device_memory_info(self.device)


class ParallelBatchProcessor(OptimizedBatchProcessor):
    """並列バッチ処理クラス"""

    def __init__(self, max_workers: int | None = None, **kwargs: Any):
        """
        Args:
            max_workers: 最大ワーカー数
            **kwargs: OptimizedBatchProcessorの引数
        """
        super().__init__(**kwargs)
        self.max_workers = max_workers or self.num_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def process_parallel_batches(
        self,
        data: list[T],
        process_fn: callable,
        chunk_size: int | None = None,
    ) -> list[Any]:
        """並列でバッチ処理を実行"""
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.max_workers * 4))

        # データをチャンクに分割
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        # 並列処理
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self.process_in_batches, chunk, process_fn, False)
            futures.append(future)

        # 結果を収集
        results = []
        for i, future in enumerate(futures):
            logger.info(f"並列バッチ処理: {i + 1}/{len(futures)}")
            chunk_results = future.result()
            results.extend(chunk_results)

        return results

    def __del__(self):
        """リソースのクリーンアップ"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)


class BatchSizeOptimizer:
    """バッチサイズの動的最適化クラス"""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 128,
        growth_factor: float = 1.5,
        reduction_factor: float = 0.7,
    ):
        """
        Args:
            initial_batch_size: 初期バッチサイズ
            min_batch_size: 最小バッチサイズ
            max_batch_size: 最大バッチサイズ
            growth_factor: 成長係数
            reduction_factor: 縮小係数
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.growth_factor = growth_factor
        self.reduction_factor = reduction_factor
        self.history: list[tuple[int, bool]] = []

    def update(self, success: bool, memory_error: bool = False) -> int:
        """バッチサイズを更新"""
        self.history.append((self.current_batch_size, success))

        if memory_error:
            # メモリエラーの場合は大幅に削減
            self.current_batch_size = int(self.current_batch_size * self.reduction_factor)
        elif success and len(self.history) > 3:
            # 直近3回成功したら増加を試みる（現在の結果を含めて4回分の履歴が必要）
            recent_success = all(h[1] for h in self.history[-3:])
            if recent_success:
                self.current_batch_size = int(self.current_batch_size * self.growth_factor)
        elif not success:
            # 失敗したら減少
            self.current_batch_size = int(self.current_batch_size * self.reduction_factor)

        # 範囲制限
        self.current_batch_size = max(
            self.min_batch_size, min(self.max_batch_size, self.current_batch_size)
        )

        # 8の倍数に調整
        self.current_batch_size = (self.current_batch_size // 8) * 8 or 8

        logger.info(f"バッチサイズを更新: {self.current_batch_size}")
        return self.current_batch_size

    def find_optimal_batch_size(self, test_fn: callable, max_iterations: int = 10) -> int:
        """バイナリサーチで最適なバッチサイズを見つける"""
        low = self.min_batch_size
        high = self.max_batch_size
        optimal = low

        for _ in range(max_iterations):
            mid = (low + high) // 2
            mid = (mid // 8) * 8 or 8  # 8の倍数に調整

            try:
                # テスト実行
                success = test_fn(mid)
                if success:
                    optimal = mid
                    low = mid + 8
                else:
                    high = mid - 8
            except Exception as e:
                logger.warning(f"バッチサイズ {mid} でエラー: {e}")
                high = mid - 8

            if low > high:
                break

        self.current_batch_size = optimal
        logger.info(f"最適バッチサイズを発見: {optimal}")
        return optimal
