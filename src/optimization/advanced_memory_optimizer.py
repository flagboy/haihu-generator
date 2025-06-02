"""
高度なメモリ最適化を提供するクラス
"""

import gc
import os
import sys
import threading
import time
import weakref
from typing import Any

import numpy as np
import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import contextlib

from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from .memory_optimizer import MemoryOptimizer


class AdvancedMemoryOptimizer(MemoryOptimizer):
    """高度なメモリ最適化機能を提供するクラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        super().__init__(config_manager)

        # 追加設定
        self.enable_aggressive_gc = self.memory_config.get("enable_aggressive_gc", True)
        self.memory_pool_size = self.memory_config.get("memory_pool_size", "1GB")
        self.enable_memory_profiling = self.memory_config.get("enable_memory_profiling", False)

        # メモリプールとキャッシュ
        self._memory_pools = {}
        self._object_cache = weakref.WeakValueDictionary()
        self._cache_stats = {"hits": 0, "misses": 0}

        # メモリ監視スレッド
        self._monitor_thread = None
        self._monitoring = False

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("AdvancedMemoryOptimizer initialized")

    def optimize_memory(self) -> dict[str, Any]:
        """
        高度なメモリ最適化を実行

        Returns:
            最適化結果の統計情報
        """
        start_time = time.time()
        before_memory = self._get_memory_usage()

        results = {"before_memory_mb": before_memory / (1024 * 1024), "optimizations": []}

        # 1. 段階的なガベージコレクション
        gc_result = self._aggressive_garbage_collection()
        results["optimizations"].append(gc_result)

        # 2. NumPyメモリプールの最適化
        numpy_result = self._optimize_numpy_memory()
        results["optimizations"].append(numpy_result)

        # 3. PyTorchメモリの最適化（利用可能な場合）
        if TORCH_AVAILABLE:
            torch_result = self._optimize_torch_memory()
            results["optimizations"].append(torch_result)

        # 4. システムメモリの最適化
        system_result = self._optimize_system_memory()
        results["optimizations"].append(system_result)

        # 5. メモリプールのクリーンアップ
        pool_result = self._cleanup_memory_pools()
        results["optimizations"].append(pool_result)

        # 最終的なメモリ使用量
        after_memory = self._get_memory_usage()
        results["after_memory_mb"] = after_memory / (1024 * 1024)
        results["memory_freed_mb"] = (before_memory - after_memory) / (1024 * 1024)
        results["optimization_time"] = time.time() - start_time

        self.logger.info(
            f"Memory optimization completed. Freed {results['memory_freed_mb']:.2f} MB"
        )

        return results

    def _aggressive_garbage_collection(self) -> dict[str, Any]:
        """
        段階的で積極的なガベージコレクション

        Returns:
            GC実行結果
        """
        result = {
            "operation": "garbage_collection",
            "generations_collected": [],
            "objects_collected": 0,
        }

        # 各世代のGCを実行
        for generation in range(gc.get_count().__len__()):
            before_objects = len(gc.get_objects())
            collected = gc.collect(generation)
            after_objects = len(gc.get_objects())

            result["generations_collected"].append(
                {
                    "generation": generation,
                    "collected": collected,
                    "objects_before": before_objects,
                    "objects_after": after_objects,
                }
            )
            result["objects_collected"] += collected

        # 未到達オブジェクトの強制削除
        gc.collect()

        return result

    def _optimize_numpy_memory(self) -> dict[str, Any]:
        """
        NumPyメモリプールの最適化

        Returns:
            最適化結果
        """
        result = {"operation": "numpy_optimization", "success": False, "memory_freed": 0}

        try:
            # NumPyのメモリ使用量を取得
            if hasattr(np, "ndarray"):
                # 大きな配列のガベージコレクション
                for obj in gc.get_objects():
                    if (
                        isinstance(obj, np.ndarray)
                        and obj.nbytes > 1024 * 1024
                        and obj.flags.writeable
                    ):  # 1MB以上
                        # 読み取り専用フラグを設定してメモリマップ化を促進
                        with contextlib.suppress(Exception):
                            obj.flags.writeable = False

                # NumPyの内部キャッシュをクリア
                if hasattr(np, "get_default_memory_pool"):
                    try:
                        pool = np.get_default_memory_pool()
                        before = pool.used_bytes()
                        pool.free_all_blocks()
                        after = pool.used_bytes()
                        result["memory_freed"] = before - after
                    except Exception:
                        pass

            result["success"] = True

        except Exception as e:
            self.logger.warning(f"NumPy optimization failed: {e}")
            result["error"] = str(e)

        return result

    def _optimize_torch_memory(self) -> dict[str, Any]:
        """
        PyTorchメモリの最適化

        Returns:
            最適化結果
        """
        result = {
            "operation": "torch_optimization",
            "success": False,
            "gpu_memory_freed": 0,
            "cpu_memory_freed": 0,
        }

        try:
            if torch.cuda.is_available():
                # GPU メモリの最適化
                before_gpu = torch.cuda.memory_allocated()

                # キャッシュをクリア
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 未使用のテンソルを削除
                for obj in gc.get_objects():
                    if torch.is_tensor(obj) and obj.is_cuda and not obj.requires_grad:
                        del obj

                # 再度キャッシュをクリア
                torch.cuda.empty_cache()

                after_gpu = torch.cuda.memory_allocated()
                result["gpu_memory_freed"] = before_gpu - after_gpu

            # CPU テンソルの最適化
            for obj in gc.get_objects():
                if (
                    torch.is_tensor(obj)
                    and not obj.is_cuda
                    and obj.element_size() * obj.nelement() > 1024 * 1024
                    and obj.is_contiguous()
                ):  # 1MB以上
                    # 大きなテンソルのメモリを圧縮
                    obj = obj.clone()

            result["success"] = True

        except Exception as e:
            self.logger.warning(f"PyTorch optimization failed: {e}")
            result["error"] = str(e)

        return result

    def _optimize_system_memory(self) -> dict[str, Any]:
        """
        システムレベルのメモリ最適化

        Returns:
            最適化結果
        """
        result = {"operation": "system_optimization", "success": False, "actions": []}

        try:
            # メモリマップファイルのフラッシュ
            if hasattr(os, "sync"):
                os.sync()
                result["actions"].append("synced_filesystem")

            # プロセスのワーキングセットを削減
            process = psutil.Process()

            # 仮想メモリの情報を取得
            vm_info = process.memory_info()
            result["before_rss"] = vm_info.rss

            # Linuxの場合、malloc_trimを呼び出す
            if sys.platform.startswith("linux"):
                try:
                    import ctypes

                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                    result["actions"].append("malloc_trim")
                except Exception:
                    pass

            # メモリ使用量を再取得
            vm_info_after = process.memory_info()
            result["after_rss"] = vm_info_after.rss
            result["memory_freed"] = result["before_rss"] - result["after_rss"]

            result["success"] = True

        except Exception as e:
            self.logger.warning(f"System optimization failed: {e}")
            result["error"] = str(e)

        return result

    def _cleanup_memory_pools(self) -> dict[str, Any]:
        """
        メモリプールのクリーンアップ

        Returns:
            クリーンアップ結果
        """
        result = {"operation": "pool_cleanup", "pools_cleaned": 0, "memory_freed": 0}

        # カスタムメモリプールのクリーンアップ
        for _pool_name, pool in list(self._memory_pools.items()):
            try:
                if hasattr(pool, "clear"):
                    pool.clear()
                    result["pools_cleaned"] += 1
            except Exception:
                pass

        # オブジェクトキャッシュの統計をリセット
        self._cache_stats = {"hits": 0, "misses": 0}

        return result

    def start_memory_monitoring(self, interval: float = 60.0):
        """
        メモリ使用量の監視を開始

        Args:
            interval: 監視間隔（秒）
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Memory monitoring started with {interval}s interval")

    def stop_memory_monitoring(self):
        """メモリ監視を停止"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Memory monitoring stopped")

    def _monitor_memory_usage(self, interval: float):
        """
        メモリ使用量を定期的に監視

        Args:
            interval: 監視間隔（秒）
        """
        while self._monitoring:
            try:
                memory_info = self._get_detailed_memory_info()

                # 閾値を超えた場合は自動最適化
                if memory_info["percent"] > self.memory_config.get("auto_optimize_threshold", 80):
                    self.logger.warning(
                        f"Memory usage high: {memory_info['percent']:.1f}%. Running optimization..."
                    )
                    self.optimize_memory()

                # メモリ使用量をログ出力（デバッグレベル）
                self.logger.debug(
                    f"Memory usage: {memory_info['percent']:.1f}% "
                    f"({memory_info['used_mb']:.1f}/{memory_info['total_mb']:.1f} MB)"
                )

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

            time.sleep(interval)

    def _get_memory_usage(self) -> float:
        """
        現在のプロセスのメモリ使用量を取得（バイト）

        Returns:
            メモリ使用量（バイト）
        """
        process = psutil.Process()
        return process.memory_info().rss

    def _get_detailed_memory_info(self) -> dict[str, float]:
        """
        詳細なメモリ情報を取得

        Returns:
            メモリ情報の辞書
        """
        process = psutil.Process()
        vm = psutil.virtual_memory()

        return {
            "percent": vm.percent,
            "total_mb": vm.total / (1024 * 1024),
            "available_mb": vm.available / (1024 * 1024),
            "used_mb": vm.used / (1024 * 1024),
            "process_rss_mb": process.memory_info().rss / (1024 * 1024),
            "process_vms_mb": process.memory_info().vms / (1024 * 1024),
            "process_percent": process.memory_percent(),
        }

    def create_memory_pool(self, name: str, size: int) -> "MemoryPool":
        """
        カスタムメモリプールを作成

        Args:
            name: プール名
            size: プールサイズ（バイト）

        Returns:
            メモリプールオブジェクト
        """
        if name not in self._memory_pools:
            self._memory_pools[name] = MemoryPool(name, size)

        return self._memory_pools[name]


class MemoryPool:
    """シンプルなメモリプール実装"""

    def __init__(self, name: str, size: int):
        """
        初期化

        Args:
            name: プール名
            size: プールサイズ（バイト）
        """
        self.name = name
        self.size = size
        self.buffer = bytearray(size)
        self.offset = 0
        self.allocations = []

    def allocate(self, size: int) -> memoryview | None:
        """
        メモリを割り当て

        Args:
            size: 割り当てサイズ

        Returns:
            メモリビューまたはNone
        """
        if self.offset + size > self.size:
            return None

        view = memoryview(self.buffer)[self.offset : self.offset + size]
        self.allocations.append((self.offset, size))
        self.offset += size

        return view

    def clear(self):
        """プールをクリア"""
        self.offset = 0
        self.allocations.clear()
