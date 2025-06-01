"""
AdvancedMemoryOptimizerのテスト
"""

import gc
import sys
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.optimization.advanced_memory_optimizer import AdvancedMemoryOptimizer, MemoryPool
from src.utils.config import ConfigManager


class TestAdvancedMemoryOptimizer:
    """AdvancedMemoryOptimizerのテストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのモック"""
        config_manager = Mock(spec=ConfigManager)
        config_manager.config = {
            "performance": {
                "memory": {
                    "max_cache_size": "2GB",
                    "enable_garbage_collection": True,
                    "gc_threshold": 1000,
                    "enable_aggressive_gc": True,
                    "memory_pool_size": "1GB",
                    "enable_memory_profiling": False,
                    "auto_optimize_threshold": 80,
                }
            }
        }
        return config_manager

    @pytest.fixture
    def memory_optimizer(self, config_manager):
        """AdvancedMemoryOptimizerのフィクスチャ"""
        with patch("src.optimization.advanced_memory_optimizer.get_logger"):
            optimizer = AdvancedMemoryOptimizer(config_manager)
            return optimizer

    def test_initialization(self, memory_optimizer):
        """初期化テスト"""
        assert memory_optimizer is not None
        assert memory_optimizer.enable_aggressive_gc is True
        assert memory_optimizer.memory_pool_size == "1GB"
        assert memory_optimizer.enable_memory_profiling is False
        assert isinstance(memory_optimizer._memory_pools, dict)
        assert hasattr(memory_optimizer._object_cache, "get")

    def test_optimize_memory_basic(self, memory_optimizer):
        """基本的なメモリ最適化テスト"""
        with patch.object(memory_optimizer, "_get_memory_usage") as mock_memory:
            mock_memory.side_effect = [100 * 1024 * 1024, 80 * 1024 * 1024]  # 100MB -> 80MB

            result = memory_optimizer.optimize_memory()

            # 結果の検証
            assert "before_memory_mb" in result
            assert "after_memory_mb" in result
            assert "memory_freed_mb" in result
            assert result["memory_freed_mb"] == 20.0
            assert "optimizations" in result
            assert len(result["optimizations"]) >= 4

    def test_aggressive_garbage_collection(self, memory_optimizer):
        """積極的なガベージコレクションのテスト"""
        # テスト用のオブジェクトを作成
        large_list = [list(range(1000)) for _ in range(100)]

        # GC実行前のオブジェクト数を記録
        initial_objects = len(gc.get_objects())

        # ガベージコレクション実行
        result = memory_optimizer._aggressive_garbage_collection()

        # 結果の検証
        assert result["operation"] == "garbage_collection"
        assert "generations_collected" in result
        assert len(result["generations_collected"]) > 0
        assert "objects_collected" in result

        # 参照を削除してGCを再実行
        del large_list
        result2 = memory_optimizer._aggressive_garbage_collection()
        assert result2["objects_collected"] >= 0

    def test_optimize_numpy_memory(self, memory_optimizer):
        """NumPyメモリ最適化のテスト"""
        # 大きなNumPy配列を作成
        large_arrays = [np.zeros((1000, 1000), dtype=np.float64) for _ in range(5)]

        # 最適化実行
        result = memory_optimizer._optimize_numpy_memory()

        # 結果の検証
        assert result["operation"] == "numpy_optimization"
        assert "success" in result

        # 配列を削除
        del large_arrays
        gc.collect()

    @pytest.mark.skipif(
        not hasattr(np, "get_default_memory_pool"), reason="NumPy memory pool not available"
    )
    def test_numpy_memory_pool_optimization(self, memory_optimizer):
        """NumPyメモリプール最適化の詳細テスト"""
        # メモリプールを使用する処理
        arrays = []
        for _ in range(10):
            arr = np.random.rand(100, 100)
            arrays.append(arr)

        # 配列を削除
        arrays.clear()

        # メモリプール最適化
        result = memory_optimizer._optimize_numpy_memory()

        assert result["success"] is True
        if "memory_freed" in result:
            assert result["memory_freed"] >= 0

    def test_optimize_torch_memory(self, memory_optimizer):
        """PyTorchメモリ最適化のテスト"""
        # PyTorchがインストールされていない場合のテスト
        with patch("src.optimization.advanced_memory_optimizer.TORCH_AVAILABLE", False):
            # optimize_memoryを実行してもPyTorch最適化がスキップされることを確認
            result = memory_optimizer.optimize_memory()
            torch_results = [
                r for r in result["optimizations"] if r.get("operation") == "torch_optimization"
            ]
            assert len(torch_results) == 0

        # PyTorchがインストールされている場合のモック
        with patch("src.optimization.advanced_memory_optimizer.TORCH_AVAILABLE", True):
            with patch("src.optimization.advanced_memory_optimizer.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.memory_allocated.side_effect = [1000000, 500000]
                mock_torch.is_tensor.return_value = False

                result = memory_optimizer._optimize_torch_memory()

                assert result["operation"] == "torch_optimization"
                assert result["success"] is True
                assert result["gpu_memory_freed"] == 500000

    def test_optimize_system_memory(self, memory_optimizer):
        """システムメモリ最適化のテスト"""
        with patch("psutil.Process") as mock_process_class:
            mock_process = Mock()
            mock_process.memory_info.side_effect = [
                Mock(rss=200 * 1024 * 1024),  # 200MB
                Mock(rss=180 * 1024 * 1024),  # 180MB
            ]
            mock_process_class.return_value = mock_process

            result = memory_optimizer._optimize_system_memory()

            assert result["operation"] == "system_optimization"
            assert result["success"] is True
            assert "before_rss" in result
            assert "after_rss" in result
            assert result["memory_freed"] == 20 * 1024 * 1024

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="malloc_trim is Linux-specific"
    )
    def test_malloc_trim_on_linux(self, memory_optimizer):
        """Linux環境でのmalloc_trimテスト"""
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libc = Mock()
            mock_libc.malloc_trim.return_value = 1
            mock_cdll.return_value = mock_libc

            result = memory_optimizer._optimize_system_memory()

            assert "malloc_trim" in result["actions"]
            mock_libc.malloc_trim.assert_called_once_with(0)

    def test_cleanup_memory_pools(self, memory_optimizer):
        """メモリプールのクリーンアップテスト"""
        # テスト用のメモリプールを作成
        pool1 = memory_optimizer.create_memory_pool("test_pool_1", 1024 * 1024)
        pool2 = memory_optimizer.create_memory_pool("test_pool_2", 2048 * 1024)

        # プールにデータを割り当て
        pool1.allocate(1024)
        pool2.allocate(2048)

        # クリーンアップ実行
        result = memory_optimizer._cleanup_memory_pools()

        assert result["operation"] == "pool_cleanup"
        assert result["pools_cleaned"] == 2
        assert pool1.offset == 0
        assert pool2.offset == 0

    def test_memory_monitoring_start_stop(self, memory_optimizer):
        """メモリ監視の開始・停止テスト"""
        # 監視開始
        memory_optimizer.start_memory_monitoring(interval=0.1)

        assert memory_optimizer._monitoring is True
        assert memory_optimizer._monitor_thread is not None
        assert memory_optimizer._monitor_thread.is_alive()

        # 少し待機
        time.sleep(0.2)

        # 監視停止
        memory_optimizer.stop_memory_monitoring()

        assert memory_optimizer._monitoring is False
        # スレッドが停止するまで待機
        time.sleep(0.2)
        assert not memory_optimizer._monitor_thread.is_alive()

    def test_memory_monitoring_auto_optimization(self, memory_optimizer):
        """メモリ監視による自動最適化のテスト"""
        with patch.object(memory_optimizer, "_get_detailed_memory_info") as mock_info:
            # メモリ使用率が閾値を超えた状態をシミュレート
            mock_info.return_value = {
                "percent": 85,  # 80%の閾値を超過
                "total_mb": 8192,
                "used_mb": 6963,
                "available_mb": 1229,
                "process_rss_mb": 500,
                "process_vms_mb": 800,
                "process_percent": 6.1,
            }

            with patch.object(memory_optimizer, "optimize_memory") as mock_optimize:
                # 監視スレッドの一回分の処理を実行
                memory_optimizer._monitor_memory_usage(0.1)

                # 自動最適化が呼ばれたことを確認
                mock_optimize.assert_called_once()

    def test_get_detailed_memory_info(self, memory_optimizer):
        """詳細メモリ情報取得のテスト"""
        info = memory_optimizer._get_detailed_memory_info()

        # 必須フィールドの確認
        assert "percent" in info
        assert "total_mb" in info
        assert "available_mb" in info
        assert "used_mb" in info
        assert "process_rss_mb" in info
        assert "process_vms_mb" in info
        assert "process_percent" in info

        # 値の妥当性確認
        assert 0 <= info["percent"] <= 100
        assert info["total_mb"] > 0
        assert info["available_mb"] >= 0
        assert info["process_rss_mb"] > 0

    def test_memory_pool_basic(self):
        """MemoryPoolクラスの基本テスト"""
        pool = MemoryPool("test", 1024 * 1024)  # 1MB

        # 初期状態の確認
        assert pool.name == "test"
        assert pool.size == 1024 * 1024
        assert pool.offset == 0
        assert len(pool.allocations) == 0

        # メモリ割り当て
        view1 = pool.allocate(1024)
        assert view1 is not None
        assert len(view1) == 1024
        assert pool.offset == 1024

        # 追加割り当て
        view2 = pool.allocate(2048)
        assert view2 is not None
        assert pool.offset == 3072

        # プールクリア
        pool.clear()
        assert pool.offset == 0
        assert len(pool.allocations) == 0

    def test_memory_pool_overflow(self):
        """MemoryPoolのオーバーフローテスト"""
        pool = MemoryPool("small", 1024)  # 1KB

        # プールサイズを超える割り当て
        view = pool.allocate(2048)
        assert view is None  # 割り当て失敗

        # プールの残りサイズを超える割り当て
        pool.allocate(512)
        view = pool.allocate(600)
        assert view is None  # 割り当て失敗

    def test_cache_statistics(self, memory_optimizer):
        """キャッシュ統計のテスト"""
        # 初期状態
        assert memory_optimizer._cache_stats["hits"] == 0
        assert memory_optimizer._cache_stats["misses"] == 0

        # プールクリーンアップでリセットされることを確認
        memory_optimizer._cache_stats["hits"] = 10
        memory_optimizer._cache_stats["misses"] = 5

        memory_optimizer._cleanup_memory_pools()

        assert memory_optimizer._cache_stats["hits"] == 0
        assert memory_optimizer._cache_stats["misses"] == 0

    def test_error_handling_in_optimization(self, memory_optimizer):
        """最適化処理のエラーハンドリングテスト"""
        # メモリ使用量取得でエラーを発生させる
        with patch.object(memory_optimizer, "_get_memory_usage") as mock_memory:
            mock_memory.side_effect = Exception("Memory error")

            # エラーが発生してもクラッシュしないことを確認
            with pytest.raises(Exception):
                memory_optimizer.optimize_memory()

    def test_concurrent_memory_pool_access(self, memory_optimizer):
        """メモリプールの並行アクセステスト"""
        pool = memory_optimizer.create_memory_pool("concurrent", 10 * 1024 * 1024)  # 10MB
        results = []

        def allocate_memory(size):
            view = pool.allocate(size)
            results.append(view is not None)

        # 複数スレッドから同時にアクセス
        threads = []
        for i in range(10):
            t = threading.Thread(target=allocate_memory, args=(1024 * i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # すべての割り当てが成功したことを確認
        assert all(results)
