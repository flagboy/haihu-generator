"""
バッチ処理最適化のテスト
"""

import time
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.optimization.batch_processing import (
    BatchSizeOptimizer,
    OptimizedBatchProcessor,
    ParallelBatchProcessor,
)


class TestOptimizedBatchProcessor:
    """OptimizedBatchProcessorのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        processor = OptimizedBatchProcessor(batch_size=64)
        assert processor.batch_size == 64
        assert str(
            processor.device.type if hasattr(processor.device, "type") else processor.device
        ) in ["cpu", "cuda", "mps"]

    def test_auto_batch_size_calculation(self):
        """自動バッチサイズ計算のテスト"""
        with patch("src.optimization.batch_processing.get_device_memory_info") as mock_memory:
            # 十分なメモリがある場合（GB単位で返される）
            mock_memory.return_value = {
                "allocated": 8.0,  # 8GB
                "reserved": 16.0,  # 16GB
                "free": 8.0,  # 8GB
            }

            processor = OptimizedBatchProcessor(batch_size=None, auto_optimize=True)
            # 8の倍数で、8-128の範囲内
            assert processor.batch_size % 8 == 0
            assert 8 <= processor.batch_size <= 128

    def test_optimal_workers_calculation(self):
        """最適ワーカー数計算のテスト"""
        processor = OptimizedBatchProcessor()

        # GPU使用時
        with patch("torch.cuda.is_available", return_value=True):
            workers = processor._get_optimal_workers()
            assert workers <= 4

        # CPU使用時
        with patch("torch.cuda.is_available", return_value=False):
            workers = processor._get_optimal_workers()
            assert workers <= 8

    def test_transfer_to_device(self):
        """デバイス転送のテスト"""
        processor = OptimizedBatchProcessor(device="cpu")

        # NumPy配列
        arr = np.random.rand(10, 10)
        tensor = torch.from_numpy(arr)
        transferred = processor.transfer_to_device(tensor)
        assert transferred.device.type == "cpu"

        # 辞書型
        data = {"image": tensor, "label": torch.tensor([1, 2, 3])}
        transferred = processor.transfer_to_device(data)
        assert isinstance(transferred, dict)
        assert transferred["image"].device.type == "cpu"

        # リスト型
        data_list = [tensor, tensor.clone()]
        transferred = processor.transfer_to_device(data_list)
        assert isinstance(transferred, list)
        assert all(t.device.type == "cpu" for t in transferred)

    def test_process_in_batches(self):
        """バッチ処理のテスト"""
        processor = OptimizedBatchProcessor(batch_size=10)

        # テストデータ
        data = list(range(25))

        # 処理関数
        def process_fn(batch):
            return [x * 2 for x in batch]

        results = processor.process_in_batches(data, process_fn, show_progress=False)

        assert len(results) == 25
        assert results == [x * 2 for x in data]

    def test_stream_batches(self):
        """バッチストリーミングのテスト"""
        processor = OptimizedBatchProcessor(batch_size=10)
        data = list(range(25))

        batches = list(processor.stream_batches(data))
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

        # シャッフルテスト
        batches_shuffled = list(processor.stream_batches(data, shuffle=True))
        assert len(batches_shuffled) == 3
        # 要素の合計は同じ
        all_elements = sum(batches_shuffled, [])
        assert sorted(all_elements) == sorted(data)

    def test_memory_optimization(self):
        """メモリ最適化のテスト"""
        processor = OptimizedBatchProcessor()

        # エラーが発生しないことを確認
        processor.optimize_memory()

        # GPUの場合の動作確認
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("torch.cuda.synchronize") as mock_sync,
        ):
            processor.optimize_memory()
            mock_empty_cache.assert_called_once()
            mock_sync.assert_called_once()

    def test_dataloader_creation(self):
        """DataLoader作成のテスト"""
        processor = OptimizedBatchProcessor(batch_size=32, num_workers=2)

        # ダミーデータセット
        dataset = [(i, np.random.rand(32, 32, 3)) for i in range(100)]

        dataloader = processor.create_dataloader(dataset)
        assert dataloader.batch_size == 32
        assert dataloader.num_workers == 2

        # pin_memoryの確認（CUDA利用可能時のみ）
        if torch.cuda.is_available():
            assert dataloader.pin_memory is True


class TestParallelBatchProcessor:
    """ParallelBatchProcessorのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        processor = ParallelBatchProcessor(max_workers=4, batch_size=32)
        assert processor.max_workers == 4
        assert processor.batch_size == 32

    def test_parallel_processing(self):
        """並列処理のテスト"""
        processor = ParallelBatchProcessor(max_workers=2, batch_size=10)

        # テストデータ
        data = list(range(50))

        # 処理関数（少し時間がかかる処理）
        def process_fn(batch):
            time.sleep(0.01)  # 10ms
            return [x**2 for x in batch]

        # 並列処理
        start_time = time.time()
        results = processor.process_parallel_batches(data, process_fn)
        parallel_time = time.time() - start_time

        # 結果の検証
        assert len(results) == 50
        assert sorted(results) == [x**2 for x in data]

        # 並列化による高速化を確認（完全な線形スケーリングは期待しない）
        # 注：テスト環境や負荷によってばらつきがあるため、緩い条件にする
        sequential_time = 0.01 * 5  # 5バッチ × 10ms
        # 並列処理のオーバーヘッドを考慮して、1.5倍以内なら許容
        assert parallel_time < sequential_time * 1.5

    def test_cleanup(self):
        """リソースクリーンアップのテスト"""
        processor = ParallelBatchProcessor(max_workers=2)
        executor = processor.executor

        # デストラクタの呼び出し
        del processor

        # executorがシャットダウンされていることを確認
        assert executor._shutdown


class TestBatchSizeOptimizer:
    """BatchSizeOptimizerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        optimizer = BatchSizeOptimizer(initial_batch_size=32)
        assert optimizer.current_batch_size == 32
        assert optimizer.min_batch_size == 8
        assert optimizer.max_batch_size == 128

    def test_update_on_success(self):
        """成功時の更新テスト"""
        optimizer = BatchSizeOptimizer(initial_batch_size=32)

        # 3回連続成功で増加
        for _ in range(3):
            new_size = optimizer.update(success=True)
            assert new_size == 32  # まだ増加しない

        # 4回目で増加
        new_size = optimizer.update(success=True)
        assert new_size > 32
        assert new_size % 8 == 0

    def test_update_on_failure(self):
        """失敗時の更新テスト"""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        # 失敗で減少
        new_size = optimizer.update(success=False)
        assert new_size < 64
        assert new_size % 8 == 0

    def test_update_on_memory_error(self):
        """メモリエラー時の更新テスト"""
        optimizer = BatchSizeOptimizer(initial_batch_size=64)

        # メモリエラーで大幅減少
        new_size = optimizer.update(success=False, memory_error=True)
        # 64 * 0.7 = 44.8 → 44 → 8の倍数に調整して40
        assert new_size == 40
        assert new_size % 8 == 0

    def test_boundary_conditions(self):
        """境界条件のテスト"""
        optimizer = BatchSizeOptimizer(initial_batch_size=120, min_batch_size=8, max_batch_size=128)

        # 最大値を超えない
        for _ in range(10):
            optimizer.update(success=True)
        assert optimizer.current_batch_size <= 128

        # 最小値を下回らない
        optimizer.current_batch_size = 16
        for _ in range(10):
            optimizer.update(success=False)
        assert optimizer.current_batch_size >= 8

    def test_find_optimal_batch_size(self):
        """最適バッチサイズ探索のテスト"""
        optimizer = BatchSizeOptimizer()

        # テスト関数（64まで成功、それ以上は失敗）
        def test_fn(batch_size):
            return batch_size <= 64

        optimal = optimizer.find_optimal_batch_size(test_fn)
        assert optimal == 64
        assert optimal % 8 == 0

    def test_find_optimal_with_exception(self):
        """例外発生時の最適バッチサイズ探索テスト"""
        optimizer = BatchSizeOptimizer()

        # テスト関数（48で例外発生）
        def test_fn(batch_size):
            if batch_size > 48:
                raise RuntimeError("Out of memory")
            return True

        optimal = optimizer.find_optimal_batch_size(test_fn)
        assert optimal <= 48
        assert optimal % 8 == 0


@pytest.mark.integration
class TestIntegration:
    """統合テスト"""

    def test_batch_processing_pipeline(self):
        """バッチ処理パイプラインの統合テスト"""
        # プロセッサーを作成
        processor = OptimizedBatchProcessor(batch_size=16, auto_optimize=False)

        # ダミーデータ（画像のシミュレーション）
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(100)]

        # 処理関数（簡単な変換）
        def transform(batch):
            return [img.mean(axis=(0, 1)) for img in batch]

        # バッチ処理
        results = processor.process_in_batches(images, transform, show_progress=False)

        assert len(results) == 100
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == (3,) for r in results)

    def test_parallel_batch_optimization(self):
        """並列バッチ最適化の統合テスト"""
        # 並列プロセッサー
        processor = ParallelBatchProcessor(max_workers=2, batch_size=20)

        # バッチサイズ最適化
        optimizer = BatchSizeOptimizer(initial_batch_size=20)

        # データ
        data = list(range(200))

        # 処理関数
        def process_fn(batch):
            # バッチサイズに応じて処理時間が変わるシミュレーション
            if len(batch) > 50:
                raise MemoryError("Batch too large")
            time.sleep(0.001 * len(batch))
            return [x * 2 for x in batch]

        # 最適なバッチサイズを探索
        def test_batch_size(size):
            try:
                test_data = data[:size]
                processor.batch_size = size
                processor.process_in_batches(test_data, process_fn, show_progress=False)
                return True
            except MemoryError:
                return False

        optimal_size = optimizer.find_optimal_batch_size(test_batch_size, max_iterations=5)
        assert optimal_size <= 50
        assert optimal_size % 8 == 0

        # 最適サイズで処理
        processor.batch_size = optimal_size
        results = processor.process_parallel_batches(data, process_fn)
        assert len(results) == 200
        assert results == [x * 2 for x in data]
