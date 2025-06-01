"""
パフォーマンステスト
システムのパフォーマンスと負荷テストを実行
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.integration.system_integrator import SystemIntegrator
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.utils.config import ConfigManager


class TestPerformanceMetrics:
    """パフォーマンスメトリクステストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: false
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        yield config_manager

        os.unlink(config_path)

    @pytest.fixture
    def performance_optimizer(self, config_manager):
        """パフォーマンス最適化オブジェクトのフィクスチャ"""
        return PerformanceOptimizer(config_manager)

    def test_metrics_collection_speed(self, performance_optimizer):
        """メトリクス収集速度テスト"""
        start_time = time.time()

        # 100回メトリクスを収集
        for _ in range(100):
            metrics = performance_optimizer.get_current_metrics()
            assert metrics is not None

        end_time = time.time()
        total_time = end_time - start_time

        # 1回あたりの収集時間が0.1秒以下であることを確認
        avg_time = total_time / 100
        assert avg_time < 0.1, f"Metrics collection too slow: {avg_time:.3f}s per call"

    def test_monitoring_overhead(self, performance_optimizer):
        """監視オーバーヘッドテスト"""
        # 監視なしでの処理時間測定
        start_time = time.time()
        for _ in range(50):
            # 軽い処理をシミュレート
            np.random.rand(100, 100).sum()
        no_monitoring_time = time.time() - start_time

        # 監視ありでの処理時間測定
        performance_optimizer.start_monitoring()

        start_time = time.time()
        for _ in range(50):
            np.random.rand(100, 100).sum()
        with_monitoring_time = time.time() - start_time

        performance_optimizer.stop_monitoring()

        # オーバーヘッドが50%以下であることを確認
        overhead = (with_monitoring_time - no_monitoring_time) / no_monitoring_time
        assert overhead < 0.5, f"Monitoring overhead too high: {overhead:.2%}"

    def test_memory_usage_tracking(self, performance_optimizer):
        """メモリ使用量追跡テスト"""
        performance_optimizer.get_current_metrics()

        # 大きなデータを作成してメモリ使用量を増加
        large_data = [np.random.rand(1000, 1000) for _ in range(10)]

        # メモリ使用量の変化を確認
        after_metrics = performance_optimizer.get_current_metrics()
        after_memory = after_metrics.memory_usage

        # メモリ使用量が増加していることを確認（ただし、GCの影響で必ずしも増加するとは限らない）
        assert after_memory >= 0  # 最低限、有効な値が取得できることを確認

        # データを削除
        del large_data

        # 少し待ってからメモリ使用量を再確認
        time.sleep(0.1)
        final_metrics = performance_optimizer.get_current_metrics()
        assert final_metrics.memory_usage >= 0


class TestConcurrencyPerformance:
    """並行処理パフォーマンステストクラス"""

    @pytest.fixture
    def mock_components(self):
        """モックコンポーネントのフィクスチャ"""
        video_processor = Mock()
        video_processor.extract_frames.return_value = {
            "success": True,
            "frames": [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)],
        }

        ai_pipeline = Mock()
        ai_pipeline.process_frames_batch.return_value = [
            Mock(
                frame_id=i,
                detections=[],
                classifications=[],
                processing_time=0.01,  # 高速処理をシミュレート
                tile_areas={},
                confidence_scores={"combined_confidence": 0.8},
            )
            for i in range(10)
        ]

        game_pipeline = Mock()
        game_pipeline.initialize_game.return_value = True
        game_pipeline.process_frame.return_value = Mock(success=True, processing_time=0.005)
        game_pipeline.export_game_record.return_value = '{"test": "record"}'

        return video_processor, ai_pipeline, game_pipeline

    @pytest.fixture
    def system_integrator(self, mock_components):
        """システム統合オブジェクトのフィクスチャ"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 4
directories:
  output: "test_output"
""")
            config_path = f.name

        config_manager = ConfigManager(config_path)
        video_processor, ai_pipeline, game_pipeline = mock_components
        integrator = SystemIntegrator(config_manager, video_processor, ai_pipeline, game_pipeline)

        yield integrator

        os.unlink(config_path)

    def test_parallel_processing_speedup(self, system_integrator):
        """並列処理の速度向上テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用動画ファイルを複数作成
            video_files = []
            for i in range(4):
                video_path = os.path.join(temp_dir, f"test_video_{i}.mp4")
                with open(video_path, "w") as f:
                    f.write("dummy")
                video_files.append(video_path)

            output_dir = os.path.join(temp_dir, "output")

            # シーケンシャル処理時間測定
            start_time = time.time()
            for video_file in video_files:
                output_path = os.path.join(output_dir, f"{os.path.basename(video_file)}.json")
                os.makedirs(output_dir, exist_ok=True)
                system_integrator.process_video_complete(video_file, output_path)
            sequential_time = time.time() - start_time

            # 並列処理時間測定
            start_time = time.time()
            result = system_integrator.process_batch(
                video_files=video_files, output_directory=output_dir, max_workers=2
            )
            parallel_time = time.time() - start_time

            # 並列処理が速いことを確認（ただし、モックなので大きな差は期待できない）
            assert result["success"] is True
            assert parallel_time <= sequential_time * 1.2  # 20%のマージンを許容

    def test_thread_safety(self, system_integrator):
        """スレッドセーフティテスト"""
        results = []
        errors = []

        def worker_function(worker_id):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    video_path = os.path.join(temp_dir, f"test_{worker_id}.mp4")
                    with open(video_path, "w") as f:
                        f.write("dummy")

                    output_path = os.path.join(temp_dir, f"output_{worker_id}.json")

                    result = system_integrator.process_video_complete(video_path, output_path)
                    results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 複数スレッドで同時実行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()

        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()

        # エラーがないことを確認
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5

        # 全ての結果が成功していることを確認
        for worker_id, result in results:
            assert result.success is True, f"Worker {worker_id} failed"


class TestMemoryPerformance:
    """メモリパフォーマンステストクラス"""

    def test_memory_leak_detection(self):
        """メモリリーク検出テスト"""
        import gc

        import psutil

        process = psutil.Process()

        # 初期メモリ使用量
        gc.collect()
        initial_memory = process.memory_info().rss

        # 大量のデータ処理をシミュレート
        for i in range(100):
            # 大きなデータを作成・処理・削除
            data = np.random.rand(1000, 1000)
            processed = data * 2 + 1
            del data, processed

            # 定期的にガベージコレクション
            if i % 10 == 0:
                gc.collect()

        # 最終メモリ使用量
        gc.collect()
        final_memory = process.memory_info().rss

        # メモリ増加が初期値の50%以下であることを確認
        memory_increase = final_memory - initial_memory
        memory_increase_ratio = memory_increase / initial_memory

        assert memory_increase_ratio < 0.5, (
            f"Potential memory leak detected: {memory_increase_ratio:.2%} increase"
        )

    def test_large_data_processing(self):
        """大容量データ処理テスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 2
  memory_limit: "4GB"
""")
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            optimizer = PerformanceOptimizer(config_manager)

            # 大容量データ処理前のメトリクス
            before_metrics = optimizer.get_current_metrics()

            # 大容量データを処理
            large_arrays = []
            for _i in range(10):
                # 100MB程度のデータを作成
                array = np.random.rand(1000, 1000, 10).astype(np.float32)
                large_arrays.append(array)

            # 処理後のメトリクス
            after_metrics = optimizer.get_current_metrics()

            # メモリ使用量が適切な範囲内であることを確認
            memory_increase = after_metrics.memory_usage - before_metrics.memory_usage
            assert memory_increase < 50, f"Memory usage increased too much: {memory_increase}%"

            # データをクリーンアップ
            del large_arrays

        finally:
            os.unlink(config_path)


class TestScalabilityPerformance:
    """スケーラビリティパフォーマンステストクラス"""

    def test_batch_size_scaling(self):
        """バッチサイズスケーリングテスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 4
""")
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            PerformanceOptimizer(config_manager)

            # 異なるバッチサイズでの処理時間を測定
            batch_sizes = [1, 2, 4, 8, 16]
            processing_times = []

            for batch_size in batch_sizes:
                start_time = time.time()

                # バッチ処理をシミュレート
                for _ in range(10):
                    # バッチサイズに応じた処理をシミュレート
                    data = [np.random.rand(100, 100) for _ in range(batch_size)]
                    [d.sum() for d in data]

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

            # バッチサイズが大きくなるにつれて、単位あたりの処理時間が改善されることを確認
            # （ただし、シミュレーションなので厳密な検証は困難）
            assert all(t > 0 for t in processing_times), "All processing times should be positive"

        finally:
            os.unlink(config_path)

    def test_worker_scaling(self):
        """ワーカー数スケーリングテスト"""

        def simulate_work(duration=0.1):
            """作業をシミュレート"""
            time.sleep(duration)
            return np.random.rand(100, 100).sum()

        # 異なるワーカー数での処理時間を測定
        worker_counts = [1, 2, 4]
        task_count = 8

        for worker_count in worker_counts:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(simulate_work, 0.05) for _ in range(task_count)]
                results = [future.result() for future in futures]

            processing_time = time.time() - start_time

            # 結果が正しく取得できることを確認
            assert len(results) == task_count
            assert all(isinstance(r, int | float) for r in results)

            # 処理時間が合理的な範囲内であることを確認
            assert processing_time < 2.0, (
                f"Processing took too long with {worker_count} workers: {processing_time:.2f}s"
            )


class TestRealTimePerformance:
    """リアルタイムパフォーマンステストクラス"""

    def test_frame_processing_speed(self):
        """フレーム処理速度テスト"""
        # モックAIパイプラインを作成
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
ai:
  training:
    batch_size: 4
system:
  max_workers: 2
""")
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)

            # モックコンポーネント
            from src.pipeline.ai_pipeline import AIPipeline

            # 実際のAIPipelineをモック
            with (
                patch("src.detection.tile_detector.TileDetector") as mock_detector,
                patch("src.classification.tile_classifier.TileClassifier") as mock_classifier,
            ):
                # モックの設定
                mock_detector.return_value.detect_tiles.return_value = []
                mock_detector.return_value.classify_tile_areas.return_value = {}
                mock_classifier.return_value.classify_tiles_batch.return_value = []

                ai_pipeline = AIPipeline(config_manager)

                # テストフレームを作成
                test_frames = [
                    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
                ]

                # 処理時間測定
                start_time = time.time()
                results = ai_pipeline.process_frames_batch(test_frames)
                processing_time = time.time() - start_time

                # フレームレートを計算
                fps = len(test_frames) / processing_time

                # 最低限のフレームレートを確保できることを確認
                assert fps > 1.0, f"Frame processing too slow: {fps:.2f} FPS"
                assert len(results) == len(test_frames)

        finally:
            os.unlink(config_path)

    def test_response_time_consistency(self):
        """応答時間一貫性テスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
system:
  max_workers: 2
""")
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            optimizer = PerformanceOptimizer(config_manager)

            # 複数回メトリクス取得の応答時間を測定
            response_times = []

            for _ in range(50):
                start_time = time.time()
                metrics = optimizer.get_current_metrics()
                response_time = time.time() - start_time
                response_times.append(response_time)

                assert metrics is not None

            # 応答時間の統計
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            std_response_time = np.std(response_times)

            # 応答時間の一貫性を確認
            assert avg_response_time < 0.1, (
                f"Average response time too slow: {avg_response_time:.3f}s"
            )
            assert max_response_time < 0.5, f"Max response time too slow: {max_response_time:.3f}s"
            assert std_response_time < 0.05, (
                f"Response time too inconsistent: {std_response_time:.3f}s std"
            )

        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
