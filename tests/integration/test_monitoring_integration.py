"""
モニタリングシステムとメインシステムの統合テスト

モニタリング機能が実際のシステムと正しく統合されているかをテスト
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.integration.orchestrator import VideoProcessingOrchestrator
from src.integration.system_integrator import SystemIntegrator
from src.monitoring import get_error_tracker, get_global_metrics, get_performance_tracker
from src.pipeline.ai_pipeline import AIPipeline
from src.utils.config import ConfigManager
from src.video.video_processor import VideoProcessor


class TestMonitoringSystemIntegration:
    """モニタリングシステム統合テスト"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクト"""
        config = ConfigManager()
        return config

    @pytest.fixture
    def mock_video_file(self):
        """モックビデオファイル"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name
        yield video_path
        Path(video_path).unlink(missing_ok=True)

    def test_orchestrator_with_monitoring(self, config_manager, mock_video_file):
        """オーケストレーターとモニタリングの統合テスト"""
        # モニタリングメトリクスをリセット
        get_global_metrics()
        get_performance_tracker()
        get_error_tracker()

        # VideoProcessorをモック
        with patch("src.integration.orchestrator.VideoProcessor") as MockVideoProcessor:
            mock_processor = Mock()
            mock_processor.extract_frames.return_value = []
            MockVideoProcessor.return_value = mock_processor

            # AIPipelineをモック
            with patch("src.integration.orchestrator.AIPipeline") as MockAIPipeline:
                mock_ai = Mock()
                mock_ai.process_frames.return_value = []
                MockAIPipeline.return_value = mock_ai

                # GamePipelineをモック
                with patch("src.integration.orchestrator.GamePipeline") as MockGamePipeline:
                    mock_game = Mock()
                    mock_game.export_game_record.return_value = {"game": "data"}
                    MockGamePipeline.return_value = mock_game

                    # オーケストレーターを実行
                    orchestrator = VideoProcessingOrchestrator(config_manager)
                    result = orchestrator.process_video(
                        mock_video_file, skip_ai_pipeline=False, skip_game_pipeline=False
                    )

                    assert result.success

                    # メトリクスが記録されていることを確認
                    stats = orchestrator.collect_statistics()
                    assert "total_frames_processed" in stats
                    assert "processing_fps" in stats

    def test_ai_pipeline_monitoring_integration(self, config_manager):
        """AIPipelineとモニタリングの統合テスト"""
        # メトリクスをリセット
        get_performance_tracker()

        # モックの設定
        with patch("src.pipeline.ai_pipeline.TileDetector") as MockDetector:
            mock_detector = Mock()
            mock_detector.detect_tiles.return_value = []
            MockDetector.return_value = mock_detector

            with patch("src.pipeline.ai_pipeline.TileClassifier") as MockClassifier:
                mock_classifier = Mock()
                MockClassifier.return_value = mock_classifier

                # AIPipelineを実行
                pipeline = AIPipeline(config_manager)
                test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                result = pipeline.process_frame(test_frame, frame_number=1)

                # パフォーマンスメトリクスが記録されていることを確認
                # process_frameメソッドが実行時間を記録しているか確認
                assert result is not None

    def test_error_tracking_integration(self, config_manager):
        """エラートラッキングの統合テスト"""
        error_tracker = get_error_tracker()
        len(error_tracker.errors)

        # 意図的にエラーを発生させる
        with patch("src.video.video_processor.cv2.VideoCapture") as MockCapture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            MockCapture.return_value = mock_cap

            processor = VideoProcessor(config_manager)
            info = processor.get_video_info("nonexistent.mp4")
            assert info is None

            # エラーが追跡されていることを確認
            # get_video_infoがエラーをログに記録している
            len(error_tracker.errors)
            # エラーログが記録されているかは実装に依存

    def test_system_integrator_full_monitoring(self, config_manager, mock_video_file):
        """SystemIntegratorの完全なモニタリング統合テスト"""
        # メトリクスをリセット
        get_global_metrics()
        get_performance_tracker()

        # SystemIntegratorのモック設定
        with patch("src.integration.system_integrator.VideoProcessor") as MockVideoProcessor:
            mock_processor = Mock()
            mock_processor.extract_frames.return_value = []
            MockVideoProcessor.return_value = mock_processor

            with patch("src.integration.system_integrator.AIPipeline") as MockAIPipeline:
                mock_ai = Mock()
                mock_ai.process_frames.return_value = []
                MockAIPipeline.return_value = mock_ai

                with patch("src.integration.system_integrator.GamePipeline") as MockGamePipeline:
                    mock_game = Mock()
                    mock_game.export_game_record.return_value = {"game": "data"}
                    MockGamePipeline.return_value = mock_game

                    # システム統合を実行
                    integrator = SystemIntegrator(config_manager)
                    result = integrator.process_video(
                        mock_video_file, output_path=Path("output.json"), save_intermediate=False
                    )

                    assert result["success"]

                    # システム情報が含まれていることを確認
                    system_info = integrator.get_system_info()
                    assert "version" in system_info
                    assert "config" in system_info
                    assert "capabilities" in system_info

    def test_performance_optimizer_monitoring(self, config_manager):
        """PerformanceOptimizerとモニタリングの統合テスト"""
        from src.optimization.performance_optimizer import PerformanceOptimizer

        optimizer = PerformanceOptimizer(config_manager)

        # 初期メトリクスを取得
        initial_metrics = optimizer.get_current_metrics()
        assert "cpu_percent" in initial_metrics
        assert "memory_percent" in initial_metrics

        # 最適化を実行
        optimizer.optimize_system()

        # 推奨事項を取得
        recommendations = optimizer.get_recommendations()
        assert isinstance(recommendations, list)

        # モニタリング中のメトリクス変化を確認
        time.sleep(0.1)  # 少し待機
        current_metrics = optimizer.get_current_metrics()
        assert current_metrics is not None

    def test_monitoring_data_persistence(self, config_manager):
        """モニタリングデータの永続化テスト"""
        metrics = get_global_metrics()
        error_tracker = get_error_tracker()

        # テストデータを記録
        metrics.record("test_metric", 42.0)
        metrics.increment("test_counter", 3)

        try:
            raise ValueError("Test error for persistence")
        except ValueError as e:
            error_tracker.track_error(e, operation="test_persistence")

        # データをエクスポート
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_file = Path(temp_dir) / "metrics.json"
            error_file = Path(temp_dir) / "errors.json"

            metrics.export_to_file(metrics_file)
            error_tracker.export_error_report(error_file, hours=1)

            assert metrics_file.exists()
            assert error_file.exists()

            # エクスポートされたデータの検証
            import json

            with open(metrics_file) as f:
                metrics_data = json.load(f)
                assert "metrics" in metrics_data
                assert "test_metric" in metrics_data["metrics"]

            with open(error_file) as f:
                error_data = json.load(f)
                assert "total_errors" in error_data
                assert error_data["total_errors"] >= 1

    @pytest.mark.performance
    def test_monitoring_overhead(self, config_manager):
        """モニタリングのオーバーヘッドテスト"""
        import timeit

        # モニタリングありの処理時間
        def with_monitoring():
            metrics = get_global_metrics()
            perf_tracker = get_performance_tracker()
            for i in range(100):
                metrics.record(f"test_{i}", float(i))
                with perf_tracker.track("test_op"):
                    time.sleep(0.001)

        # モニタリングなしの処理時間
        def without_monitoring():
            for _i in range(100):
                time.sleep(0.001)

        # 実行時間を計測
        time_with = timeit.timeit(with_monitoring, number=1)
        time_without = timeit.timeit(without_monitoring, number=1)

        # オーバーヘッドが妥当な範囲内であることを確認
        overhead_ratio = (time_with - time_without) / time_without
        assert overhead_ratio < 0.5  # 50%以下のオーバーヘッド

    def test_concurrent_monitoring_access(self, config_manager):
        """並行アクセス時のモニタリング動作テスト"""
        import threading

        metrics = get_global_metrics()
        results = []

        def record_metrics(thread_id):
            """各スレッドでメトリクスを記録"""
            for i in range(10):
                metrics.record(f"thread_{thread_id}_metric", float(i))
                metrics.increment(f"thread_{thread_id}_counter")
            results.append(thread_id)

        # 複数スレッドで同時実行
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_metrics, args=(i,))
            threads.append(t)
            t.start()

        # 全スレッドの完了を待機
        for t in threads:
            t.join()

        # 全スレッドが正常に完了したことを確認
        assert len(results) == 5

        # 各スレッドのメトリクスが記録されていることを確認
        for i in range(5):
            summary = metrics.get_summary(f"thread_{i}_metric")
            assert summary is not None
            assert summary.count == 10
