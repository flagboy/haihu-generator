"""
メトリクス収集システムのテスト
"""

import time
from datetime import datetime, timedelta

import pytest

from src.monitoring.metrics import MetricsCollector, global_metrics, performance_tracker


class TestMetricsCollector:
    """MetricsCollectorのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        collector = MetricsCollector("test_collector")
        assert collector.name == "test_collector"
        assert isinstance(collector.metrics, dict)

    def test_record_metric(self):
        """メトリクス記録のテスト"""
        collector = MetricsCollector("test")

        # 値を記録
        collector.record("test_metric", 10.5)
        collector.record("test_metric", 20.3)
        collector.record("test_metric", 15.7)

        # サマリーを取得
        summary = collector.get_summary("test_metric")
        assert summary is not None
        assert summary.count == 3
        assert summary.mean == pytest.approx(15.5, 0.1)
        assert summary.min == 10.5
        assert summary.max == 20.3

    def test_increment(self):
        """インクリメントのテスト"""
        collector = MetricsCollector("test")

        # incrementメソッドを使用
        collector.increment("counter")
        collector.increment("counter")
        collector.increment("counter")

        # 各値が記録されていることを確認
        summary = collector.get_summary("counter")
        assert summary.count == 3
        # incrementは累積値を記録するので、最終値は3
        assert summary.max == 3

    def test_gauge(self):
        """ゲージメトリクスのテスト"""
        collector = MetricsCollector("test")

        # ゲージ値を設定
        collector.gauge("memory_usage", 1024.5)
        collector.gauge("memory_usage", 2048.7)

        # 最新値を確認
        summary = collector.get_summary("memory_usage")
        assert summary.count == 2
        assert summary.max == 2048.7

    def test_timing(self):
        """タイミング計測のテスト"""
        collector = MetricsCollector("test")

        # 時間計測をrecordで記録
        collector.record("operation_time", 0.123)
        collector.record("operation_time", 0.456)

        summary = collector.get_summary("operation_time")
        assert summary.count == 2
        assert summary.mean == pytest.approx(0.2895, 0.001)

    def test_get_all_metrics(self):
        """全メトリクス取得のテスト"""
        collector = MetricsCollector("test")

        # 複数のメトリクスを記録
        collector.record("metric1", 10)
        collector.record("metric2", 20)
        collector.increment("counter1")

        all_metrics = collector.get_all_metrics()
        assert "metric1" in all_metrics
        assert "metric2" in all_metrics
        assert "counter1" in all_metrics

    def test_clear_metrics(self):
        """メトリクスクリアのテスト"""
        collector = MetricsCollector("test")

        # メトリクスを記録
        collector.record("test_metric", 10)
        assert collector.get_summary("test_metric") is not None

        # クリア
        collector.clear("test_metric")
        assert collector.get_summary("test_metric") is None

    def test_export_metrics(self):
        """メトリクスエクスポートのテスト"""
        import tempfile
        from pathlib import Path

        collector = MetricsCollector("test")

        # メトリクスを記録
        collector.record("export_test", 10)
        collector.record("export_test", 20)
        collector.increment("export_counter", 5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            # エクスポート
            collector.export_to_file(output_path)

            # ファイルが作成されていることを確認
            assert output_path.exists()

            # 内容を確認
            import json

            with open(output_path) as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "collector" in data
            assert "metrics" in data
            assert "export_test" in data["metrics"]
            assert "export_counter" in data["metrics"]

        finally:
            output_path.unlink()

    def test_window_filtering(self):
        """時間ウィンドウフィルタリングのテスト"""
        collector = MetricsCollector("test")

        # 古いデータと新しいデータを記録
        now = datetime.now()
        old_time = now - timedelta(seconds=10)

        # タイムスタンプを偽装して記録
        from src.monitoring.metrics import Metric

        with collector.lock:
            collector.metrics["old_metric"].append(
                Metric(name="old_metric", value=100, timestamp=old_time, tags={})
            )

        # 新しいデータを記録
        collector.record("old_metric", 200)

        # 5秒のウィンドウでサマリーを取得
        summary = collector.get_summary("old_metric", window_seconds=5)

        # 古いデータは含まれない
        assert summary.count == 1
        assert summary.mean == 200


class TestPerformanceTracker:
    """PerformanceTrackerのテスト"""

    def test_track_operation(self):
        """オペレーション追跡のテスト"""
        tracker = performance_tracker

        # オペレーションを追跡
        tracker.track_operation(
            operation="test_op", duration=1.5, success=True, items_processed=100
        )

        # メトリクスが記録されていることを確認
        duration_summary = tracker.metrics.get_summary("test_op_duration")
        assert duration_summary is not None
        assert duration_summary.count >= 1

        success_summary = tracker.metrics.get_summary("test_op_success_rate")
        assert success_summary is not None

    def test_track_batch_processing(self):
        """バッチ処理追跡のテスト"""
        tracker = performance_tracker

        # バッチ処理を追跡
        tracker.track_batch_processing(
            batch_size=50, processing_time=2.0, success_count=48, error_count=2
        )

        # メトリクスを確認
        throughput_summary = tracker.metrics.get_summary("batch_throughput")
        assert throughput_summary is not None

        success_rate_summary = tracker.metrics.get_summary("batch_success_rate")
        assert success_rate_summary is not None
        assert success_rate_summary.mean == pytest.approx(0.96, 0.01)  # 48/50

    def test_measure_decorator(self):
        """計測デコレーターのテスト"""

        @performance_tracker.measure("test_function")
        def sample_function(x, y):
            time.sleep(0.01)
            return x + y

        result = sample_function(10, 20)
        assert result == 30

        # メトリクスが記録されていることを確認
        summary = performance_tracker.metrics.get_summary("test_function_duration_seconds")
        assert summary is not None
        assert summary.count >= 1
        assert summary.mean >= 0.01


class TestGlobalMetrics:
    """グローバルメトリクスのテスト"""

    def test_global_metrics_singleton(self):
        """グローバルメトリクスがシングルトンであることを確認"""
        from src.monitoring.metrics import global_metrics as gm1
        from src.monitoring.metrics import global_metrics as gm2

        assert gm1 is gm2

    def test_global_metrics_usage(self):
        """グローバルメトリクスの使用テスト"""
        global_metrics.record("global_test", 42)

        summary = global_metrics.get_summary("global_test")
        assert summary is not None
        assert summary.count >= 1
        # 値が記録されていることを確認
        assert summary.min <= 42 <= summary.max
