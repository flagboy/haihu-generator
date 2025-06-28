"""
モニタリングシステムの統合テスト
"""

import time

from src.monitoring import (
    error_tracker,
    get_structured_logger,
    global_metrics,
    performance_tracker,
    system_monitor,
)
from src.monitoring.config import MonitoringConfig, get_monitoring_config
from src.monitoring.decorators import monitor_batch_processing, monitor_performance


class TestMonitoringIntegration:
    """モニタリングシステムの統合テスト"""

    def test_full_monitoring_workflow(self):
        """完全なモニタリングワークフローのテスト"""
        # 設定を取得
        config = get_monitoring_config()
        assert isinstance(config, MonitoringConfig)

        # ロガーを取得
        logger = get_structured_logger("test_integration")

        # 正常な処理を記録
        logger.info("Starting integration test")

        # パフォーマンスを記録
        with performance_tracker.measure("test_operation"):
            time.sleep(0.01)

        # メトリクスを記録
        global_metrics.record("test_metric", 42)
        global_metrics.increment("test_counter")

        # エラーを追跡
        try:
            raise ValueError("Test error")
        except ValueError as e:
            error_tracker.track_error(e, operation="test_operation")

        # システムステータスを取得
        status = system_monitor.get_current_status()
        assert status.cpu_percent >= 0

        # ヘルスチェック
        health = system_monitor.get_health_check()
        assert health["status"] in ["healthy", "unhealthy"]

    def test_decorated_function_monitoring(self):
        """デコレートされた関数のモニタリングテスト"""

        @monitor_performance(operation_name="test_function", track_items=True)
        def process_items(items):
            """アイテムを処理する関数"""
            return [item * 2 for item in items]

        # 関数を実行
        result = process_items([1, 2, 3, 4, 5])
        assert result == [2, 4, 6, 8, 10]

        # メトリクスが記録されていることを確認
        summary = performance_tracker.metrics.get_summary("test_function_duration_seconds")
        assert summary is not None
        assert summary.count >= 1

    def test_batch_processing_monitoring(self):
        """バッチ処理のモニタリングテスト"""

        @monitor_batch_processing
        def process_batch(batch_size):
            """バッチ処理のシミュレーション"""
            success_count = int(batch_size * 0.9)  # 90%成功
            error_count = batch_size - success_count

            return {
                "success_count": success_count,
                "error_count": error_count,
                "processed": list(range(success_count)),
            }

        # バッチ処理を実行
        result = process_batch(100)
        assert result["success_count"] == 90
        assert result["error_count"] == 10

        # バッチメトリクスが記録されていることを確認
        throughput_summary = performance_tracker.metrics.get_summary("batch_throughput")
        assert throughput_summary is not None

        success_rate_summary = performance_tracker.metrics.get_summary("batch_success_rate")
        assert success_rate_summary is not None

    def test_error_tracking_with_alerts(self):
        """エラー追跡とアラートのテスト"""
        # エラートラッカーを低い閾値で設定
        tracker = error_tracker
        original_threshold = tracker.alert_threshold
        tracker.alert_threshold = 3

        try:
            # 複数のエラーを発生させる
            for i in range(5):
                try:
                    raise RuntimeError(f"Test error {i}")
                except RuntimeError as e:
                    tracker.track_error(e, operation="alert_test")
                time.sleep(0.1)

            # エラーサマリーを確認
            summaries = tracker.get_error_summary(hours=1)
            assert len(summaries) > 0

            runtime_error_summary = next(
                (s for s in summaries if s.error_type == "RuntimeError"), None
            )
            assert runtime_error_summary is not None
            assert runtime_error_summary.count >= 5

        finally:
            # 閾値を元に戻す
            tracker.alert_threshold = original_threshold

    def test_monitoring_data_export(self, tmp_path):
        """モニタリングデータのエクスポートテスト"""
        # メトリクスをエクスポート
        metrics_file = tmp_path / "metrics.json"
        global_metrics.export_to_file(metrics_file)
        assert metrics_file.exists()

        # エラーレポートをエクスポート
        error_report = tmp_path / "error_report.json"
        error_tracker.export_error_report(error_report, hours=24)
        assert error_report.exists()

        # システムステータスをエクスポート
        status_file = tmp_path / "system_status.json"
        system_monitor.export_status_history(status_file, hours=1)
        assert status_file.exists()

    def test_monitored_logger_integration(self):
        """モニタリング機能付きロガーの統合テスト"""
        from src.utils.logger import get_monitored_logger

        logger = get_monitored_logger("test_monitored")

        # 通常のログ
        logger.info("Test info message", key="value")

        # パフォーマンスログ
        logger.log_performance("test_operation", 1.234, items=100)

        # エラーログ
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Error occurred", exc_info=True, operation="test_op")

        # メトリクスが記録されていることを確認
        perf_summary = global_metrics.get_summary("test_operation_duration_seconds")
        assert perf_summary is not None
