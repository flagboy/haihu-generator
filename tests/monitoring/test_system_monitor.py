"""
システムモニタリングのテスト
"""

import time
from pathlib import Path

from src.monitoring.system_monitor import SystemMonitor, SystemStatus


class TestSystemMonitor:
    """SystemMonitorのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        monitor = SystemMonitor(interval=5)
        assert monitor.interval == 5
        assert monitor.alert_thresholds["cpu_percent"] == 90.0
        assert monitor.alert_thresholds["memory_percent"] == 90.0

    def test_get_current_status(self):
        """現在のステータス取得テスト"""
        monitor = SystemMonitor()
        status = monitor.get_current_status()

        assert isinstance(status, SystemStatus)
        assert status.cpu_percent >= 0
        assert status.cpu_percent <= 100
        assert status.memory_percent >= 0
        assert status.memory_percent <= 100
        assert status.memory_used_mb > 0
        assert status.memory_available_mb > 0
        assert status.disk_usage_percent >= 0
        assert status.disk_usage_percent <= 100
        assert status.process_count > 0
        assert status.thread_count > 0

    def test_health_check_healthy(self):
        """ヘルスチェック（正常）のテスト"""
        monitor = SystemMonitor()

        # 現在のステータスを取得
        monitor._last_status = monitor.get_current_status()

        health = monitor.get_health_check()
        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        assert "message" in health
        assert "timestamp" in health

    def test_health_check_unhealthy(self):
        """ヘルスチェック（異常）のテスト"""
        from datetime import datetime

        monitor = SystemMonitor(
            alert_thresholds={
                "cpu_percent": 1.0,  # 非常に低い閾値
                "memory_percent": 1.0,
                "disk_usage_percent": 1.0,
                "gpu_memory_percent": 1.0,
            }
        )

        # ダミーのステータスを設定
        monitor._last_status = SystemStatus(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=50.0,
            memory_used_mb=1000,
            memory_available_mb=1000,
            disk_usage_percent=50.0,
            process_count=100,
            thread_count=200,
        )

        health = monitor.get_health_check()
        assert health["status"] == "unhealthy"
        assert "High CPU usage" in health["message"]
        assert "High memory usage" in health["message"]
        assert "High disk usage" in health["message"]

    def test_export_status_history(self):
        """ステータス履歴エクスポートのテスト"""
        import json
        import tempfile

        monitor = SystemMonitor()

        # いくつかのメトリクスを記録
        status = monitor.get_current_status()
        monitor._record_metrics(status)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            # エクスポート
            monitor.export_status_history(output_path, hours=1)

            # ファイルが作成されていることを確認
            assert output_path.exists()

            # 内容を確認
            with open(output_path) as f:
                data = json.load(f)

            assert "export_time" in data
            assert "window_hours" in data
            assert "summaries" in data

        finally:
            output_path.unlink()

    def test_custom_alert_thresholds(self):
        """カスタムアラート閾値のテスト"""
        custom_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 95.0,
            "gpu_memory_percent": 90.0,
        }

        monitor = SystemMonitor(alert_thresholds=custom_thresholds)

        assert monitor.alert_thresholds == custom_thresholds

    def test_monitoring_thread(self):
        """モニタリングスレッドのテスト"""
        monitor = SystemMonitor(interval=1)  # 1秒間隔

        # モニタリング開始
        monitor.start()
        assert monitor._monitoring is True
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        # 少し待機
        time.sleep(2)

        # モニタリング停止
        monitor.stop()
        assert monitor._monitoring is False

        # スレッドが停止するまで待機
        time.sleep(1)
        assert not monitor._thread.is_alive()

    def test_metrics_recording(self):
        """メトリクス記録のテスト"""
        from src.monitoring.metrics import MetricsCollector

        metrics = MetricsCollector("test_system")
        monitor = SystemMonitor(metrics_collector=metrics)

        # ステータスを取得して記録
        status = monitor.get_current_status()
        monitor._record_metrics(status)

        # メトリクスが記録されていることを確認
        cpu_summary = metrics.get_summary("system_cpu_percent")
        assert cpu_summary is not None
        assert cpu_summary.count == 1

        memory_summary = metrics.get_summary("system_memory_percent")
        assert memory_summary is not None
        assert memory_summary.count == 1
