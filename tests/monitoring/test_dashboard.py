"""
モニタリングダッシュボードのテスト
"""

import json
from unittest.mock import Mock, patch

import pytest
from flask import Flask

from src.monitoring.dashboard import dashboard_bp, get_monitoring_data


class TestDashboard:
    """ダッシュボードのテスト"""

    @pytest.fixture
    def app(self):
        """Flaskアプリケーションのフィクスチャ"""
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(dashboard_bp)
        return app

    @pytest.fixture
    def client(self, app):
        """テストクライアントのフィクスチャ"""
        return app.test_client()

    def test_dashboard_route(self, client):
        """ダッシュボードルートのテスト"""
        response = client.get("/monitoring/")
        assert response.status_code == 200
        assert "システムモニタリングダッシュボード".encode() in response.data
        assert b'<canvas id="resource-chart">' in response.data
        assert b'<canvas id="performance-chart">' in response.data

    def test_api_status_route(self, client):
        """APIステータスルートのテスト"""
        response = client.get("/monitoring/api/status")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "timestamp" in data
        assert "system" in data
        assert "performance" in data
        assert "errors" in data
        assert "health" in data
        assert "alerts" in data

    def test_api_health_route(self, client):
        """APIヘルスチェックルートのテスト"""
        response = client.get("/monitoring/api/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "unknown"]
        assert "message" in data
        assert "timestamp" in data

    def test_get_monitoring_data(self):
        """モニタリングデータ取得のテスト"""
        data = get_monitoring_data()

        assert isinstance(data, dict)
        assert "timestamp" in data
        assert "system" in data
        assert "performance" in data
        assert "errors" in data
        assert "health" in data
        assert "alerts" in data

        # システムデータの検証
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "disk_usage_percent" in system
        assert isinstance(system["cpu_percent"], int | float)
        assert 0 <= system["cpu_percent"] <= 100

        # パフォーマンスデータの検証
        performance = data["performance"]
        assert "fps" in performance
        assert "success_rate" in performance
        assert "avg_time" in performance
        assert "throughput" in performance

        # エラーデータの検証
        errors = data["errors"]
        assert "total" in errors
        assert "rate" in errors
        assert "critical" in errors
        assert "recent" in errors
        assert isinstance(errors["recent"], list)

        # ヘルスチェックデータの検証
        health = data["health"]
        assert "status" in health
        assert "message" in health

        # アラートデータの検証
        assert isinstance(data["alerts"], list)

    def test_monitoring_data_with_unhealthy_system(self):
        """異常状態でのモニタリングデータのテスト"""
        from datetime import datetime

        from src.monitoring.system_monitor import SystemStatus

        # システムモニターをモック
        with patch("src.monitoring.dashboard.get_system_monitor") as mock_get_monitor:
            # 高いCPU使用率を返すように設定
            mock_status = SystemStatus(
                timestamp=datetime.now(),
                cpu_percent=95.0,
                memory_percent=85.0,
                memory_used_mb=8000,
                memory_available_mb=2000,
                disk_usage_percent=90.0,
                process_count=100,
                thread_count=200,
            )
            mock_monitor = Mock()
            mock_get_monitor.return_value = mock_monitor
            mock_monitor.get_current_status.return_value = mock_status
            mock_monitor.get_health_check.return_value = {
                "status": "unhealthy",
                "message": "High CPU usage: 95.0%; High disk usage: 90.0%",
                "timestamp": datetime.now().isoformat(),
            }

            data = get_monitoring_data()

            # アラートが生成されていることを確認
            assert len(data["alerts"]) > 0
            assert any(alert["type"] == "system" for alert in data["alerts"])

    def test_api_metric_detail(self, client):
        """メトリクス詳細APIのテスト"""
        # 存在しないメトリクス
        response = client.get("/monitoring/api/metrics/non_existent")
        assert response.status_code == 404

        # テスト用メトリクスを追加
        from src.monitoring.metrics import get_global_metrics

        metrics = get_global_metrics()
        metrics.record("test_metric", 10)
        metrics.record("test_metric", 20)
        metrics.record("test_metric", 30)

        # メトリクス詳細を取得
        response = client.get("/monitoring/api/metrics/test_metric")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["metric"] == "test_metric"
        assert "summary" in data
        summary = data["summary"]
        assert summary["count"] >= 3
        assert "mean" in summary
        assert "min" in summary
        assert "max" in summary

    def test_api_errors_with_parameters(self, client):
        """エラーAPI（パラメータ付き）のテスト"""
        # デフォルト（24時間）
        response = client.get("/monitoring/api/errors")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["hours"] == 24

        # カスタム時間窓
        response = client.get("/monitoring/api/errors?hours=48")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["hours"] == 48
        assert "errors" in data
        assert isinstance(data["errors"], list)
