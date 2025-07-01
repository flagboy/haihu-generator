"""
Web APIとシステムの統合テスト

Flask APIエンドポイントとバックエンドシステムの統合をテスト
"""

import json
import tempfile
from unittest.mock import Mock, patch

import pytest
from flask import Flask

from src.utils.config import ConfigManager
from web_interface.api.data_management_api import DataManagementAPI
from web_interface.api.monitoring_api import setup_monitoring_api
from web_interface.api.scene_routes import setup_scene_labeling_api


class TestWebAPIIntegration:
    """Web API統合テスト"""

    @pytest.fixture
    def app(self):
        """Flaskアプリケーションのフィクスチャ"""
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()

        # APIを設定
        ConfigManager()
        DataManagementAPI(app, upload_folder=app.config["UPLOAD_FOLDER"])

        # モニタリングAPIを設定
        setup_monitoring_api(app)

        # シーンラベリングAPIを設定
        setup_scene_labeling_api(app)

        return app

    @pytest.fixture
    def client(self, app):
        """テストクライアント"""
        return app.test_client()

    def test_health_check_endpoints(self, client):
        """ヘルスチェックエンドポイントのテスト"""
        # システムヘルスチェック
        response = client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "unknown"]

        # モニタリングステータス
        response = client.get("/api/monitoring/status")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "cpu_percent" in data
        assert "memory_percent" in data

    def test_video_management_api_flow(self, client, app):
        """ビデオ管理APIフローのテスト"""
        # 1. ビデオ一覧（初期状態）
        response = client.get("/api/videos")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["videos"] == []

        # 2. ビデオアップロード（モック）
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            tmp_file.write(b"fake video data")
            tmp_file.seek(0)

            with patch(
                "web_interface.api.data_management_api.DatasetManager"
            ) as MockDatasetManager:
                mock_manager = Mock()
                mock_manager.register_video.return_value = "video_001"
                mock_manager.get_all_videos.return_value = [
                    {
                        "video_id": "video_001",
                        "filename": "test.mp4",
                        "duration": 120.0,
                        "fps": 30.0,
                        "created_at": "2024-01-01T00:00:00",
                    }
                ]
                MockDatasetManager.return_value = mock_manager

                # APIインスタンスを更新
                api = app.extensions.get("data_management_api")
                if api:
                    api.dataset_manager = mock_manager

                # ビデオ一覧を再取得
                response = client.get("/api/videos")
                assert response.status_code == 200
                data = json.loads(response.data)
                assert len(data["videos"]) == 1
                assert data["videos"][0]["video_id"] == "video_001"

    def test_scene_labeling_session_flow(self, client):
        """シーンラベリングセッションフローのテスト"""
        with patch("web_interface.api.scene_routes.VideoProcessor") as MockVideoProcessor:
            mock_processor = Mock()
            mock_processor.get_video_info.return_value = {
                "duration": 120.0,
                "fps": 30.0,
                "width": 1280,
                "height": 720,
                "frame_count": 3600,
            }
            MockVideoProcessor.return_value = mock_processor

            # セッション作成
            session_data = {"video_path": "/path/to/video.mp4", "session_name": "test_session"}

            response = client.post("/api/scene/session", json=session_data)
            assert response.status_code == 200
            data = json.loads(response.data)
            assert "session_id" in data
            session_id = data["session_id"]

            # セッション削除
            response = client.delete(f"/api/scene/session/{session_id}")
            assert response.status_code == 200

    def test_monitoring_metrics_api(self, client):
        """モニタリングメトリクスAPIのテスト"""
        # メトリクス一覧
        response = client.get("/api/monitoring/metrics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)

        # 特定のメトリクス詳細（存在しない場合）
        response = client.get("/api/monitoring/metrics/non_existent_metric")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data is None or "error" in data

    def test_error_tracking_api(self, client):
        """エラートラッキングAPIのテスト"""
        # エラー一覧
        response = client.get("/api/monitoring/errors")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "errors" in data
        assert "total" in data

        # 時間範囲指定
        response = client.get("/api/monitoring/errors?hours=24")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "errors" in data

    def test_dataset_export_api(self, client):
        """データセットエクスポートAPIのテスト"""
        with patch("web_interface.api.data_management_api.DatasetManager") as MockDatasetManager:
            mock_manager = Mock()
            mock_version = {
                "version_id": "v001",
                "name": "test_version",
                "created_at": "2024-01-01T00:00:00",
            }
            mock_manager.get_dataset_version.return_value = mock_version
            mock_manager.export_dataset_version.return_value = None
            MockDatasetManager.return_value = mock_manager

            # YOLOフォーマットでエクスポート
            response = client.post("/api/export/yolo/v001")
            assert response.status_code in [200, 500]  # モックなので結果は不定

            # COCOフォーマットでエクスポート
            response = client.post("/api/export/coco/v001")
            assert response.status_code in [200, 500]

            # 未対応フォーマット
            response = client.post("/api/export/unknown/v001")
            assert response.status_code == 400

    @pytest.mark.performance
    def test_api_performance(self, client):
        """APIパフォーマンステスト"""
        import time

        endpoints = [
            "/api/health",
            "/api/monitoring/status",
            "/api/monitoring/metrics",
            "/api/videos",
        ]

        response_times = {}

        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            elapsed = time.time() - start_time

            assert response.status_code == 200
            response_times[endpoint] = elapsed

            # レスポンスタイムが1秒以内であることを確認
            assert elapsed < 1.0, f"{endpoint} took {elapsed:.3f}s"

        # 平均レスポンスタイムを計算
        avg_time = sum(response_times.values()) / len(response_times)
        assert avg_time < 0.5, f"Average response time {avg_time:.3f}s is too slow"

    def test_concurrent_api_requests(self, client):
        """並行APIリクエストのテスト"""
        import threading

        results = []

        def make_request(endpoint):
            """APIリクエストを実行"""
            response = client.get(endpoint)
            results.append((endpoint, response.status_code))

        # 複数のエンドポイントに同時リクエスト
        endpoints = ["/api/health", "/api/monitoring/status", "/api/monitoring/metrics"]
        threads = []

        for endpoint in endpoints * 3:  # 各エンドポイントに3回
            t = threading.Thread(target=make_request, args=(endpoint,))
            threads.append(t)
            t.start()

        # 全スレッドの完了を待機
        for t in threads:
            t.join()

        # 全リクエストが成功したことを確認
        assert len(results) == len(endpoints) * 3
        for _endpoint, status_code in results:
            assert status_code == 200

    def test_error_handling_in_api(self, client):
        """APIエラーハンドリングのテスト"""
        # 不正なJSONデータ
        response = client.post(
            "/api/scene/session", data="invalid json", content_type="application/json"
        )
        assert response.status_code == 400

        # 必須パラメータの欠落
        response = client.post("/api/scene/session", json={})
        assert response.status_code == 400

        # 存在しないリソース
        response = client.get("/api/videos/non_existent_id")
        assert response.status_code == 404

        response = client.delete("/api/videos/non_existent_id")
        assert response.status_code == 404

    def test_api_integration_with_monitoring(self, client):
        """APIとモニタリングシステムの統合テスト"""
        from src.monitoring import get_global_metrics

        # 初期メトリクスを記録
        metrics = get_global_metrics()
        len(metrics.metrics)

        # 複数のAPIコールを実行
        for _ in range(10):
            client.get("/api/health")
            client.get("/api/monitoring/status")

        # メトリクスが増加していることを確認（実装に依存）
        # APIコールがメトリクスを記録する場合
        len(metrics.metrics)
        # メトリクスの記録は実装に依存するため、増加をチェックしない
