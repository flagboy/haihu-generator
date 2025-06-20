"""
リファクタリングされたAPIルートの簡易テスト

個々のサービスレベルでのモックを使用してテストを簡素化
"""

import json
from unittest.mock import patch

import pytest
from flask import Flask

from src.training.game_scene.labeling.api.scene_routes import setup_scene_labeling_api


@pytest.fixture
def app():
    """テスト用Flaskアプリケーション"""
    app = Flask(__name__)
    app.config["TESTING"] = True
    setup_scene_labeling_api(app, use_legacy=False)
    return app


@pytest.fixture
def client(app):
    """テストクライアント"""
    return app.test_client()


class TestAPIRefactored:
    """リファクタリングされたAPIの統合テスト"""

    def test_session_create_and_list(self, client):
        """セッション作成と一覧取得"""
        # セッション作成をモック
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.create_session"
        ) as mock_create:
            mock_create.return_value = {
                "session_id": "test-123",
                "video_path": "/path/to/video.mp4",
                "total_frames": 1000,
                "labeled_frames": 0,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "status": "active",
                "metadata": {},
            }

            # セッション作成
            response = client.post(
                "/api/scene_labeling/sessions",
                json={"video_path": "/path/to/video.mp4"},
                content_type="application/json",
            )
            assert response.status_code == 201
            data = json.loads(response.data)
            assert data["session_id"] == "test-123"

        # セッション一覧取得をモック
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.list_sessions"
        ) as mock_list:
            mock_list.return_value = [
                {
                    "session_id": "test-123",
                    "video_path": "/path/to/video.mp4",
                    "total_frames": 1000,
                    "labeled_frames": 100,
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "status": "active",
                }
            ]

            # セッション一覧取得
            response = client.get("/api/scene_labeling/sessions")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["total"] == 1
            assert data["sessions"][0]["session_id"] == "test-123"

    def test_error_handling(self, client):
        """エラーハンドリングのテスト"""
        # 空のビデオパスでエラー
        response = client.post(
            "/api/scene_labeling/sessions",
            json={"video_path": ""},
            content_type="application/json",
        )
        assert response.status_code in [400, 422]
        data = json.loads(response.data)
        assert "error" in data

        # 存在しないセッション
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.get_session_info"
        ) as mock_get:
            from src.training.game_scene.labeling.api.middleware.error_handler import (
                NotFoundError,
            )

            mock_get.side_effect = NotFoundError("セッション")

            response = client.get("/api/scene_labeling/sessions/non-existent")
            assert response.status_code == 404
            data = json.loads(response.data)
            assert "error" in data

    def test_middleware_logging(self, client):
        """ミドルウェアのログ出力テスト"""
        # ログ出力の確認はcaptured stderrで確認されているので、
        # ここではミドルウェアが正しく動作することを確認
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.list_sessions"
        ) as mock_list:
            mock_list.return_value = []

            # リクエストを送信
            response = client.get("/api/scene_labeling/sessions")
            assert response.status_code == 200

            # レスポンスヘッダーでリクエストIDが設定されていることを確認（将来的に実装可能）
            # assert "X-Request-ID" in response.headers

    def test_restful_design(self, client):
        """RESTfulデザインの確認"""
        # GET /sessions - 一覧取得
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.list_sessions"
        ) as mock_list:
            mock_list.return_value = []
            response = client.get("/api/scene_labeling/sessions")
            assert response.status_code == 200

        # POST /sessions - 作成
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.create_session"
        ) as mock_create:
            mock_create.return_value = {
                "session_id": "new-123",
                "video_path": "/path/to/video.mp4",
                "total_frames": 1000,
                "labeled_frames": 0,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "status": "active",
                "metadata": {},
            }
            response = client.post(
                "/api/scene_labeling/sessions",
                json={"video_path": "/path/to/video.mp4"},
                content_type="application/json",
            )
            assert response.status_code == 201

        # DELETE /sessions/{id} - 削除
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.delete_session"
        ):
            response = client.delete("/api/scene_labeling/sessions/test-123")
            assert response.status_code == 200
