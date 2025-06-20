"""
リファクタリングされたAPIルートのテスト
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from src.training.game_scene.labeling.api.middleware.error_handler import (
    NotFoundError,
)
from src.training.game_scene.labeling.api.routes.auto_label_routes import init_auto_label_service
from src.training.game_scene.labeling.api.scene_routes import setup_scene_labeling_api


@pytest.fixture
def app():
    """テスト用Flaskアプリケーション"""
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """テストクライアント"""
    setup_scene_labeling_api(app, use_legacy=False)
    return app.test_client()


@pytest.fixture
def mock_session():
    """モックセッション"""
    session = MagicMock()
    session.session_id = "test-session-123"
    session.video_path = "/path/to/video.mp4"
    session.total_frames = 1000
    session.get_labeled_frames_count.return_value = 100
    session.created_at = "2024-01-01T00:00:00"
    session.updated_at = "2024-01-01T00:00:00"
    session.status = "active"
    session.is_closed.return_value = False  # アクティブなセッション
    return session


@pytest.fixture
def mock_classifier():
    """モック分類器"""
    classifier = MagicMock()
    classifier.predict.return_value = {
        "label": "game",
        "confidence": 0.95,
        "probabilities": {"game": 0.95, "menu": 0.03, "loading": 0.02},
    }
    return classifier


class TestSessionRoutes:
    """セッションルートのテスト"""

    def test_create_session(self, client):
        """セッション作成のテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.create_session"
        ) as mock_create:
            # サービスメソッドの返り値を設定
            mock_create.return_value = {
                "session_id": "new-session-123",
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
            data = json.loads(response.data)
            assert data["session_id"] == "new-session-123"
            assert data["video_path"] == "/path/to/video.mp4"
            assert data["total_frames"] == 1000
            assert data["labeled_frames"] == 0

    def test_create_session_invalid_path(self, client):
        """無効なパスでセッション作成"""
        response = client.post(
            "/api/scene_labeling/sessions",
            json={"video_path": ""},
            content_type="application/json",
        )

        assert response.status_code in [400, 422]  # バリデーションエラー
        data = json.loads(response.data)
        assert "error" in data

    def test_list_sessions(self, client):
        """セッション一覧取得のテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.list_sessions"
        ) as mock_list:
            mock_list.return_value = [
                {
                    "session_id": "session-1",
                    "video_path": "/path/to/video1.mp4",
                    "total_frames": 1000,
                    "labeled_frames": 100,
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "status": "active",
                }
            ]

            response = client.get("/api/scene_labeling/sessions")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["total"] == 1
            assert len(data["sessions"]) == 1

    def test_get_session(self, client, mock_session):
        """セッション情報取得のテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.get_session_info"
        ) as mock_get_info:
            mock_get_info.return_value = {
                "session_id": "test-session-123",
                "video_path": "/path/to/video.mp4",
                "total_frames": 1000,
                "labeled_frames": 100,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "status": "active",
                "metadata": {},
            }

            response = client.get("/api/scene_labeling/sessions/test-session-123")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["session_id"] == "test-session-123"

    def test_delete_session(self, client):
        """セッション削除のテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.delete_session"
        ) as mock_delete:
            response = client.delete("/api/scene_labeling/sessions/test-session-123")

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            mock_delete.assert_called_once_with("test-session-123")


class TestFrameRoutes:
    """フレームルートのテスト"""

    def test_get_frame(self, client, mock_session):
        """フレーム取得のテスト"""
        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.frame_service.FrameService.get_frame"
                ) as mock_get_frame:
                    mock_get_frame.return_value = {
                        "frame_number": 100,
                        "timestamp": 3.33,
                        "image": "base64_encoded_image",
                        "label": "game",
                        "confidence": 0.95,
                        "is_labeled": True,
                    }

                    response = client.get(
                        "/api/scene_labeling/sessions/test-session-123/frames/100"
                    )

                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data["frame_number"] == 100
                    assert data["label"] == "game"

    def test_get_next_unlabeled_frame(self, client, mock_session):
        """次の未ラベルフレーム取得のテスト"""
        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.frame_service.FrameService.get_next_unlabeled_frame"
                ) as mock_get_next:
                    mock_get_next.return_value = {
                        "frame_number": 200,
                        "timestamp": 6.67,
                        "image": "base64_encoded_image",
                        "label": None,
                        "confidence": None,
                        "is_labeled": False,
                    }

                    response = client.get(
                        "/api/scene_labeling/sessions/test-session-123/frames/next_unlabeled"
                    )

                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data["frame_number"] == 200
                    assert data["is_labeled"] is False


class TestLabelingRoutes:
    """ラベリングルートのテスト"""

    def test_label_frame(self, client, mock_session):
        """フレームラベル付けのテスト"""
        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.labeling_service.LabelingService.label_frame"
                ) as mock_label:
                    mock_label.return_value = {
                        "success": True,
                        "frame_number": 100,
                        "label": "game",
                        "confidence": 0.95,
                    }

                    response = client.post(
                        "/api/scene_labeling/sessions/test-session-123/label",
                        json={"frame_number": 100, "label": "game", "confidence": 0.95},
                        content_type="application/json",
                    )

                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data["success"] is True

    def test_batch_label_frames(self, client, mock_session):
        """バッチラベル付けのテスト"""
        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.labeling_service.LabelingService.batch_label_frames"
                ) as mock_batch:
                    mock_batch.return_value = {
                        "success": True,
                        "results": [
                            {"success": True, "frame_number": 100, "label": "game"},
                            {"success": True, "frame_number": 101, "label": "menu"},
                        ],
                        "summary": {"total": 2, "success": 2, "error": 0},
                    }

                response = client.post(
                    "/api/scene_labeling/sessions/test-session-123/batch_label",
                    json={
                        "labels": [
                            {"frame_number": 100, "label": "game"},
                            {"frame_number": 101, "label": "menu"},
                        ]
                    },
                    content_type="application/json",
                )

                assert response.status_code == 200
                data = json.loads(response.data)
                assert data["success"] is True


class TestAutoLabelRoutes:
    """自動ラベリングルートのテスト"""

    def test_auto_label_frames(self, client, mock_session, mock_classifier):
        """自動ラベリングのテスト"""
        # 分類器を初期化
        init_auto_label_service(mock_classifier)

        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.auto_label_service.AutoLabelService.auto_label_frames"
                ) as mock_auto_label:
                    mock_auto_label.return_value = {
                        "success": True,
                        "summary": {
                            "processed": 10,
                            "labeled": 8,
                            "skipped": 1,
                            "error": 1,
                            "success_rate": 0.8,
                        },
                        "results": [],
                    }

                    response = client.post(
                        "/api/scene_labeling/sessions/test-session-123/auto_label",
                        json={"confidence_threshold": 0.8, "max_frames": 10},
                        content_type="application/json",
                    )

                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data["success"] is True

    def test_predict_frame(self, client, mock_session, mock_classifier):
        """フレーム予測のテスト"""
        init_auto_label_service(mock_classifier)

        # SessionServiceのグローバルインスタンスを直接モック
        with patch(
            "src.training.game_scene.labeling.api.routes.session_routes._session_service.session_exists"
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "src.training.game_scene.labeling.api.routes.session_routes._session_service.get_session"
            ) as mock_get_session:
                mock_get_session.return_value = mock_session

                with patch(
                    "src.training.game_scene.labeling.api.services.auto_label_service.AutoLabelService.predict_frame"
                ) as mock_predict:
                    mock_predict.return_value = {
                        "frame_number": 100,
                        "prediction": {
                            "label": "game",
                            "confidence": 0.95,
                            "probabilities": {"game": 0.95, "menu": 0.03, "loading": 0.02},
                        },
                    }

                    response = client.get(
                        "/api/scene_labeling/sessions/test-session-123/predict/100"
                    )

                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data["frame_number"] == 100
                    assert data["prediction"]["label"] == "game"


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_validation_error(self, client):
        """検証エラーのテスト"""
        response = client.post(
            "/api/scene_labeling/sessions",
            json={},  # video_pathが欠けている
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_not_found_error(self, client):
        """NotFoundエラーのテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.get_session"
        ) as mock_get:
            mock_get.side_effect = NotFoundError("セッション")

            response = client.get("/api/scene_labeling/sessions/non-existent")

            assert response.status_code == 404
            data = json.loads(response.data)
            assert "error" in data
            assert "見つかりません" in data["error"]["message"]

    def test_internal_error(self, client):
        """内部エラーのテスト"""
        with patch(
            "src.training.game_scene.labeling.api.services.session_service.SessionService.create_session"
        ) as mock_create:
            mock_create.side_effect = Exception("予期しないエラー")

            response = client.post(
                "/api/scene_labeling/sessions",
                json={"video_path": "/path/to/video.mp4"},
                content_type="application/json",
            )

            assert response.status_code == 500
            data = json.loads(response.data)
            assert "error" in data
