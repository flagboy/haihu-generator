"""
データ管理APIのテスト
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from web_interface.app import app


@pytest.fixture
def client():
    """テスト用クライアント"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_dataset_manager():
    """モックのDatasetManager"""
    with patch("web_interface.app.DatasetManager") as mock:
        manager = MagicMock()
        mock.return_value = manager
        yield manager


class TestVideoManagementAPI:
    """動画管理APIのテスト"""

    def test_get_videos(self, client, mock_dataset_manager):
        """動画一覧取得のテスト"""
        # モックデータ
        mock_dataset_manager.list_videos.return_value = [
            {
                "id": 1,
                "name": "test1.mp4",
                "path": "/path/to/test1.mp4",
                "upload_date": "2024-01-01",
            },
            {
                "id": 2,
                "name": "test2.mp4",
                "path": "/path/to/test2.mp4",
                "upload_date": "2024-01-02",
            },
        ]
        mock_dataset_manager.get_frame_count.side_effect = [10, 20]
        mock_dataset_manager.get_annotation_count.side_effect = [5, 0]

        response = client.get("/api/videos")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert len(data) == 2
        assert data[0]["frame_count"] == 10
        assert data[0]["annotation_count"] == 5
        assert data[0]["status"] == "annotated"
        assert data[1]["status"] == "pending"

    def test_get_video_detail(self, client, mock_dataset_manager):
        """動画詳細取得のテスト"""
        mock_dataset_manager.get_video_info.return_value = {
            "id": 1,
            "name": "test.mp4",
            "path": "/path/to/test.mp4",
            "upload_date": "2024-01-01",
            "fps": 30,
            "width": 1920,
            "height": 1080,
            "duration": 120,
        }
        mock_dataset_manager.get_frame_count.return_value = 100
        mock_dataset_manager.get_annotation_count.return_value = 50

        response = client.get("/api/videos/1")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["id"] == 1
        assert data["frame_count"] == 100
        assert data["annotation_count"] == 50
        assert data["fps"] == 30

    def test_get_video_detail_not_found(self, client, mock_dataset_manager):
        """存在しない動画の詳細取得のテスト"""
        mock_dataset_manager.get_video_info.return_value = None

        response = client.get("/api/videos/999")
        assert response.status_code == 404

    def test_get_video_frames(self, client, mock_dataset_manager):
        """動画のフレーム一覧取得のテスト"""
        mock_dataset_manager.list_frames.return_value = [
            {"id": 1, "video_id": 1, "frame_number": 0, "timestamp": 0.0, "path": "/frame1.jpg"},
            {"id": 2, "video_id": 1, "frame_number": 1, "timestamp": 0.033, "path": "/frame2.jpg"},
        ]
        mock_dataset_manager.get_frame_annotations.side_effect = [
            [{"id": 1, "class_id": 0}],  # 1つのアノテーション
            [],  # アノテーションなし
        ]

        response = client.get("/api/videos/1/frames")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["total"] == 2
        assert len(data["frames"]) == 2
        assert data["frames"][0]["annotation_count"] == 1
        assert data["frames"][0]["annotated"] is True
        assert data["frames"][1]["annotated"] is False

    def test_delete_video(self, client, mock_dataset_manager):
        """動画削除のテスト"""
        mock_dataset_manager.get_video_info.return_value = {
            "id": 1,
            "name": "test.mp4",
            "path": "/path/to/test.mp4",
            "upload_date": "2024-01-01",
        }

        response = client.delete("/api/videos/1")
        assert response.status_code == 200

        mock_dataset_manager.delete_video.assert_called_once_with(1)

    def test_delete_video_not_found(self, client, mock_dataset_manager):
        """存在しない動画の削除のテスト"""
        mock_dataset_manager.get_video_info.return_value = None

        response = client.delete("/api/videos/999")
        assert response.status_code == 404


class TestDatasetVersionAPI:
    """データセットバージョンAPIのテスト"""

    def test_create_dataset_version(self, client, mock_dataset_manager):
        """データセットバージョン作成のテスト"""
        mock_dataset_manager.create_dataset_version.return_value = {
            "id": "test-version-id",
            "description": "Test version",
            "created_at": "2024-01-01T00:00:00",
        }

        response = client.post("/api/dataset/create_version", json={"description": "Test version"})
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["message"] == "Dataset version created successfully"
        assert data["version"]["id"] == "test-version-id"

    def test_delete_dataset_version(self, client, mock_dataset_manager):
        """データセットバージョン削除のテスト"""
        mock_dataset_manager.list_versions.return_value = [
            {"id": "version-1", "created_at": "2024-01-01"},
            {"id": "version-2", "created_at": "2024-01-02"},
        ]
        mock_dataset_manager.dataset_dir = Path("/tmp/dataset")

        with patch("shutil.rmtree"):
            response = client.delete("/api/dataset/versions/version-1")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["message"] == "Version deleted successfully"

    def test_delete_dataset_version_not_found(self, client, mock_dataset_manager):
        """存在しないバージョンの削除のテスト"""
        mock_dataset_manager.list_versions.return_value = []

        response = client.delete("/api/dataset/versions/non-existent")
        assert response.status_code == 404


class TestExportAPI:
    """エクスポートAPIのテスト"""

    def test_export_yolo_format(self, client, mock_dataset_manager):
        """YOLO形式エクスポートのテスト"""
        # モックデータ設定
        mock_dataset_manager.list_videos.return_value = [
            {"id": 1, "name": "test.mp4", "path": "/path/to/test.mp4"}
        ]
        mock_dataset_manager.list_frames.return_value = [
            {"id": 1, "video_id": 1, "frame_number": 0, "path": "/frame1.jpg"}
        ]
        mock_dataset_manager.get_frame_annotations.return_value = [
            {"id": 1, "class_id": 0, "x": 10, "y": 20, "width": 30, "height": 40, "confidence": 0.9}
        ]
        mock_dataset_manager.create_dataset_version.return_value = {
            "id": "export-version",
            "created_at": "2024-01-01",
        }
        mock_dataset_manager.dataset_dir = Path("/tmp/dataset")

        with patch("src.training.annotation_data.AnnotationData") as mock_annotation:
            annotation_instance = MagicMock()
            mock_annotation.return_value = annotation_instance

            response = client.post("/api/dataset/export", json={"format": "yolo"})
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "yolo" in data["message"]
            assert data["version_id"] == "export-version"

            # AnnotationDataのメソッドが呼ばれたことを確認
            annotation_instance.add_frame_annotation.assert_called()
            annotation_instance.export_yolo_format.assert_called()

    def test_export_coco_format(self, client, mock_dataset_manager):
        """COCO形式エクスポートのテスト"""
        mock_dataset_manager.list_videos.return_value = [
            {"id": 1, "name": "test.mp4", "path": "/path/to/test.mp4"}
        ]
        mock_dataset_manager.list_frames.return_value = [
            {"id": 1, "video_id": 1, "frame_number": 0, "path": "/frame1.jpg"}
        ]
        mock_dataset_manager.get_frame_annotations.return_value = [
            {"id": 1, "class_id": 0, "x": 10, "y": 20, "width": 30, "height": 40}
        ]
        mock_dataset_manager.create_dataset_version.return_value = {
            "id": "export-version",
            "created_at": "2024-01-01",
        }
        mock_dataset_manager.dataset_dir = Path("/tmp/dataset")

        with patch("cv2.imread") as mock_imread:
            mock_imread.return_value = MagicMock(shape=(1080, 1920, 3))

            with patch("builtins.open", create=True), patch("shutil.copy2"):
                response = client.post("/api/dataset/export", json={"format": "coco"})
                assert response.status_code == 200

                data = json.loads(response.data)
                assert "coco" in data["message"]

    def test_export_voc_format(self, client, mock_dataset_manager):
        """Pascal VOC形式エクスポートのテスト"""
        mock_dataset_manager.list_videos.return_value = [
            {"id": 1, "name": "test.mp4", "path": "/path/to/test.mp4"}
        ]
        mock_dataset_manager.list_frames.return_value = [
            {"id": 1, "video_id": 1, "frame_number": 0, "path": "/frame1.jpg"}
        ]
        mock_dataset_manager.get_frame_annotations.return_value = [
            {"id": 1, "class_id": 0, "x": 10, "y": 20, "width": 30, "height": 40}
        ]
        mock_dataset_manager.create_dataset_version.return_value = {
            "id": "export-version",
            "created_at": "2024-01-01",
        }
        mock_dataset_manager.dataset_dir = Path("/tmp/dataset")

        with patch("cv2.imread") as mock_imread:
            mock_imread.return_value = MagicMock(shape=(1080, 1920, 3))

            with patch("builtins.open", create=True), patch("shutil.copy2"):
                response = client.post("/api/dataset/export", json={"format": "voc"})
                assert response.status_code == 200

                data = json.loads(response.data)
                assert "voc" in data["message"]

    def test_export_unsupported_format(self, client, mock_dataset_manager):
        """サポートされていない形式のエクスポートのテスト"""
        response = client.post("/api/dataset/export", json={"format": "unsupported"})
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "Unsupported format" in data["error"]
