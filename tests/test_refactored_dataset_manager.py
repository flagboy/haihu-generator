"""
リファクタリングされたDatasetManagerのテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.training.annotation_data import (
    AnnotationData,
    BoundingBox,
    FrameAnnotation,
    TileAnnotation,
    VideoAnnotation,
)
from src.training.data.database import DatabaseConnection, DatabaseMigration
from src.training.data.models import (
    BoundingBox as ModelBBox,
)
from src.training.data.models import (
    Frame,
    Video,
)
from src.training.data.models import (
    TileAnnotation as ModelTileAnnotation,
)
from src.training.data.repositories import (
    FrameRepository,
    VideoRepository,
)
from src.training.data.services import DatasetService
from src.training.refactored_dataset_manager import RefactoredDatasetManager


class TestDatabaseConnection:
    """DatabaseConnectionのテスト"""

    def test_connection_creation(self):
        """接続作成のテスト"""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            connection = DatabaseConnection(tmp.name)
            assert connection.exists()

    def test_transaction_rollback(self):
        """トランザクションロールバックのテスト"""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            connection = DatabaseConnection(tmp.name)

            # テーブル作成
            connection.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

            # ロールバックテスト
            try:
                with connection.transaction() as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
                    # 意図的にエラーを発生させる
                    raise Exception("Test error")
            except Exception:
                pass

            # データが挿入されていないことを確認
            result = connection.fetch_one("SELECT COUNT(*) as count FROM test")
            assert result["count"] == 0


class TestModels:
    """モデルクラスのテスト"""

    def test_video_model(self):
        """Videoモデルのテスト"""
        video = Video(
            video_id="test_video_1",
            name="test_video.mp4",
            path="/path/to/video.mp4",
            duration=300.0,
            fps=30.0,
            width=1920,
            height=1080,
        )

        # 基本属性の確認
        assert video.video_id == "test_video_1"
        assert video.name == "test_video.mp4"
        assert video.created_at is not None
        assert video.updated_at is not None

        # メタデータ更新
        video.update_metadata("key", "value")
        assert video.metadata["key"] == "value"

        # 辞書変換
        video_dict = video.to_dict()
        assert video_dict["video_id"] == "test_video_1"
        assert "created_at" in video_dict

        # 辞書から生成
        new_video = Video.from_dict(video_dict)
        assert new_video.video_id == video.video_id

    def test_frame_model(self):
        """Frameモデルのテスト"""
        frame = Frame(
            frame_id="frame_1",
            video_id="video_1",
            image_path="/path/to/frame.jpg",
            timestamp=1.5,
            width=1920,
            height=1080,
        )

        # 基本属性の確認
        assert frame.is_valid
        assert frame.scene_type == "game"
        assert frame.is_annotated() is False

        # 品質スコア更新
        frame.update_quality_score(0.9)
        assert frame.quality_score == 0.9

        # 無効化
        frame.mark_as_invalid("ブレている")
        assert not frame.is_valid
        assert "ブレている" in frame.notes

    def test_tile_annotation_model(self):
        """TileAnnotationモデルのテスト"""
        bbox = ModelBBox(x1=100, y1=200, x2=150, y2=250)
        annotation = ModelTileAnnotation(
            annotation_id="ann_1",
            frame_id="frame_1",
            tile_id="1m",
            bbox=bbox,
            confidence=0.95,
            area_type="hand",
        )

        # バウンディングボックスのプロパティ
        assert bbox.width == 50
        assert bbox.height == 50
        assert bbox.area == 2500
        assert bbox.center_x == 125
        assert bbox.center_y == 225

        # YOLO形式変換
        yolo_bbox = bbox.to_yolo_format(1920, 1080)
        assert len(yolo_bbox) == 4

        # アノテーションの信頼性
        assert annotation.is_reliable(0.8)


class TestRepositories:
    """リポジトリのテスト"""

    @pytest.fixture
    def test_db(self):
        """テスト用データベース"""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            connection = DatabaseConnection(tmp.name)
            migration = DatabaseMigration(connection)
            migration.create_tables()
            yield connection

    def test_video_repository(self, test_db):
        """VideoRepositoryのテスト"""
        repo = VideoRepository(test_db)

        # 作成
        video = Video(
            video_id="test_1",
            name="test.mp4",
            path="/test/path.mp4",
        )
        assert repo.create(video)

        # 検索
        found = repo.find_by_id("test_1")
        assert found is not None
        assert found.name == "test.mp4"

        # 更新
        found.name = "updated.mp4"
        assert repo.update(found)

        # 全件取得
        all_videos = repo.find_all()
        assert len(all_videos) == 1

        # 削除
        assert repo.delete("test_1")
        assert repo.find_by_id("test_1") is None

    def test_frame_repository(self, test_db):
        """FrameRepositoryのテスト"""
        # 先に動画を作成
        video_repo = VideoRepository(test_db)
        video = Video(video_id="video_1", name="test.mp4", path="/test.mp4")
        video_repo.create(video)

        # フレームリポジトリのテスト
        repo = FrameRepository(test_db)

        frame = Frame(
            frame_id="frame_1",
            video_id="video_1",
            image_path="/frame1.jpg",
            timestamp=1.0,
            width=1920,
            height=1080,
        )
        assert repo.create(frame)

        # 動画IDで検索
        frames = repo.find_by_video_id("video_1")
        assert len(frames) == 1
        assert frames[0].frame_id == "frame_1"

        # カウント
        count = repo.count_by_video_id("video_1")
        assert count == 1


class TestServices:
    """サービスのテスト"""

    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config_manager = Mock()
        config_manager.get_config.return_value = {
            "training": {
                "database_path": ":memory:",
                "dataset_root": "data/training",
            }
        }
        return config_manager

    def test_dataset_service(self, mock_config):
        """DatasetServiceのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.get_config.return_value = {
                "training": {
                    "database_path": str(Path(tmpdir) / "test.db"),
                    "dataset_root": tmpdir,
                }
            }
            service = DatasetService(mock_config)

            # アノテーションデータの作成
            annotation_data = AnnotationData()

            video_annotation = VideoAnnotation(
                video_id="video_1",
                video_path="/test.mp4",
                video_name="test.mp4",
                duration=300.0,
                fps=30.0,
                width=1920,
                height=1080,
                frames=[],
            )

            frame_annotation = FrameAnnotation(
                frame_id="frame_1",
                image_path="/frame1.jpg",
                image_width=1920,
                image_height=1080,
                timestamp=1.0,
                tiles=[
                    TileAnnotation(
                        tile_id="1m",
                        bbox=BoundingBox(x1=100, y1=200, x2=150, y2=250),
                        confidence=0.95,
                    )
                ],
            )

            video_annotation.frames.append(frame_annotation)
            annotation_data.video_annotations["video_1"] = video_annotation

            # 保存と読み込み
            assert service.save_annotation_data(annotation_data)

            loaded_data = service.load_annotation_data()
            assert len(loaded_data.video_annotations) == 1
            assert "video_1" in loaded_data.video_annotations

            # 統計情報
            stats = service.get_dataset_statistics()
            assert stats["video_count"] == 1
            assert stats["frame_count"] == 1
            assert stats["annotation_count"] == 1


class TestRefactoredDatasetManager:
    """RefactoredDatasetManagerのテスト"""

    @pytest.fixture
    def manager(self):
        """テスト用マネージャー"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_manager = Mock()
            config_manager.get_config.return_value = {
                "training": {
                    "database_path": str(Path(tmpdir) / "test.db"),
                    "dataset_root": tmpdir,
                }
            }
            yield RefactoredDatasetManager(config_manager)

    def test_save_and_load_annotation_data(self, manager):
        """アノテーションデータの保存と読み込み"""
        # テストデータ作成
        annotation_data = AnnotationData()

        video_annotation = VideoAnnotation(
            video_id="video_1",
            video_path="/test.mp4",
            video_name="test.mp4",
            duration=300.0,
            fps=30.0,
            width=1920,
            height=1080,
            frames=[
                FrameAnnotation(
                    frame_id="frame_1",
                    image_path="/frame1.jpg",
                    image_width=1920,
                    image_height=1080,
                    timestamp=1.0,
                    tiles=[],
                )
            ],
        )

        annotation_data.video_annotations["video_1"] = video_annotation

        # 保存
        assert manager.save_annotation_data(annotation_data)

        # 読み込み
        loaded = manager.load_annotation_data()
        assert len(loaded.video_annotations) == 1

    def test_version_management(self, manager):
        """バージョン管理のテスト"""
        # アノテーションデータ作成
        annotation_data = AnnotationData()
        video_annotation = VideoAnnotation(
            video_id="video_1",
            video_path="/test.mp4",
            video_name="test.mp4",
            duration=300.0,
            fps=30.0,
            width=1920,
            height=1080,
            frames=[],
        )
        annotation_data.video_annotations["video_1"] = video_annotation

        # バージョン作成
        version_id = manager.create_dataset_version(annotation_data, "v1.0", "Initial version")
        assert version_id is not None

        # バージョン一覧
        versions = manager.list_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == "v1.0"

    def test_compatibility_methods(self, manager):
        """互換性メソッドのテスト"""
        # 統計情報
        stats = manager.get_dataset_statistics()
        assert "video_count" in stats
        assert "dataset_root" in stats

        # 動画一覧（空）
        videos = manager.list_videos()
        assert isinstance(videos, list)
