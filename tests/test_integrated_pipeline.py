"""
統合パイプラインのテスト

対局画面検出 → 牌検出 → 牌譜生成の統合フローをテスト
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.pipeline.ai_pipeline import AIPipeline
from src.utils.config import ConfigManager


class TestIntegratedPipeline:
    """統合パイプラインのテストクラス"""

    @pytest.fixture
    def mock_config(self):
        """モック設定"""
        config = MagicMock(spec=ConfigManager)
        config.get_config.return_value = {
            "ai": {
                "enable_scene_filtering": True,
                "scene_confidence_threshold": 0.8,
                "detection": {
                    "confidence_threshold": 0.5,
                },
                "training": {
                    "batch_size": 8,
                },
            },
            "system": {
                "max_workers": 4,
            },
            "directories": {
                "game_scene_models": "models/game_scene",
            },
        }
        return config

    @pytest.fixture
    def game_scene_frame(self):
        """対局画面のモックフレーム"""
        # 緑色の背景（麻雀卓を模擬）
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [0, 128, 0]  # 緑色

        # 牌を模擬（白い長方形）
        for i in range(4):
            x = 100 + i * 150
            y = 800
            cv2.rectangle(frame, (x, y), (x + 100, y + 140), (255, 255, 255), -1)

        return frame

    @pytest.fixture
    def non_game_scene_frame(self):
        """非対局画面のモックフレーム"""
        # 青色の背景（メニュー画面を模擬）
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # 青色

        # テキストを模擬
        cv2.putText(frame, "MENU", (800, 540), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

        return frame

    def test_scene_filtering_enabled(self, mock_config, game_scene_frame, non_game_scene_frame):
        """対局画面フィルタリングが有効な場合のテスト"""
        with (
            patch("src.pipeline.ai_pipeline.TileDetector") as mock_detector,
            patch("src.pipeline.ai_pipeline.TileClassifier"),
            patch("src.pipeline.ai_pipeline.GameSceneClassifier") as mock_scene_classifier,
        ):
            # モックの設定
            mock_scene_instance = MagicMock()
            mock_scene_classifier.return_value = mock_scene_instance

            # 対局画面と非対局画面の判定を設定
            mock_scene_instance.predict.side_effect = [
                (True, 0.95),  # game_scene_frame は対局画面
                (False, 0.2),  # non_game_scene_frame は非対局画面
            ]

            # 牌検出のモック
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect_tiles.return_value = [
                MagicMock(bbox=[100, 800, 200, 940], confidence=0.9),
                MagicMock(bbox=[250, 800, 350, 940], confidence=0.85),
            ]
            mock_detector_instance.classify_tile_areas.return_value = {
                "hand": [MagicMock(bbox=[100, 800, 200, 940])],
            }

            # パイプライン作成
            pipeline = AIPipeline(mock_config)

            # 対局画面の処理
            result1 = pipeline.process_frame(game_scene_frame, frame_id=1)
            assert len(result1.detections) == 2  # 牌が検出される
            assert pipeline.stats["game_scene_frames"] == 1
            assert pipeline.stats["skipped_frames"] == 0

            # 非対局画面の処理
            result2 = pipeline.process_frame(non_game_scene_frame, frame_id=2)
            assert len(result2.detections) == 0  # 牌は検出されない
            assert pipeline.stats["game_scene_frames"] == 1  # 増えない
            assert pipeline.stats["skipped_frames"] == 1

    def test_scene_filtering_disabled(self, mock_config, game_scene_frame, non_game_scene_frame):
        """対局画面フィルタリングが無効な場合のテスト"""
        # フィルタリングを無効化
        mock_config.get_config.return_value["ai"]["enable_scene_filtering"] = False

        with (
            patch("src.pipeline.ai_pipeline.TileDetector") as mock_detector,
            patch("src.pipeline.ai_pipeline.TileClassifier"),
        ):
            # 牌検出のモック
            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_detector_instance.detect_tiles.return_value = [
                MagicMock(bbox=[100, 800, 200, 940], confidence=0.9),
            ]
            mock_detector_instance.classify_tile_areas.return_value = {}

            # パイプライン作成
            pipeline = AIPipeline(mock_config)

            # 両方のフレームで牌検出が実行される
            pipeline.process_frame(game_scene_frame, frame_id=1)
            pipeline.process_frame(non_game_scene_frame, frame_id=2)

            # 両方とも処理される
            assert mock_detector_instance.detect_tiles.call_count == 2
            assert pipeline.stats["skipped_frames"] == 0

    def test_resource_management(self, mock_config):
        """リソース管理のテスト"""
        from src.utils.video_capture_manager import VideoCaptureContext

        # 一時的な動画ファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

            # ダミーの動画を作成
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, 1.0, (640, 480))

            # 3フレーム書き込み
            for _ in range(3):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)

            writer.release()

            # VideoCaptureContextのテスト
            try:
                with VideoCaptureContext(video_path) as cap:
                    assert cap.isOpened()
                    ret, frame = cap.read()
                    assert ret
                    assert frame is not None
                # コンテキスト終了後、自動的に解放される
            finally:
                # 一時ファイルを削除
                Path(video_path).unlink(missing_ok=True)

    def test_config_path_resolution(self):
        """設定パスの解決テスト"""
        # ConfigManagerの初期化を回避するため、scene_routesのインポート前にモックする
        with patch("src.utils.config.ConfigManager") as mock_config_manager:
            mock_instance = MagicMock()
            mock_config_manager.return_value = mock_instance
            mock_instance.get_config.return_value = {
                "directories": {"game_scene_db": "test/path/game_scene_labels.db"}
            }

            # インポート時にモックされたConfigManagerが使われる
            from src.training.game_scene.labeling.api.scene_routes import get_db_path

            db_path = get_db_path()
            assert "test/path/game_scene_labels.db" in str(db_path)

    def test_scene_dataset_with_config(self, mock_config):
        """SceneDatasetの設定管理テスト"""
        from src.training.game_scene.learning.scene_dataset import SceneDataset

        # テスト用のDBとキャッシュディレクトリ
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test.db"
            test_cache = Path(tmpdir) / "cache"

            mock_config.get_config.return_value = {
                "directories": {
                    "game_scene_db": str(test_db),
                    "game_scene_cache": str(test_cache),
                }
            }

            # データセット作成（設定から読み込まれるはず）
            dataset = SceneDataset(config_manager=mock_config)

            assert test_db.name in dataset.db_path
            assert test_cache.name in str(dataset.cache_dir)
