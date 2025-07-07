"""
SceneDetectorの拡張機能のテスト
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.detection import SceneDetector, SceneType
from src.detection.tile_detector import DetectionResult, TileDetector


class TestSceneDetectorEnhanced:
    """SceneDetectorの拡張機能のテスト"""

    @pytest.fixture
    def scene_detector(self):
        """SceneDetectorのインスタンス"""
        config = {
            "confidence_threshold": 0.7,
            "enable_tile_detection": True,
        }
        return SceneDetector(config)

    @pytest.fixture
    def mock_frame(self):
        """モックフレーム"""
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_detect_by_tile_arrangement_with_tile_detector(self, scene_detector, mock_frame):
        """TileDetectorを使用した牌配置による検出のテスト"""
        # TileDetectorのモックを作成
        mock_tile_detector = Mock(spec=TileDetector)

        # 検出結果のモック
        mock_detections = [
            DetectionResult(
                bbox=(100, 500, 150, 550), confidence=0.9, class_id=0, class_name="tile"
            ),
            DetectionResult(
                bbox=(200, 500, 250, 550), confidence=0.8, class_id=0, class_name="tile"
            ),
        ]

        # detect_tilesメソッドのモック
        mock_tile_detector.detect_tiles.return_value = mock_detections
        mock_tile_detector.load_model.return_value = True

        # TileDetectorのインスタンス化をモック
        with patch("src.detection.tile_detector.TileDetector", return_value=mock_tile_detector):
            with patch("src.utils.config.ConfigManager"):
                # classify_tile_areasメソッドのモック
                # 手牌と捨て牌の両方がある場合をシミュレート
                mock_tile_detector.classify_tile_areas.return_value = {
                    "hand_tiles": [mock_detections[0]],  # 手牌1枚
                    "discarded_tiles": [mock_detections[1]],  # 捨て牌1枚
                    "called_tiles": [],
                }

                # _detect_by_tile_arrangementを呼び出し
                scene_type, confidence, metadata = scene_detector._detect_by_tile_arrangement(
                    mock_frame
                )

                # 検証
                assert mock_tile_detector.load_model.called
                assert mock_tile_detector.detect_tiles.called
                assert scene_type == SceneType.GAME_PLAY  # 手牌と捨て牌がある場合
                assert confidence > 0.5
                assert "total_tiles" in metadata
                assert metadata["total_tiles"] == 2
                assert metadata["hand_tiles"] == 1
                assert metadata["discarded_tiles"] == 1

    def test_detect_by_tile_arrangement_no_tiles(self, scene_detector, mock_frame):
        """牌が検出されない場合のテスト"""
        # TileDetectorのモックを作成
        mock_tile_detector = Mock(spec=TileDetector)

        # 空の検出結果
        mock_tile_detector.detect_tiles.return_value = []
        mock_tile_detector.load_model.return_value = True

        # TileDetectorのインスタンス化をモック
        with patch("src.detection.tile_detector.TileDetector", return_value=mock_tile_detector):
            with patch("src.utils.config.ConfigManager"):
                # classify_tile_areasメソッドのモック
                mock_tile_detector.classify_tile_areas.return_value = {
                    "hand_tiles": [],
                    "discarded_tiles": [],
                    "called_tiles": [],
                }

                # _detect_by_tile_arrangementを呼び出し
                scene_type, confidence, metadata = scene_detector._detect_by_tile_arrangement(
                    mock_frame
                )

                # 検証
                assert scene_type == SceneType.MENU  # 牌が検出されない場合はメニュー
                assert confidence == 0.7
                assert metadata["total_tiles"] == 0

    def test_detect_by_tile_arrangement_initialization_error(self, scene_detector, mock_frame):
        """TileDetector初期化エラーのテスト"""
        # TileDetectorの初期化で例外を発生させる
        with patch(
            "src.detection.tile_detector.TileDetector", side_effect=Exception("初期化エラー")
        ):
            # _detect_by_tile_arrangementを呼び出し
            scene_type, confidence, metadata = scene_detector._detect_by_tile_arrangement(
                mock_frame
            )

            # エラー時はUNKNOWNを返すことを確認
            assert scene_type == SceneType.UNKNOWN
            assert confidence == 0.0
            assert metadata == {}

    def test_detect_scene_with_tile_detection_enabled(self, scene_detector, mock_frame):
        """牌検出が有効な場合のシーン検出テスト"""
        # 各検出メソッドをモック
        with patch.object(
            scene_detector, "_detect_by_histogram", return_value=(SceneType.MENU, 0.6, {})
        ):
            with patch.object(
                scene_detector, "_detect_by_text", return_value=(SceneType.UNKNOWN, 0.0, {})
            ):
                with patch.object(
                    scene_detector, "_detect_ui_patterns", return_value=(SceneType.UNKNOWN, 0.0, {})
                ):
                    with patch.object(
                        scene_detector,
                        "_detect_by_tile_arrangement",
                        return_value=(SceneType.GAME_PLAY, 0.9, {"total_tiles": 10}),
                    ):
                        # detect_sceneを呼び出し
                        result = scene_detector.detect_scene(mock_frame, 1, 1.0)

                        # 牌配置による検出が呼ばれたことを確認
                        scene_detector._detect_by_tile_arrangement.assert_called_once_with(
                            mock_frame
                        )

                        # 最も信頼度の高い結果が選ばれることを確認
                        assert result.scene_type == SceneType.GAME_PLAY
                        assert result.confidence > 0.8  # integrate_resultsで調整される
                        assert "total_tiles" in result.metadata

    def test_detect_scene_with_histogram_detection(self, scene_detector, mock_frame):
        """ヒストグラムベースのシーン検出テスト"""
        # 各検出メソッドをモック
        with patch.object(
            scene_detector,
            "_detect_by_histogram",
            return_value=(SceneType.GAME_PLAY, 0.8, {"green_ratio": 0.7}),
        ):
            with patch.object(
                scene_detector, "_detect_by_text", return_value=(SceneType.UNKNOWN, 0.0, {})
            ):
                with patch.object(
                    scene_detector, "_detect_ui_patterns", return_value=(SceneType.UNKNOWN, 0.0, {})
                ):
                    with patch.object(
                        scene_detector,
                        "_detect_by_tile_arrangement",
                        return_value=(SceneType.UNKNOWN, 0.0, {}),
                    ):
                        # detect_sceneを呼び出し
                        result = scene_detector.detect_scene(mock_frame, 1, 1.0)

                        # 色ヒストグラムの結果が使われることを確認
                        assert result.scene_type == SceneType.GAME_PLAY
                        assert result.confidence > 0.7  # integrate_resultsで調整される
                        assert "green_ratio" in result.metadata
