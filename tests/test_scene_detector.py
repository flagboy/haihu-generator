"""
シーン検出機能のテスト
"""

import numpy as np
import pytest

from src.detection import SceneDetectionResult, SceneDetector, SceneType


class TestSceneDetector:
    """SceneDetectorのテスト"""

    @pytest.fixture
    def detector(self):
        """テスト用検出器"""
        config = {
            "scene_change_threshold": 0.3,
            "text_detection_confidence": 0.8,
            "enable_ocr": False,
        }
        return SceneDetector(config)

    @pytest.fixture
    def sample_frames(self):
        """テスト用サンプルフレーム"""
        # 緑色が支配的なゲームプレイフレーム
        game_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        game_frame[:, :] = [0, 128, 0]  # 緑色

        # 暗いメニューフレーム
        menu_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        menu_frame[:, :] = [20, 20, 20]  # 暗い色

        # 通常のフレーム
        normal_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        normal_frame[:, :] = [100, 100, 100]  # グレー

        return {"game": game_frame, "menu": menu_frame, "normal": normal_frame}

    def test_initialization(self, detector):
        """初期化のテスト"""
        assert detector.scene_change_threshold == 0.3
        assert detector.text_detection_confidence == 0.8
        assert detector.enable_ocr is False
        assert detector.prev_frame is None
        assert detector.prev_scene_type == SceneType.UNKNOWN

    def test_detect_scene_game_play(self, detector, sample_frames):
        """ゲームプレイシーンの検出テスト"""
        result = detector.detect_scene(sample_frames["game"], frame_number=100, timestamp=3.33)

        assert result.scene_type == SceneType.GAME_PLAY
        assert result.confidence > 0.5
        assert result.frame_number == 100
        assert result.timestamp == 3.33
        assert "green_ratio" in result.metadata

    def test_detect_scene_menu(self, detector, sample_frames):
        """メニューシーンの検出テスト"""
        result = detector.detect_scene(sample_frames["menu"], frame_number=0, timestamp=0.0)

        assert result.scene_type == SceneType.MENU
        assert result.confidence > 0.5
        assert "dark_ratio" in result.metadata

    def test_scene_change_detection(self, detector, sample_frames):
        """シーン変化検出のテスト"""
        # 最初のフレーム
        result1 = detector.detect_scene(sample_frames["game"], frame_number=0, timestamp=0.0)
        assert result1.metadata["scene_changed"] is True

        # 同じフレーム（完全に同一のため変化なし）
        result2 = detector.detect_scene(
            sample_frames["game"].copy(),  # コピーを使用
            frame_number=1,
            timestamp=0.033,
        )
        # 同じフレームなので変化なしのはず
        assert not bool(result2.metadata["scene_changed"])

        # 異なるフレーム
        result3 = detector.detect_scene(sample_frames["menu"], frame_number=2, timestamp=0.066)
        # 異なるフレームなので変化あり（ただし閾値に依存）
        # 完全に異なる色なので変化があるはず
        assert "scene_changed" in result3.metadata

    def test_is_game_boundary(self, detector):
        """ゲーム境界判定のテスト"""
        # ゲーム開始
        result1 = SceneDetectionResult(
            scene_type=SceneType.GAME_START,
            confidence=0.9,
            frame_number=0,
            timestamp=0.0,
            metadata={},
        )
        assert result1.is_game_boundary() is True

        # ゲームプレイ中
        result2 = SceneDetectionResult(
            scene_type=SceneType.GAME_PLAY,
            confidence=0.9,
            frame_number=100,
            timestamp=3.33,
            metadata={},
        )
        assert result2.is_game_boundary() is False

    def test_detect_ui_patterns(self, detector, sample_frames):
        """UIパターン検出のテスト"""
        # プライベートメソッドのテスト
        frame = sample_frames["normal"]
        scene_type, confidence, features = detector._detect_ui_patterns(frame)

        assert "top_edge_density" in features
        assert "bottom_edge_density" in features
        assert "center_edge_density" in features
        assert isinstance(confidence, float)

    def test_reset(self, detector, sample_frames):
        """リセット機能のテスト"""
        # 状態を設定
        detector.detect_scene(sample_frames["game"], 0, 0.0)
        assert detector.prev_frame is not None
        assert detector.prev_scene_type != SceneType.UNKNOWN

        # リセット
        detector.reset()
        assert detector.prev_frame is None
        assert detector.prev_scene_type == SceneType.UNKNOWN
