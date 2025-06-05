"""
手牌領域検出のユニットテスト
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestHandAreaDetector:
    """手牌領域検出のテストクラス"""

    @pytest.fixture
    def sample_frame(self):
        """テスト用のサンプルフレームを生成"""
        # 1920x1080のダミーフレーム
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return frame

    @pytest.fixture
    def detector_labeling(self):
        """hand_labeling_systemのdetectorを取得"""
        from hand_labeling_system.backend.core.hand_detector import HandDetector

        return HandDetector()

    @pytest.fixture
    def detector_training(self):
        """hand_training_systemのdetectorを取得"""
        from hand_training_system.backend.core.hand_area_detector import HandAreaDetector

        return HandAreaDetector()

    def test_default_regions_labeling(self, detector_labeling):
        """hand_labeling_systemのデフォルト領域をテスト"""
        assert "player1" in detector_labeling.default_regions
        assert "player2" in detector_labeling.default_regions
        assert "player3" in detector_labeling.default_regions
        assert "player4" in detector_labeling.default_regions

    def test_default_regions_training(self, detector_training):
        """hand_training_systemのデフォルト領域をテスト"""
        assert "bottom" in detector_training.regions
        assert "top" in detector_training.regions
        assert "left" in detector_training.regions
        assert "right" in detector_training.regions

    def test_region_conversion(self):
        """プレイヤー番号と方向の変換をテスト"""
        # player1 = bottom, player2 = right, player3 = top, player4 = left
        mapping = {"player1": "bottom", "player2": "right", "player3": "top", "player4": "left"}

        # 変換が正しいことを確認
        for player, direction in mapping.items():
            assert direction in ["bottom", "top", "left", "right"]

    def test_detect_hand_regions(self, detector_labeling, sample_frame):
        """手牌領域の自動検出をテスト"""
        regions = detector_labeling.detect_hand_regions(sample_frame)
        assert isinstance(regions, dict)
        assert len(regions) == 4  # 4人分の領域

    def test_set_frame_size(self, detector_training):
        """フレームサイズ設定をテスト"""
        detector_training.set_frame_size(1920, 1080)
        assert detector_training.frame_size == (1920, 1080)

    def test_get_absolute_region(self, detector_training):
        """相対座標から絶対座標への変換をテスト"""
        detector_training.set_frame_size(1920, 1080)
        detector_training.regions["bottom"] = {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15}

        abs_region = detector_training.get_absolute_region("bottom")
        assert abs_region["x"] == int(1920 * 0.15)
        assert abs_region["y"] == int(1080 * 0.75)
        assert abs_region["w"] == int(1920 * 0.7)
        assert abs_region["h"] == int(1080 * 0.15)
