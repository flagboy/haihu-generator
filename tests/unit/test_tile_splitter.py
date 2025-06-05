"""
牌分割のユニットテスト
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# テスト対象モジュールのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestTileSplitter:
    """牌分割のテストクラス"""

    @pytest.fixture
    def sample_hand_area(self):
        """テスト用の手牌領域画像を生成"""
        # 14枚の牌が並んでいる想定の画像
        # 幅: 40px * 14枚 = 560px
        # 高さ: 60px
        hand_area = np.ones((60, 560, 3), dtype=np.uint8) * 255

        # 牌の境界を黒線で描画
        for i in range(1, 14):
            x = i * 40
            cv2.line(hand_area, (x, 0), (x, 59), (0, 0, 0), 1)

        return hand_area

    @pytest.fixture
    def tile_splitter(self):
        """TileSplitterのインスタンスを取得"""
        from hand_training_system.backend.core.tile_splitter import TileSplitter

        return TileSplitter()

    def test_split_tiles_basic(self, tile_splitter, sample_hand_area):
        """基本的な牌分割のテスト"""
        tiles = tile_splitter.split_tiles(sample_hand_area)

        # 14枚の牌が検出されることを確認
        assert len(tiles) == 14

        # 各牌のサイズを確認
        for tile in tiles:
            assert tile.shape[0] == 60  # 高さ
            assert tile.shape[1] == 40  # 幅

    def test_tile_count_estimation(self, tile_splitter, sample_hand_area):
        """牌の枚数推定のテスト"""
        count = tile_splitter.estimate_tile_count(sample_hand_area)
        assert count == 14

    def test_adjust_tile_boundaries(self, tile_splitter):
        """牌の境界調整のテスト"""
        # 初期境界
        boundaries = [(i * 40, (i + 1) * 40) for i in range(14)]

        # 境界調整メソッドの存在を確認
        assert hasattr(tile_splitter, "adjust_boundaries")

    def test_enhance_tile_image(self, tile_splitter):
        """牌画像の補正・強調のテスト"""
        # ダミーの牌画像
        tile_img = np.ones((60, 40, 3), dtype=np.uint8) * 128

        # 補正メソッドの存在を確認
        assert hasattr(tile_splitter, "enhance_tile_image")

        # 補正後の画像を取得
        enhanced = tile_splitter.enhance_tile_image(tile_img)
        assert enhanced.shape == tile_img.shape

    def test_horizontal_tiles(self, tile_splitter):
        """横向き牌の検出テスト"""
        # 横向きの牌を含む手牌領域
        hand_area = np.ones((80, 520, 3), dtype=np.uint8) * 255

        # 横向き牌の検出メソッドの存在を確認
        assert hasattr(tile_splitter, "detect_horizontal_tiles")

    def test_empty_hand_area(self, tile_splitter):
        """空の手牌領域のテスト"""
        empty_area = np.zeros((60, 100, 3), dtype=np.uint8)
        tiles = tile_splitter.split_tiles(empty_area)

        # 空の場合は0枚または空リストを返す
        assert len(tiles) == 0 or tiles == []
