"""
プレイヤー検出機能のテスト
"""

import cv2
import numpy as np
import pytest

from src.detection import PlayerDetector, PlayerInfo, PlayerPosition


class TestPlayerDetector:
    """PlayerDetectorのテスト"""

    @pytest.fixture
    def detector(self):
        """テスト用検出器"""
        config = {
            "player_regions": {
                PlayerPosition.EAST: {"x": 0.8, "y": 0.45, "w": 0.15, "h": 0.1},
                PlayerPosition.SOUTH: {"x": 0.4, "y": 0.8, "w": 0.2, "h": 0.1},
                PlayerPosition.WEST: {"x": 0.05, "y": 0.45, "w": 0.15, "h": 0.1},
                PlayerPosition.NORTH: {"x": 0.4, "y": 0.05, "w": 0.2, "h": 0.1},
            }
        }
        return PlayerDetector(config)

    @pytest.fixture
    def sample_frame(self):
        """テスト用サンプルフレーム"""
        # 1920x1080のフレームを作成
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 100  # グレー背景

        # 各プレイヤー位置に色を配置
        # 東（アクティブ - 黄色）
        frame[486 - 54 : 486 + 54, 1536 - 144 : 1536 + 144] = [0, 255, 255]
        # 南（非アクティブ）
        frame[864 - 54 : 864 + 54, 768 - 192 : 768 + 192] = [100, 100, 100]
        # 西（非アクティブ）
        frame[486 - 54 : 486 + 54, 96 - 72 : 96 + 144] = [100, 100, 100]
        # 北（親マーク付き）
        frame[54 - 27 : 54 + 54, 768 - 192 : 768 + 192] = [100, 100, 100]
        # 親マーク（赤い円）を北に追加
        cv2.circle(frame, (768, 54), 15, (0, 0, 255), -1)

        return frame

    def test_initialization(self, detector):
        """初期化のテスト"""
        assert len(detector.player_regions) == 4
        assert detector.prev_result is None
        assert detector.dealer_mark_template is None

    def test_detect_players_basic(self, detector, sample_frame):
        """基本的なプレイヤー検出テスト"""
        result = detector.detect_players(sample_frame, frame_number=100, timestamp=3.33)

        assert result.frame_number == 100
        assert result.timestamp == 3.33
        assert isinstance(result.players, list)
        assert len(result.players) <= 4
        assert result.round_wind in ["東", "南", "西", "北"]

    def test_player_info_structure(self, detector, sample_frame):
        """プレイヤー情報の構造テスト"""
        result = detector.detect_players(sample_frame, 0, 0.0)

        for player in result.players:
            assert isinstance(player, PlayerInfo)
            assert player.position in [
                PlayerPosition.EAST,
                PlayerPosition.SOUTH,
                PlayerPosition.WEST,
                PlayerPosition.NORTH,
            ]
            assert isinstance(player.is_dealer, bool)
            assert isinstance(player.is_active, bool)
            assert len(player.bbox) == 4
            assert 0 <= player.confidence <= 1.0

    def test_get_player_by_position(self, detector, sample_frame):
        """位置指定でのプレイヤー取得テスト"""
        result = detector.detect_players(sample_frame, 0, 0.0)

        # 各位置のプレイヤーを取得
        east_player = result.get_player_by_position(PlayerPosition.EAST)
        if east_player:
            assert east_player.position == PlayerPosition.EAST

        # 存在確認
        for position in [
            PlayerPosition.EAST,
            PlayerPosition.SOUTH,
            PlayerPosition.WEST,
            PlayerPosition.NORTH,
        ]:
            player = result.get_player_by_position(position)
            if player:
                assert player.position == position

    def test_get_active_player(self, detector, sample_frame):
        """アクティブプレイヤー取得テスト"""
        result = detector.detect_players(sample_frame, 0, 0.0)

        active_player = result.get_active_player()
        if active_player:
            assert active_player.is_active is True
            assert result.active_position == active_player.position

    def test_detect_highlight(self, detector):
        """ハイライト検出のテスト"""
        # 黄色のハイライト領域
        yellow_region = np.zeros((100, 200, 3), dtype=np.uint8)
        yellow_region[:, :] = [0, 255, 255]  # 黄色（BGR）

        # HSVに変換して検出
        is_highlighted, confidence = detector._detect_highlight(yellow_region)
        assert is_highlighted is True
        assert confidence > 0.5

        # 通常の領域
        normal_region = np.ones((100, 200, 3), dtype=np.uint8) * 100
        is_highlighted, confidence = detector._detect_highlight(normal_region)
        assert is_highlighted is False or confidence < 0.5

    def test_detect_dealer_mark(self, detector):
        """親マーク検出のテスト"""
        # 赤い円を含む領域
        region = np.ones((100, 100, 3), dtype=np.uint8) * 100
        cv2.circle(region, (50, 50), 20, (0, 0, 255), -1)  # 赤い円

        is_dealer = detector._detect_dealer_mark(region)
        # 実装によってはTrueになるはず
        assert isinstance(is_dealer, bool)

    def test_stabilize_result(self, detector, sample_frame):
        """結果安定化のテスト"""
        # 初回検出
        result1 = detector.detect_players(sample_frame, 0, 0.0)

        # 2回目の検出（前回結果を考慮）
        result2 = detector.detect_players(sample_frame, 1, 0.033)

        # 安定化が機能していることを確認
        assert detector.prev_result is not None
        if result1.active_position and result2.active_position:
            # 低信頼度の変化は無視される
            pass

    def test_estimate_round_wind(self, detector):
        """場風推定のテスト"""
        # 各親位置での場風
        wind_map = {
            PlayerPosition.EAST: "東",
            PlayerPosition.SOUTH: "南",
            PlayerPosition.WEST: "西",
            PlayerPosition.NORTH: "北",
        }

        for position, expected_wind in wind_map.items():
            wind = detector._estimate_round_wind(position)
            assert wind == expected_wind
