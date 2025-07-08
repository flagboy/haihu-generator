"""
SimplifiedActionDetectorのテスト
"""

import pytest

from src.tracking.simplified_action_detector import SimplifiedActionDetector


class TestSimplifiedActionDetector:
    """SimplifiedActionDetectorのテストクラス"""

    @pytest.fixture
    def detector(self):
        """テスト用のdetectorインスタンス"""
        return SimplifiedActionDetector()

    def test_initial_detection(self, detector):
        """初回検出のテスト"""
        hand = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"]
        result = detector.detect_hand_change(hand, frame_number=1)

        assert result.action_type == "initial"
        assert result.confidence == 1.0
        assert result.hand_after == hand
        assert detector.previous_hand == hand

    def test_draw_detection(self, detector):
        """ツモ検出のテスト"""
        # 初期手牌（13枚）
        initial_hand = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "2z",
            "3z",
            "4z",
        ]
        detector.detect_hand_change(initial_hand, frame_number=1)

        # ツモ後（14枚）
        after_draw = initial_hand + ["5z"]
        result = detector.detect_hand_change(after_draw, frame_number=2)

        assert result.action_type == "draw"
        assert result.tile == "5z"
        assert result.confidence == 0.9
        assert len(result.hand_after) == 14

    def test_discard_detection(self, detector):
        """捨て牌検出のテスト"""
        # 初期手牌（14枚）
        initial_hand = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "2z",
            "3z",
            "4z",
            "5z",
        ]
        detector.detect_hand_change(initial_hand, frame_number=1)

        # 捨て牌後（13枚）
        after_discard = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "2z",
            "3z",
            "4z",
        ]
        result = detector.detect_hand_change(after_discard, frame_number=2)

        assert result.action_type == "discard"
        assert result.tile == "5z"
        assert result.confidence == 0.9
        assert len(result.hand_after) == 13

    def test_turn_change_detection(self, detector):
        """手番切り替え検出のテスト"""
        # プレイヤー1の手牌
        player1_hand = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "2z",
            "3z",
            "4z",
        ]
        detector.detect_hand_change(player1_hand, frame_number=1)

        # 完全に異なるプレイヤー2の手牌
        player2_hand = [
            "1p",
            "1p",
            "2p",
            "3s",
            "4s",
            "5s",
            "2z",
            "3z",
            "4z",
            "5z",
            "6z",
            "7z",
            "7z",
        ]
        result = detector.detect_hand_change(player2_hand, frame_number=2)

        assert result.action_type == "turn_change"
        assert result.confidence > 0.5
        assert detector.previous_hand == player2_hand

    def test_pon_detection(self, detector):
        """ポン検出のテスト"""
        # 初期手牌（13枚）
        initial_hand = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "1z",
            "2z",
            "3z",
        ]
        detector.detect_hand_change(initial_hand, frame_number=1)

        # ポン後（10枚 - 3枚鳴き）
        after_pon = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "3z"]
        result = detector.detect_hand_change(after_pon, frame_number=2)

        assert result.action_type == "call"
        assert result.confidence == 0.7
        assert result.metadata["call_type"] == "pon_or_chi"

    def test_kan_detection(self, detector):
        """カン検出のテスト"""
        # 初期手牌（14枚 - ツモ後）
        initial_hand = [
            "1m",
            "2m",
            "3m",
            "4p",
            "5p",
            "6p",
            "7s",
            "8s",
            "9s",
            "1z",
            "1z",
            "1z",
            "1z",
            "2z",
        ]
        detector.detect_hand_change(initial_hand, frame_number=1)

        # カン後（10枚 - 4枚カン）
        after_kan = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "2z"]
        result = detector.detect_hand_change(after_kan, frame_number=2)

        assert result.action_type == "call"
        assert result.confidence == 0.7
        assert result.metadata["call_type"] == "kan"

    def test_invalid_hand_size(self, detector):
        """無効な手牌サイズのテスト"""
        # 手牌が少なすぎる
        small_hand = ["1m", "2m", "3m"]
        result = detector.detect_hand_change(small_hand, frame_number=1)

        assert result.action_type == "unknown"
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "invalid_hand_size"

    def test_hand_similarity_calculation(self, detector):
        """手牌類似度計算のテスト"""
        hand1 = ["1m", "2m", "3m", "4p", "5p"]
        hand2 = ["1m", "2m", "3m", "6p", "7p"]

        similarity = detector._calculate_hand_similarity(hand1, hand2)

        # 3/7 = 0.428...
        assert 0.4 < similarity < 0.5

    def test_action_sequence_tracking(self, detector):
        """アクションシーケンスの追跡テスト"""
        # 複数のアクションを実行
        hands = [
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"],  # 初期
            [
                "1m",
                "2m",
                "3m",
                "4p",
                "5p",
                "6p",
                "7s",
                "8s",
                "9s",
                "1z",
                "2z",
                "3z",
                "4z",
                "5z",
            ],  # ツモ
            [
                "1m",
                "2m",
                "3m",
                "4p",
                "5p",
                "6p",
                "7s",
                "8s",
                "9s",
                "1z",
                "2z",
                "3z",
                "4z",
            ],  # 捨て牌
        ]

        for i, hand in enumerate(hands):
            detector.detect_hand_change(hand, frame_number=i)

        sequence = detector.get_action_sequence()
        assert len(sequence) == 2  # initialは記録されない
        assert sequence[0]["action_type"] == "draw"
        assert sequence[1]["action_type"] == "discard"

    def test_convert_to_tenhou_format(self, detector):
        """天鳳形式への変換テスト"""
        # アクションシーケンスを作成
        detector.detect_hand_change(["1m"] * 13, frame_number=1)  # 初期
        detector.detect_hand_change(["1m"] * 14, frame_number=2)  # ツモ
        detector.detect_hand_change(["1m"] * 13, frame_number=3)  # 捨て牌
        detector.detect_hand_change(["2m"] * 13, frame_number=4)  # 手番切り替え
        detector.detect_hand_change(["2m"] * 14, frame_number=5)  # ツモ

        # デバッグ用：アクションシーケンスを確認
        sequence = detector.get_action_sequence()
        print(f"\nAction sequence: {sequence}")

        tenhou_actions = detector.convert_to_tenhou_format()
        print(f"Tenhou actions: {tenhou_actions}")

        assert len(tenhou_actions) == 3  # turn_changeとinitialは除外される
        assert tenhou_actions[0]["player"] == 0
        assert tenhou_actions[0]["type"] == "draw"
        assert tenhou_actions[1]["player"] == 0
        assert tenhou_actions[1]["type"] == "discard"
        assert tenhou_actions[2]["player"] == 1  # 手番切り替え後
        assert tenhou_actions[2]["type"] == "draw"

    def test_reset(self, detector):
        """リセット機能のテスト"""
        # いくつかのアクションを追加
        detector.detect_hand_change(["1m"] * 13, frame_number=1)
        detector.detect_hand_change(["1m"] * 14, frame_number=2)

        # リセット
        detector.reset()

        assert detector.previous_hand == []
        assert detector.actions_sequence == []

        # リセット後の最初の検出
        result = detector.detect_hand_change(["2m"] * 13, frame_number=3)
        assert result.action_type == "initial"
