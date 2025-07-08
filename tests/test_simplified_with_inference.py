"""
推測機能を含むSimplifiedActionDetectorの統合テスト
"""

import pytest

from src.tracking.simplified_action_detector import SimplifiedActionDetector


class TestSimplifiedDetectorWithInference:
    """推測機能を含むテストクラス"""

    @pytest.fixture
    def detector_with_inference(self):
        """推測機能を有効にしたdetector"""
        config = {"enable_inference": True}
        return SimplifiedActionDetector(config)

    def test_missing_action_inference(self, detector_with_inference):
        """欠落したアクションの推測テスト"""
        detector = detector_with_inference

        # プレイヤー0の手牌（巡0）
        detector.detect_hand_change(
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"],
            frame_number=1,
        )

        # プレイヤー1,2,3の手牌をスキップ（カメラ切り替えで見逃し）

        # 別のプレイヤーの手牌で手番切り替えを3回
        # 各プレイヤーで異なる手牌を使用
        other_hands = [
            [
                "9m",
                "9m",
                "9m",
                "8p",
                "8p",
                "8p",
                "5s",
                "5s",
                "5s",
                "7z",
                "7z",
                "7z",
                "6z",
            ],  # プレイヤー1
            [
                "1p",
                "2p",
                "3p",
                "4s",
                "5s",
                "6s",
                "1z",
                "1z",
                "6z",
                "6z",
                "6z",
                "7z",
                "7z",
            ],  # プレイヤー2
            [
                "2m",
                "3m",
                "4m",
                "5m",
                "6m",
                "7m",
                "8m",
                "9m",
                "9m",
                "2s",
                "3s",
                "4s",
                "5s",
            ],  # プレイヤー3
        ]
        for i, hand in enumerate(other_hands):
            detector.detect_hand_change(hand, frame_number=2 + i)

        # プレイヤー0の手牌（巡1 = 4手番後）
        # 5zをツモって4zを切った
        detector.detect_hand_change(
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "5z"],
            frame_number=5,
        )

        # デバッグ情報を出力
        print(f"\nCurrent player: {detector.current_player}")
        print(f"Turn number: {detector.turn_number}")
        if detector.inferencer:
            print(f"Player hands history: {detector.inferencer.player_hands_history}")

        # 推測されたアクションを確認
        inferred = detector.get_inferred_actions()
        print(f"Inferred actions: {inferred}")
        assert len(inferred) == 2  # ツモと捨て牌

        assert inferred[0]["action_type"] == "draw"
        assert inferred[0]["tile"] == "5z"
        assert inferred[0]["confidence"] == 0.8
        assert "次巡の手牌から推測" in inferred[0]["reason"]

        assert inferred[1]["action_type"] == "discard"
        assert inferred[1]["tile"] == "4z"

    def test_convert_with_inferred_actions(self, detector_with_inference):
        """推測アクションを含む天鳳形式への変換テスト"""
        detector = detector_with_inference

        # 巡0: プレイヤー0の手牌
        detector.detect_hand_change(["1m"] * 13, frame_number=1)
        detector.detect_hand_change(["1m"] * 14, frame_number=2)  # ツモ
        detector.detect_hand_change(["1m"] * 13, frame_number=3)  # 捨て牌

        # 巡0: プレイヤー1,2,3の手番をスキップ（手番切り替えで進める）
        detector.detect_hand_change(["2m"] * 13, frame_number=4)  # プレイヤー1
        detector.detect_hand_change(["3m"] * 13, frame_number=5)  # プレイヤー2
        detector.detect_hand_change(["4m"] * 13, frame_number=6)  # プレイヤー3

        # 巡1: プレイヤー0に戻る（手牌が変化している）
        # 前回から1mが1枚減って3mが1枚増えた（ツモ3m、切り1m）
        detector.detect_hand_change(["1m"] * 12 + ["3m"], frame_number=7)

        # デバッグ情報を出力
        print(f"\nCurrent player: {detector.current_player}")
        print(f"Turn number: {detector.turn_number}")
        if detector.inferencer:
            print(f"Player hands history: {detector.inferencer.player_hands_history}")
            print(f"Inferred actions: {detector.get_inferred_actions()}")

        # 天鳳形式に変換（推測アクションを含む）
        tenhou_actions = detector.convert_to_tenhou_format(include_inferred=True)
        print(f"Tenhou actions: {tenhou_actions}")

        # 推測されたアクションが含まれているか確認
        inferred_actions = [a for a in tenhou_actions if a.get("inferred")]
        assert len(inferred_actions) > 0

        # 推測アクションには理由が含まれている
        assert "reason" in inferred_actions[0]

    def test_no_inference_when_disabled(self):
        """推測機能が無効の場合のテスト"""
        config = {"enable_inference": False}
        detector = SimplifiedActionDetector(config)

        # 同じシナリオでテスト
        detector.detect_hand_change(["1m"] * 13, frame_number=1)
        detector.detect_hand_change(["2m"] * 13, frame_number=5)  # 手番切り替え
        detector.detect_hand_change(["1m"] * 12 + ["3m"], frame_number=9)

        # 推測アクションは生成されない
        inferred = detector.get_inferred_actions()
        assert len(inferred) == 0

        # 天鳳形式にも含まれない
        tenhou_actions = detector.convert_to_tenhou_format()
        inferred_in_tenhou = [a for a in tenhou_actions if a.get("inferred")]
        assert len(inferred_in_tenhou) == 0
