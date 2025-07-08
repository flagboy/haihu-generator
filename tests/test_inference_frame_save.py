"""
推測フレーム保存機能のテスト
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tracking.action_inferencer import ActionInferencer
from src.tracking.inference_frame_manager import InferenceFrameManager
from src.tracking.simplified_action_detector import SimplifiedActionDetector


class TestInferenceFrameSave:
    """推測フレーム保存機能のテストクラス"""

    @pytest.fixture
    def test_dir(self, tmp_path):
        """テスト用ディレクトリ"""
        test_dir = tmp_path / "test_inference_frames"
        test_dir.mkdir(exist_ok=True)
        yield test_dir
        # クリーンアップ
        if test_dir.exists():
            shutil.rmtree(test_dir)

    @pytest.fixture
    def frame_manager(self, test_dir):
        """テスト用フレーム管理器"""
        return InferenceFrameManager(str(test_dir))

    @pytest.fixture
    def sample_frame(self):
        """サンプルフレーム画像"""
        # 640x480のダミー画像を作成
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # テスト用のテキストを描画
        cv2.putText(
            frame,
            "Test Frame",
            (200, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        return frame

    def test_save_inference_frame(self, frame_manager, sample_frame):
        """推測フレーム保存のテスト"""
        frame_id = frame_manager.save_inference_frame(
            frame=sample_frame,
            frame_number=100,
            turn_number=1,
            player_index=0,
            action_type="draw",
            inferred_tile="5z",
            confidence=0.8,
            reason="次巡の手牌から推測",
            prev_hand=[
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
            ],
            curr_hand=[
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
                "5z",
            ],
        )

        # フレームIDが生成されたか確認
        assert frame_id is not None
        assert "f100_t1_p0" in frame_id

        # 画像ファイルが保存されたか確認
        image_path = frame_manager.images_dir / f"{frame_id}.jpg"
        assert image_path.exists()

        # インデックスに追加されたか確認
        assert len(frame_manager.frames_index) == 1
        frame_info = frame_manager.frames_index[0]
        assert frame_info.frame_number == 100
        assert frame_info.action_type == "draw"
        assert frame_info.inferred_tile == "5z"

    def test_save_without_frame(self, frame_manager):
        """フレームなしでの保存テスト"""
        frame_id = frame_manager.save_inference_frame(
            frame=None,  # フレームなし
            frame_number=200,
            turn_number=2,
            player_index=1,
            action_type="discard",
            inferred_tile="1m",
            confidence=0.7,
            reason="手牌変化から推測",
            prev_hand=["1m"] * 14,
            curr_hand=["1m"] * 13,
        )

        # プレースホルダー画像が作成されたか確認
        image_path = frame_manager.images_dir / f"{frame_id}.jpg"
        assert image_path.exists()

    def test_generate_review_html(self, frame_manager, sample_frame):
        """レビューHTML生成のテスト"""
        # いくつかのフレームを保存
        for i in range(3):
            frame_manager.save_inference_frame(
                frame=sample_frame,
                frame_number=i * 10,
                turn_number=i,
                player_index=i % 4,
                action_type="draw" if i % 2 == 0 else "discard",
                inferred_tile=f"{i + 1}m",
                confidence=0.8 - i * 0.1,
                reason="テスト推測",
                prev_hand=["1m"] * 13,
                curr_hand=["1m"] * 14 if i % 2 == 0 else ["1m"] * 13,
            )

        # HTML生成
        html_path = frame_manager.generate_review_html()
        assert Path(html_path).exists()

        # HTMLの内容を確認
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()
            assert "推測フレームレビュー" in html_content
            assert "合計フレーム数: 3" in html_content
            assert "未検証: 3" in html_content

    def test_update_frame_correction(self, frame_manager, sample_frame):
        """フレーム修正更新のテスト"""
        # フレームを保存
        frame_id = frame_manager.save_inference_frame(
            frame=sample_frame,
            frame_number=300,
            turn_number=3,
            player_index=2,
            action_type="call",
            inferred_tile=None,
            confidence=0.6,
            reason="3枚減少",
            prev_hand=["1z"] * 13,
            curr_hand=["1z"] * 10,
        )

        # 修正情報を更新
        correction = {"action_type": "pon", "tile": "1z", "comment": "1zのポン"}
        frame_manager.update_frame_correction(frame_id, correction, verified=True)

        # 更新が反映されたか確認
        frame_info = frame_manager.frames_index[0]
        assert frame_info.human_verified is True
        assert frame_info.human_correction == correction

    def test_export_corrections(self, frame_manager, sample_frame):
        """修正情報エクスポートのテスト"""
        # フレームを保存して修正
        frame_id1 = frame_manager.save_inference_frame(
            frame=sample_frame,
            frame_number=400,
            turn_number=4,
            player_index=3,
            action_type="draw",
            inferred_tile="9p",
            confidence=0.8,
            reason="推測",
            prev_hand=["1p"] * 13,
            curr_hand=["1p"] * 12 + ["9p"],
        )

        frame_manager.update_frame_correction(
            frame_id1, {"action_type": "draw", "tile": "9p", "comment": "正しい"}, verified=True
        )

        # エクスポート
        corrections = frame_manager.export_corrections()
        assert len(corrections) == 1
        assert frame_id1 in corrections
        assert corrections[frame_id1]["correction"]["comment"] == "正しい"

    def test_action_inferencer_with_frame_save(self, test_dir, sample_frame):
        """ActionInferencerのフレーム保存機能テスト"""
        inferencer = ActionInferencer(enable_frame_save=True)
        inferencer.frame_manager.base_dir = test_dir  # テスト用ディレクトリに変更

        # 巡0: プレイヤー0の手牌を記録（フレーム付き）
        hand0 = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"]
        inferencer.record_player_hand(0, hand0, 0, sample_frame, 1)

        # 巡1: プレイヤー0の手牌（変化あり）
        hand1 = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "5z"]
        inferencer.record_player_hand(0, hand1, 1, sample_frame, 5)

        # 推測が発生したか確認
        inferred = inferencer.get_inferred_actions()
        assert len(inferred) == 2  # ツモと捨て牌

        # フレームが保存されたか確認
        saved_frames = list(inferencer.frame_manager.images_dir.glob("*.jpg"))
        assert len(saved_frames) == 2  # ツモと捨て牌の2枚

    def test_simplified_detector_with_frame_save(self, test_dir, sample_frame):
        """SimplifiedActionDetectorのフレーム保存機能テスト"""
        config = {"enable_inference": True, "enable_frame_save": True}
        detector = SimplifiedActionDetector(config)
        detector.inferencer.frame_manager.base_dir = test_dir  # テスト用ディレクトリに変更

        # プレイヤー0の手牌（巡0）
        detector.detect_hand_change(
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "4z"],
            frame_number=1,
            frame=sample_frame,
        )

        # 他のプレイヤーの手牌で手番切り替え
        for i in range(3):
            detector.detect_hand_change([f"{i + 1}p"] * 13, frame_number=2 + i, frame=sample_frame)

        # プレイヤー0の手牌（巡1、変化あり）
        detector.detect_hand_change(
            ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z", "3z", "5z"],
            frame_number=5,
            frame=sample_frame,
        )

        # フレーム管理器を取得
        frame_manager = detector.get_frame_manager()
        assert frame_manager is not None

        # フレームが保存されたか確認
        saved_frames = list(frame_manager.images_dir.glob("*.jpg"))
        assert len(saved_frames) > 0

    def test_cleanup_old_sessions(self, frame_manager):
        """古いセッションのクリーンアップテスト"""
        # 古いセッションディレクトリを作成
        old_session_dir = frame_manager.base_dir / "20200101_000000"
        old_session_dir.mkdir(exist_ok=True)

        # クリーンアップ実行
        frame_manager.cleanup_old_sessions(days=0)  # 今日より古いものを削除

        # 古いセッションが削除されたか確認
        assert not old_session_dir.exists()
        assert frame_manager.session_dir.exists()  # 現在のセッションは残る

    def test_get_statistics(self, frame_manager, sample_frame):
        """統計情報取得のテスト"""
        # いくつかのフレームを保存
        actions = ["draw", "discard", "call", "draw", "discard"]
        for i, action in enumerate(actions):
            frame_manager.save_inference_frame(
                frame=sample_frame,
                frame_number=i * 10,
                turn_number=i,
                player_index=i % 4,
                action_type=action,
                inferred_tile=f"{i + 1}m" if action != "call" else None,
                confidence=0.8,
                reason="テスト",
                prev_hand=["1m"] * 13,
                curr_hand=["1m"] * 14,
            )

        # いくつかを検証済みにする
        for i in range(2):
            frame_id = frame_manager.frames_index[i].frame_id
            frame_manager.update_frame_correction(frame_id, {}, verified=True)

        # 統計情報を取得
        stats = frame_manager.get_statistics()
        assert stats["total_frames"] == 5
        assert stats["verified_frames"] == 2
        assert stats["action_type_distribution"]["draw"] == 2
        assert stats["action_type_distribution"]["discard"] == 2
        assert stats["action_type_distribution"]["call"] == 1
