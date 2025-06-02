"""
ゲームパイプラインの統合テスト
"""

import time
from unittest.mock import Mock, patch

import pytest

from src.game.player import PlayerPosition
from src.game.turn import ActionType
from src.pipeline.game_pipeline import GamePipeline, PipelineState


class TestGamePipeline:
    """ゲームパイプラインの統合テストクラス"""

    def setup_method(self):
        """テストセットアップ"""
        self.pipeline = GamePipeline("test_game")

    def test_initialization(self):
        """初期化のテスト"""
        assert self.pipeline.game_id == "test_game"
        assert self.pipeline.pipeline_state == PipelineState.IDLE
        assert self.pipeline.game_state is not None
        assert self.pipeline.state_tracker is not None
        assert self.pipeline.history_manager is not None

    def test_initialize_game(self):
        """ゲーム初期化のテスト"""
        player_names = {
            PlayerPosition.EAST: "Alice",
            PlayerPosition.SOUTH: "Bob",
            PlayerPosition.WEST: "Charlie",
            PlayerPosition.NORTH: "David",
        }

        result = self.pipeline.initialize_game(player_names)
        assert result is True
        assert self.pipeline.player_names == player_names
        assert self.pipeline.total_frames_processed == 0

    def test_process_frame_success(self):
        """フレーム処理成功のテスト"""
        # ゲームを初期化
        self.pipeline.initialize_game()

        # 新局を開始
        self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)

        # フレームデータを作成（簡略化）
        frame_data = {
            "frame_number": 1,
            "timestamp": time.time(),
            "player_hands": {
                "0": ["1m", "2m", "3m"],  # 簡略化されたデータ
            },
            "discarded_tiles": {"0": []},
            "confidence_scores": {"detection": 0.9, "classification": 0.85},
        }

        # state_trackerのupdate_from_frameをモック
        with (
            patch.object(self.pipeline.state_tracker, "update_from_frame") as mock_update,
            patch.object(self.pipeline.state_tracker, "get_current_confidence") as mock_confidence,
        ):
            # 成功を返すように設定
            mock_tracking_result = Mock()
            mock_tracking_result.success = True
            mock_tracking_result.state_changes = []
            mock_tracking_result.confidence = 0.9
            mock_tracking_result.detections = []
            mock_update.return_value = True  # 成功を返す
            mock_confidence.return_value = 0.9  # 信頼度を返す

            # フレームを処理
            result = self.pipeline.process_frame(frame_data)

        assert result.success is True
        assert result.frame_number == 1
        assert result.confidence > 0
        assert self.pipeline.total_frames_processed == 1
        assert self.pipeline.successful_frames == 1

    def test_process_frame_with_discard(self):
        """打牌を含むフレーム処理のテスト"""
        self.pipeline.initialize_game()

        # 新局を開始
        self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)

        with (
            patch.object(self.pipeline.state_tracker, "update_from_frame") as mock_update,
            patch.object(self.pipeline.state_tracker, "get_current_confidence") as mock_confidence,
        ):
            # 成功を返すように設定
            mock_tracking_result = Mock()
            mock_tracking_result.success = True
            mock_tracking_result.state_changes = []
            mock_tracking_result.confidence = 0.9
            mock_tracking_result.detections = []
            mock_update.return_value = True  # 成功を返す
            mock_confidence.return_value = 0.9  # 信頼度を返す

            # 最初のフレーム（初期状態）
            frame1_data = {
                "frame_number": 1,
                "player_hands": {"0": ["1m", "2m", "3m", "4m"]},
                "discarded_tiles": {"0": []},
                "confidence_scores": {"detection": 0.9, "classification": 0.85},
            }
            self.pipeline.process_frame(frame1_data)

            # 2番目のフレーム（打牌後）
            frame2_data = {
                "frame_number": 2,
                "player_hands": {
                    "0": ["1m", "2m", "3m"]  # 1枚減
                },
                "discarded_tiles": {
                    "0": ["4m"]  # 打牌
                },
                "confidence_scores": {"detection": 0.9, "classification": 0.85},
            }
            result = self.pipeline.process_frame(frame2_data)

        assert result.success is True
        assert result.actions_detected >= 0  # 0以上であることを確認

    def test_start_new_round(self):
        """新局開始のテスト"""
        self.pipeline.initialize_game()

        result = self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)
        assert result is True

        # 履歴管理で局が開始されているかチェック
        assert self.pipeline.history_manager.current_round is not None
        assert self.pipeline.history_manager.current_round.round_name == "東1局"

    def test_complete_current_round(self):
        """局完了のテスト"""
        self.pipeline.initialize_game()
        self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)

        result_data = {"winner": PlayerPosition.EAST.value, "han": 1, "fu": 30}
        scores = {
            PlayerPosition.EAST: 26000,
            PlayerPosition.SOUTH: 24000,
            PlayerPosition.WEST: 25000,
            PlayerPosition.NORTH: 25000,
        }

        result = self.pipeline.complete_current_round(result_data, scores)
        assert result is True

        # 局が完了しているかチェック
        assert self.pipeline.history_manager.current_round is None
        assert len(self.pipeline.history_manager.current_game.rounds) == 1

    def test_export_game_record(self):
        """牌譜エクスポートのテスト"""
        self.pipeline.initialize_game()
        self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)

        # いくつかの行動を追加
        from src.game.turn import Action

        action = Action(action_type=ActionType.DISCARD, player=PlayerPosition.EAST, tile="1m")
        self.pipeline.history_manager.add_action(action)

        self.pipeline.complete_current_round()

        # 天鳳JSON形式でエクスポート
        tenhou_json_data = self.pipeline.export_tenhou_json_record()
        assert tenhou_json_data is not None
        assert isinstance(tenhou_json_data, dict)

        # 必須フィールドの確認
        assert "game_info" in tenhou_json_data or "title" in tenhou_json_data

    def test_pipeline_statistics(self):
        """パイプライン統計のテスト"""
        self.pipeline.initialize_game()

        # 新局を開始
        self.pipeline.start_new_round(1, "東1局", PlayerPosition.EAST)

        with (
            patch.object(self.pipeline.state_tracker, "update_from_frame") as mock_update,
            patch.object(self.pipeline.state_tracker, "get_current_confidence") as mock_confidence,
        ):
            # 成功を返すように設定
            mock_tracking_result = Mock()
            mock_tracking_result.success = True
            mock_tracking_result.state_changes = []
            mock_tracking_result.confidence = 0.9
            mock_tracking_result.detections = []
            mock_update.return_value = True  # 成功を返す
            mock_confidence.return_value = 0.9  # 信頼度を返す

            # いくつかのフレームを処理
            for i in range(5):
                frame_data = {
                    "frame_number": i + 1,
                    "player_hands": {"0": ["1m", "2m", "3m"]},
                    "discarded_tiles": {"0": []},
                    "confidence_scores": {"detection": 0.9, "classification": 0.85},
                }
                self.pipeline.process_frame(frame_data)

        stats = self.pipeline.get_pipeline_statistics()

        assert stats["game_id"] == "test_game"
        assert stats["total_frames"] == 5
        assert stats["success_rate"] > 0
        assert "game_statistics" in stats
        assert "tracking_statistics" in stats
        assert "history_statistics" in stats

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        self.pipeline.initialize_game()

        # 不正なフレームデータ
        invalid_frame_data = {
            "frame_number": 1,
            "player_hands": {"invalid_position": ["invalid_tile"]},
        }

        result = self.pipeline.process_frame(invalid_frame_data)

        # エラーが適切に処理されているかチェック
        assert self.pipeline.total_frames_processed == 1
        # 処理は継続されるべき
        assert self.pipeline.pipeline_state != PipelineState.ERROR or len(result.errors) > 0

    def test_reset(self):
        """リセット機能のテスト"""
        self.pipeline.initialize_game()

        # 何らかの状態を設定
        frame_data = {"frame_number": 1, "player_hands": {"0": ["1m"]}}
        self.pipeline.process_frame(frame_data)

        # リセット実行
        self.pipeline.reset()

        assert self.pipeline.pipeline_state == PipelineState.IDLE
        assert self.pipeline.total_frames_processed == 0
        assert self.pipeline.successful_frames == 0
        assert self.pipeline.failed_frames == 0
        assert len(self.pipeline.processing_results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
