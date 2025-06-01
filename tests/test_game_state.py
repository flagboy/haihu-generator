"""
ゲーム状態管理のテスト
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.game.game_state import GameState, GamePhase, FrameState
from src.game.player import PlayerPosition
from src.game.table import GameType
from src.game.turn import Action, ActionType


class TestGameState:
    """ゲーム状態管理のテストクラス"""
    
    def setup_method(self):
        """テストセットアップ"""
        self.game_state = GameState(GameType.HANCHAN)
    
    def test_initialization(self):
        """初期化のテスト"""
        assert self.game_state.phase == GamePhase.WAITING
        assert len(self.game_state.players) == 4
        assert self.game_state.table is not None
        assert self.game_state.turn_manager is not None
    
    def test_update_from_frame_detection(self):
        """フレーム検出結果からの更新テスト"""
        frame_data = {
            'frame_number': 1,
            'timestamp': time.time(),
            'player_hands': {
                '0': ['1m', '2m', '3m'],
                '1': ['4p', '5p', '6p']
            },
            'discarded_tiles': {
                '0': ['7s'],
                '1': []
            }
        }
        
        result = self.game_state.update_from_frame_detection(frame_data)
        assert result is True
        assert self.game_state.current_frame is not None
        assert len(self.game_state.frame_history) == 1
    
    def test_state_change_detection(self):
        """状態変化検出のテスト"""
        # 最初のフレーム
        frame1_data = {
            'frame_number': 1,
            'timestamp': time.time(),
            'player_hands': {
                '0': ['1m', '2m', '3m']
            }
        }
        self.game_state.update_from_frame_detection(frame1_data)
        
        # 2番目のフレーム（手牌変化）
        frame2_data = {
            'frame_number': 2,
            'timestamp': time.time(),
            'player_hands': {
                '0': ['1m', '2m', '3m', '4m']  # 牌が追加
            }
        }
        self.game_state.update_from_frame_detection(frame2_data)
        
        # 状態変化が検出されているかチェック
        assert len(self.game_state.state_changes) > 0
    
    def test_action_inference(self):
        """行動推定のテスト"""
        # 捨て牌が増加するフレーム
        frame1_data = {
            'frame_number': 1,
            'discarded_tiles': {'0': []}
        }
        self.game_state.update_from_frame_detection(frame1_data)
        
        frame2_data = {
            'frame_number': 2,
            'discarded_tiles': {'0': ['1m']}  # 捨て牌追加
        }
        self.game_state.update_from_frame_detection(frame2_data)
        
        # 打牌行動が推定されているかチェック
        assert len(self.game_state.pending_actions) > 0
        action = self.game_state.pending_actions[0]
        assert action.action_type == ActionType.DISCARD
        assert action.tile == '1m'
    
    def test_apply_pending_actions(self):
        """保留行動の適用テスト"""
        # プレイヤーに手牌を追加
        player = self.game_state.players[PlayerPosition.EAST]
        player.add_tile_to_hand('1m')
        
        # 打牌行動を作成
        action = Action(
            action_type=ActionType.DISCARD,
            player=PlayerPosition.EAST,
            tile='1m'
        )
        self.game_state.pending_actions.append(action)
        
        # 行動を適用
        applied_actions = self.game_state.apply_pending_actions()
        
        assert len(applied_actions) == 1
        assert not player.has_tile_in_hand('1m')
        assert '1m' in player.state.discarded_tiles
    
    def test_reset_for_new_round(self):
        """新局リセットのテスト"""
        # 何らかの状態を設定
        self.game_state.set_phase(GamePhase.PLAYING)
        self.game_state.frame_history.append(FrameState(1, time.time()))
        
        # リセット実行
        self.game_state.reset_for_new_round()
        
        assert self.game_state.phase == GamePhase.DEALING
        assert len(self.game_state.frame_history) == 0
        assert len(self.game_state.state_changes) == 0
        assert len(self.game_state.pending_actions) == 0
    
    def test_get_game_summary(self):
        """ゲーム概要取得のテスト"""
        summary = self.game_state.get_game_summary()
        
        assert 'phase' in summary
        assert 'round' in summary
        assert 'players' in summary
        assert 'turn_stats' in summary
        assert len(summary['players']) == 4


class TestFrameState:
    """フレーム状態のテストクラス"""
    
    def test_frame_state_creation(self):
        """フレーム状態作成のテスト"""
        frame_state = FrameState(
            frame_number=1,
            timestamp=time.time()
        )
        
        assert frame_state.frame_number == 1
        assert len(frame_state.player_hands) == 4
        assert len(frame_state.discarded_tiles) == 4
        assert isinstance(frame_state.table_info, dict)
        assert isinstance(frame_state.detected_actions, list)
        assert isinstance(frame_state.confidence_scores, dict)


if __name__ == '__main__':
    pytest.main([__file__])