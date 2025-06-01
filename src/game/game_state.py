"""
ゲーム状態管理クラス（動画解析結果統合）
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
import time

from .player import Player, PlayerPosition, PlayerState
from .table import Table, TableState, Wind, GameType
from .turn import Turn, TurnState, Action, ActionType
from ..utils.tile_definitions import TileDefinitions


class GamePhase(Enum):
    """ゲームフェーズ"""
    WAITING = "waiting"         # 待機中
    DEALING = "dealing"         # 配牌中
    PLAYING = "playing"         # 対局中
    FINISHED = "finished"       # 終了
    PAUSED = "paused"          # 一時停止


class DetectionSource(Enum):
    """検出ソース"""
    AI_DETECTION = "ai_detection"       # AI検出
    AI_CLASSIFICATION = "ai_classification"  # AI分類
    RULE_ENGINE = "rule_engine"         # ルールエンジン
    MANUAL = "manual"                   # 手動
    INTERPOLATION = "interpolation"     # 補間


@dataclass
class FrameState:
    """フレーム状態情報"""
    frame_number: int
    timestamp: float
    player_hands: Dict[PlayerPosition, List[str]] = field(default_factory=dict)
    discarded_tiles: Dict[PlayerPosition, List[str]] = field(default_factory=dict)
    table_info: Dict[str, Any] = field(default_factory=dict)
    detected_actions: List[Action] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後の処理"""
        if not self.player_hands:
            self.player_hands = {pos: [] for pos in PlayerPosition}
        if not self.discarded_tiles:
            self.discarded_tiles = {pos: [] for pos in PlayerPosition}
        if not self.table_info:
            self.table_info = {}
        if not self.detected_actions:
            self.detected_actions = []
        if not self.confidence_scores:
            self.confidence_scores = {}


class GameState:
    """ゲーム状態管理クラス（動画解析結果統合）"""
    
    def __init__(self, game_type: GameType = GameType.HANCHAN):
        """
        ゲーム状態管理クラスを初期化
        
        Args:
            game_type: ゲームタイプ
        """
        self.tile_definitions = TileDefinitions()
        
        # 基本コンポーネント
        self.players = {
            pos: Player(pos, f"Player{pos.value + 1}")
            for pos in PlayerPosition
        }
        self.table = Table(game_type)
        self.turn_manager = Turn()
        
        # ゲーム状態
        self.phase = GamePhase.WAITING
        self.current_frame: Optional[FrameState] = None
        self.frame_history: List[FrameState] = []
        
        # 検出・追跡情報
        self.last_detection_time = 0.0
        self.confidence_threshold = 0.7
        self.inconsistency_count = 0
        self.max_inconsistencies = 5
        
        # 状態変化追跡
        self.state_changes: List[Dict[str, Any]] = []
        self.pending_actions: List[Action] = []
    
    def update_from_frame_detection(self, frame_data: Dict[str, Any]) -> bool:
        """
        フレーム検出結果からゲーム状態を更新
        
        Args:
            frame_data: フレーム検出結果
            
        Returns:
            bool: 更新に成功したかどうか
        """
        frame_number = frame_data.get('frame_number', 0)
        timestamp = frame_data.get('timestamp', time.time())
        
        # 新しいフレーム状態を作成
        frame_state = FrameState(
            frame_number=frame_number,
            timestamp=timestamp
        )
        
        # 検出結果を解析
        success = self._process_detection_results(frame_data, frame_state)
        
        if success:
            # フレーム履歴に追加
            self.current_frame = frame_state
            self.frame_history.append(frame_state)
            
            # 状態変化を検出
            self._detect_state_changes(frame_state)
            
            # 行動を推定
            self._infer_actions(frame_state)
            
            self.last_detection_time = timestamp
        
        return success
    
    def _process_detection_results(self, frame_data: Dict[str, Any], frame_state: FrameState) -> bool:
        """
        検出結果を処理
        
        Args:
            frame_data: フレーム検出結果
            frame_state: フレーム状態
            
        Returns:
            bool: 処理に成功したかどうか
        """
        try:
            # プレイヤー手牌の更新
            if 'player_hands' in frame_data:
                for pos_str, tiles in frame_data['player_hands'].items():
                    try:
                        pos = PlayerPosition(int(pos_str))
                        frame_state.player_hands[pos] = tiles
                    except (ValueError, KeyError):
                        continue
            
            # 捨て牌の更新
            if 'discarded_tiles' in frame_data:
                for pos_str, tiles in frame_data['discarded_tiles'].items():
                    try:
                        pos = PlayerPosition(int(pos_str))
                        frame_state.discarded_tiles[pos] = tiles
                    except (ValueError, KeyError):
                        continue
            
            # テーブル情報の更新
            if 'table_info' in frame_data:
                frame_state.table_info = frame_data['table_info']
            
            # 信頼度スコアの更新
            if 'confidence_scores' in frame_data:
                frame_state.confidence_scores = frame_data['confidence_scores']
            
            return True
            
        except Exception as e:
            print(f"Error processing detection results: {e}")
            return False
    
    def _detect_state_changes(self, frame_state: FrameState):
        """
        状態変化を検出
        
        Args:
            frame_state: 現在のフレーム状態
        """
        if not self.frame_history or len(self.frame_history) < 2:
            return
        
        previous_frame = self.frame_history[-2]
        changes = []
        
        # 手牌の変化を検出
        for pos in PlayerPosition:
            prev_hand = previous_frame.player_hands.get(pos, [])
            curr_hand = frame_state.player_hands.get(pos, [])
            
            if prev_hand != curr_hand:
                changes.append({
                    'type': 'hand_change',
                    'player': pos,
                    'previous': prev_hand,
                    'current': curr_hand,
                    'frame': frame_state.frame_number
                })
        
        # 捨て牌の変化を検出
        for pos in PlayerPosition:
            prev_discards = previous_frame.discarded_tiles.get(pos, [])
            curr_discards = frame_state.discarded_tiles.get(pos, [])
            
            if len(curr_discards) > len(prev_discards):
                new_discards = curr_discards[len(prev_discards):]
                changes.append({
                    'type': 'discard_change',
                    'player': pos,
                    'new_discards': new_discards,
                    'frame': frame_state.frame_number
                })
        
        self.state_changes.extend(changes)
    
    def _infer_actions(self, frame_state: FrameState):
        """
        フレーム状態から行動を推定
        
        Args:
            frame_state: フレーム状態
        """
        # 最近の状態変化から行動を推定
        recent_changes = [
            change for change in self.state_changes
            if change.get('frame', 0) == frame_state.frame_number
        ]
        
        for change in recent_changes:
            if change['type'] == 'discard_change':
                # 打牌行動を推定
                player = change['player']
                for tile in change['new_discards']:
                    action = Action(
                        action_type=ActionType.DISCARD,
                        player=player,
                        tile=tile,
                        frame_number=frame_state.frame_number,
                        timestamp=frame_state.timestamp,
                        detected_by="state_inference"
                    )
                    frame_state.detected_actions.append(action)
                    self.pending_actions.append(action)
    
    def apply_pending_actions(self) -> List[Action]:
        """
        保留中の行動を適用
        
        Returns:
            List[Action]: 適用された行動のリスト
        """
        applied_actions = []
        
        for action in self.pending_actions:
            if self._validate_action(action):
                success = self._apply_action(action)
                if success:
                    applied_actions.append(action)
                    self.turn_manager.add_action(action)
        
        # 適用された行動を保留リストから削除
        for action in applied_actions:
            if action in self.pending_actions:
                self.pending_actions.remove(action)
        
        return applied_actions
    
    def _validate_action(self, action: Action) -> bool:
        """
        行動の妥当性を検証
        
        Args:
            action: 検証する行動
            
        Returns:
            bool: 妥当かどうか
        """
        player = self.players[action.player]
        
        if action.action_type == ActionType.DISCARD:
            # 打牌の場合、手牌にその牌があるかチェック
            return action.tile and player.has_tile_in_hand(action.tile)
        
        elif action.action_type == ActionType.DRAW:
            # ツモの場合、手牌枚数をチェック
            return player.get_hand_size() < 14
        
        # その他の行動は基本的に有効とする
        return True
    
    def _apply_action(self, action: Action) -> bool:
        """
        行動を実際のゲーム状態に適用
        
        Args:
            action: 適用する行動
            
        Returns:
            bool: 適用に成功したかどうか
        """
        player = self.players[action.player]
        
        try:
            if action.action_type == ActionType.DISCARD:
                return player.discard_tile(action.tile)
            
            elif action.action_type == ActionType.DRAW:
                return player.add_tile_to_hand(action.tile)
            
            elif action.action_type == ActionType.RIICHI:
                current_turn = self.turn_manager.get_current_turn()
                turn_number = current_turn.turn_number if current_turn else 0
                return player.declare_riichi(turn_number)
            
            # その他の行動は今後実装
            return True
            
        except Exception as e:
            print(f"Error applying action: {e}")
            return False
    
    def get_current_player_states(self) -> Dict[PlayerPosition, PlayerState]:
        """現在のプレイヤー状態を取得"""
        return {pos: player.get_state_copy() for pos, player in self.players.items()}
    
    def get_current_table_state(self) -> TableState:
        """現在のテーブル状態を取得"""
        return self.table.get_state_copy()
    
    def get_game_summary(self) -> Dict[str, Any]:
        """ゲーム概要を取得"""
        return {
            'phase': self.phase.value,
            'round': self.table.get_current_round_name(),
            'players': {
                pos.name: {
                    'name': player.state.name,
                    'score': player.state.score,
                    'hand_size': player.get_hand_size(),
                    'riichi': player.state.riichi
                }
                for pos, player in self.players.items()
            },
            'turn_stats': self.turn_manager.get_statistics(),
            'frame_count': len(self.frame_history),
            'last_detection': self.last_detection_time
        }
    
    def reset_for_new_round(self):
        """新しい局のためにリセット"""
        for player in self.players.values():
            player.reset_for_new_round()
        
        self.table.reset_for_new_round()
        self.turn_manager.reset_for_new_round()
        
        self.phase = GamePhase.DEALING
        self.current_frame = None
        self.frame_history = []
        self.state_changes = []
        self.pending_actions = []
        self.inconsistency_count = 0
    
    def set_phase(self, phase: GamePhase):
        """ゲームフェーズを設定"""
        self.phase = phase
    
    def get_detection_confidence(self) -> float:
        """現在の検出信頼度を取得"""
        if not self.current_frame or not self.current_frame.confidence_scores:
            return 0.0
        
        scores = list(self.current_frame.confidence_scores.values())
        return sum(scores) / len(scores) if scores else 0.0
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"GameState({self.phase.value}, {self.table.get_current_round_name()})"
    
    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (f"GameState(phase={self.phase.value}, "
                f"round={self.table.get_current_round_name()}, "
                f"frames={len(self.frame_history)}, "
                f"pending_actions={len(self.pending_actions)})")