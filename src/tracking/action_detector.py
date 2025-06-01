"""
行動検出クラス
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..game.turn import Action, ActionType
from ..game.player import PlayerPosition
from ..utils.tile_definitions import TileDefinitions


class DetectionConfidence(Enum):
    """検出信頼度レベル"""
    HIGH = "high"       # 高信頼度 (>0.8)
    MEDIUM = "medium"   # 中信頼度 (0.5-0.8)
    LOW = "low"         # 低信頼度 (<0.5)


@dataclass
class TileChange:
    """牌の変化情報"""
    player: PlayerPosition
    tile: str
    change_type: str  # "added", "removed", "moved"
    location: str     # "hand", "discard", "call"
    confidence: float
    frame_number: int


@dataclass
class DetectionResult:
    """検出結果"""
    actions: List[Action]
    tile_changes: List[TileChange]
    confidence_level: DetectionConfidence
    metadata: Dict[str, Any]


class ActionDetector:
    """プレイヤーの行動検出クラス"""
    
    def __init__(self):
        """行動検出クラスを初期化"""
        self.tile_definitions = TileDefinitions()
        self.previous_frame_data: Optional[Dict[str, Any]] = None
        self.detection_history: List[DetectionResult] = []
        
        # 検出パラメータ
        self.min_confidence = 0.3
        self.hand_size_tolerance = 1  # 手牌枚数の許容誤差
        self.discard_detection_threshold = 0.7
    
    def detect_actions(self, current_frame: Dict[str, Any], 
                      frame_number: int) -> DetectionResult:
        """
        フレーム間の変化から行動を検出
        
        Args:
            current_frame: 現在のフレームデータ
            frame_number: フレーム番号
            
        Returns:
            DetectionResult: 検出結果
        """
        actions = []
        tile_changes = []
        
        if self.previous_frame_data is None:
            # 初回フレームの場合は配牌として処理
            actions.extend(self._detect_initial_deal(current_frame, frame_number))
        else:
            # フレーム間の変化を解析
            tile_changes = self._analyze_tile_changes(
                self.previous_frame_data, current_frame, frame_number
            )
            
            # 変化から行動を推定
            actions = self._infer_actions_from_changes(tile_changes, frame_number)
        
        # 信頼度を計算
        confidence_level = self._calculate_confidence_level(actions, tile_changes)
        
        # 結果を作成
        result = DetectionResult(
            actions=actions,
            tile_changes=tile_changes,
            confidence_level=confidence_level,
            metadata={
                'frame_number': frame_number,
                'detection_count': len(actions),
                'change_count': len(tile_changes)
            }
        )
        
        # 履歴に追加
        self.detection_history.append(result)
        self.previous_frame_data = current_frame.copy()
        
        return result
    
    def _detect_initial_deal(self, frame_data: Dict[str, Any], 
                           frame_number: int) -> List[Action]:
        """
        初期配牌を検出
        
        Args:
            frame_data: フレームデータ
            frame_number: フレーム番号
            
        Returns:
            List[Action]: 検出された行動
        """
        actions = []
        
        # 各プレイヤーの手牌をチェック
        player_hands = frame_data.get('player_hands', {})
        
        for pos_str, tiles in player_hands.items():
            try:
                pos = PlayerPosition(int(pos_str))
                
                # 配牌として各牌を追加
                for tile in tiles:
                    if self.tile_definitions.is_valid_tile(tile):
                        action = Action(
                            action_type=ActionType.DRAW,
                            player=pos,
                            tile=tile,
                            frame_number=frame_number,
                            detected_by="initial_deal",
                            confidence=0.9
                        )
                        actions.append(action)
            except (ValueError, KeyError):
                continue
        
        return actions
    
    def _analyze_tile_changes(self, previous_frame: Dict[str, Any],
                            current_frame: Dict[str, Any],
                            frame_number: int) -> List[TileChange]:
        """
        フレーム間の牌の変化を解析
        
        Args:
            previous_frame: 前フレームデータ
            current_frame: 現フレームデータ
            frame_number: フレーム番号
            
        Returns:
            List[TileChange]: 牌の変化リスト
        """
        changes = []
        
        # 手牌の変化を検出
        changes.extend(self._detect_hand_changes(
            previous_frame, current_frame, frame_number
        ))
        
        # 捨て牌の変化を検出
        changes.extend(self._detect_discard_changes(
            previous_frame, current_frame, frame_number
        ))
        
        # 鳴きの変化を検出
        changes.extend(self._detect_call_changes(
            previous_frame, current_frame, frame_number
        ))
        
        return changes
    
    def _detect_hand_changes(self, previous_frame: Dict[str, Any],
                           current_frame: Dict[str, Any],
                           frame_number: int) -> List[TileChange]:
        """手牌の変化を検出"""
        changes = []
        
        prev_hands = previous_frame.get('player_hands', {})
        curr_hands = current_frame.get('player_hands', {})
        
        for pos_str in set(prev_hands.keys()) | set(curr_hands.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_hands.get(pos_str, [])
                curr_tiles = curr_hands.get(pos_str, [])
                
                # 追加された牌
                added_tiles = [tile for tile in curr_tiles if tile not in prev_tiles]
                for tile in added_tiles:
                    changes.append(TileChange(
                        player=pos,
                        tile=tile,
                        change_type="added",
                        location="hand",
                        confidence=0.8,
                        frame_number=frame_number
                    ))
                
                # 削除された牌
                removed_tiles = [tile for tile in prev_tiles if tile not in curr_tiles]
                for tile in removed_tiles:
                    changes.append(TileChange(
                        player=pos,
                        tile=tile,
                        change_type="removed",
                        location="hand",
                        confidence=0.8,
                        frame_number=frame_number
                    ))
                    
            except (ValueError, KeyError):
                continue
        
        return changes
    
    def _detect_discard_changes(self, previous_frame: Dict[str, Any],
                              current_frame: Dict[str, Any],
                              frame_number: int) -> List[TileChange]:
        """捨て牌の変化を検出"""
        changes = []
        
        prev_discards = previous_frame.get('discarded_tiles', {})
        curr_discards = current_frame.get('discarded_tiles', {})
        
        for pos_str in set(prev_discards.keys()) | set(curr_discards.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_discards.get(pos_str, [])
                curr_tiles = curr_discards.get(pos_str, [])
                
                # 新しく捨てられた牌
                if len(curr_tiles) > len(prev_tiles):
                    new_discards = curr_tiles[len(prev_tiles):]
                    for tile in new_discards:
                        changes.append(TileChange(
                            player=pos,
                            tile=tile,
                            change_type="added",
                            location="discard",
                            confidence=0.9,
                            frame_number=frame_number
                        ))
                        
            except (ValueError, KeyError):
                continue
        
        return changes
    
    def _detect_call_changes(self, previous_frame: Dict[str, Any],
                           current_frame: Dict[str, Any],
                           frame_number: int) -> List[TileChange]:
        """鳴きの変化を検出"""
        changes = []
        
        # 鳴き情報の変化を検出（実装は簡略化）
        prev_calls = previous_frame.get('calls', {})
        curr_calls = current_frame.get('calls', {})
        
        for pos_str in set(curr_calls.keys()) - set(prev_calls.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                call_tiles = curr_calls.get(pos_str, [])
                
                for tile in call_tiles:
                    changes.append(TileChange(
                        player=pos,
                        tile=tile,
                        change_type="added",
                        location="call",
                        confidence=0.7,
                        frame_number=frame_number
                    ))
                    
            except (ValueError, KeyError):
                continue
        
        return changes
    
    def _infer_actions_from_changes(self, tile_changes: List[TileChange],
                                  frame_number: int) -> List[Action]:
        """牌の変化から行動を推定"""
        actions = []
        
        # プレイヤーごとに変化を整理
        player_changes = {}
        for change in tile_changes:
            if change.player not in player_changes:
                player_changes[change.player] = []
            player_changes[change.player].append(change)
        
        # 各プレイヤーの行動を推定
        for player, changes in player_changes.items():
            actions.extend(self._infer_player_actions(player, changes, frame_number))
        
        return actions
    
    def _infer_player_actions(self, player: PlayerPosition, 
                            changes: List[TileChange],
                            frame_number: int) -> List[Action]:
        """特定プレイヤーの行動を推定"""
        actions = []
        
        hand_added = [c for c in changes if c.location == "hand" and c.change_type == "added"]
        hand_removed = [c for c in changes if c.location == "hand" and c.change_type == "removed"]
        discard_added = [c for c in changes if c.location == "discard" and c.change_type == "added"]
        call_added = [c for c in changes if c.location == "call" and c.change_type == "added"]
        
        # ツモ行動の検出
        for change in hand_added:
            action = Action(
                action_type=ActionType.DRAW,
                player=player,
                tile=change.tile,
                frame_number=frame_number,
                confidence=change.confidence,
                detected_by="change_analysis"
            )
            actions.append(action)
        
        # 打牌行動の検出
        for change in discard_added:
            action = Action(
                action_type=ActionType.DISCARD,
                player=player,
                tile=change.tile,
                frame_number=frame_number,
                confidence=change.confidence,
                detected_by="change_analysis"
            )
            actions.append(action)
        
        # 鳴き行動の検出
        if call_added:
            call_tiles = [c.tile for c in call_added]
            # 鳴きの種類を推定（簡略化）
            action_type = self._infer_call_type(call_tiles)
            
            action = Action(
                action_type=action_type,
                player=player,
                tiles=call_tiles,
                frame_number=frame_number,
                confidence=min(c.confidence for c in call_added),
                detected_by="change_analysis"
            )
            actions.append(action)
        
        return actions
    
    def _infer_call_type(self, tiles: List[str]) -> ActionType:
        """鳴きの種類を推定"""
        if len(tiles) == 3:
            # 3枚の場合はチーまたはポン
            if len(set(tiles)) == 1:
                return ActionType.PON
            else:
                return ActionType.CHI
        elif len(tiles) == 4:
            return ActionType.KAN
        else:
            return ActionType.CHI  # デフォルト
    
    def _calculate_confidence_level(self, actions: List[Action],
                                  tile_changes: List[TileChange]) -> DetectionConfidence:
        """検出信頼度レベルを計算"""
        if not actions and not tile_changes:
            return DetectionConfidence.HIGH
        
        # 行動の平均信頼度を計算
        action_confidences = [action.confidence for action in actions if action.confidence > 0]
        change_confidences = [change.confidence for change in tile_changes]
        
        all_confidences = action_confidences + change_confidences
        
        if not all_confidences:
            return DetectionConfidence.LOW
        
        avg_confidence = sum(all_confidences) / len(all_confidences)
        
        if avg_confidence > 0.8:
            return DetectionConfidence.HIGH
        elif avg_confidence > 0.5:
            return DetectionConfidence.MEDIUM
        else:
            return DetectionConfidence.LOW
    
    def get_recent_detections(self, count: int = 10) -> List[DetectionResult]:
        """最近の検出結果を取得"""
        return self.detection_history[-count:] if len(self.detection_history) > count else self.detection_history
    
    def reset(self):
        """検出器をリセット"""
        self.previous_frame_data = None
        self.detection_history = []
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"ActionDetector(history: {len(self.detection_history)})"
    
    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (f"ActionDetector(detection_count={len(self.detection_history)}, "
                f"min_confidence={self.min_confidence})")