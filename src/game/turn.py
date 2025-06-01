"""
ターン管理クラス
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .player import PlayerPosition


class ActionType(Enum):
    """行動の種類"""

    DRAW = "draw"  # ツモ
    DISCARD = "discard"  # 打牌
    CHI = "chi"  # チー
    PON = "pon"  # ポン
    KAN = "kan"  # カン
    RIICHI = "riichi"  # リーチ
    TSUMO = "tsumo"  # ツモ和了
    RON = "ron"  # ロン和了
    KYUSHU = "kyushu"  # 九種九牌
    RYUKYOKU = "ryukyoku"  # 流局


@dataclass
class Action:
    """プレイヤーの行動情報"""

    action_type: ActionType
    player: PlayerPosition
    tile: str | None = None
    tiles: list[str] | None = None
    from_player: PlayerPosition | None = None
    timestamp: float = field(default_factory=time.time)
    frame_number: int | None = None
    confidence: float = 1.0
    detected_by: str = "manual"  # "ai", "manual", "rule_engine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後の処理"""
        if self.tiles is None:
            self.tiles = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TurnState:
    """ターンの状態情報"""

    turn_number: int
    current_player: PlayerPosition
    actions: list[Action] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    is_completed: bool = False

    def __post_init__(self):
        """初期化後の処理"""
        if not self.actions:
            self.actions = []


class Turn:
    """ターン管理クラス"""

    def __init__(self):
        """ターン管理クラスを初期化"""
        self.current_turn: TurnState | None = None
        self.turn_history: list[TurnState] = []
        self.turn_counter = 0

    def start_new_turn(self, player: PlayerPosition) -> TurnState:
        """
        新しいターンを開始

        Args:
            player: ターンを開始するプレイヤー

        Returns:
            TurnState: 新しいターンの状態
        """
        # 前のターンを完了
        if self.current_turn and not self.current_turn.is_completed:
            self.complete_current_turn()

        # 新しいターンを作成
        self.turn_counter += 1
        self.current_turn = TurnState(turn_number=self.turn_counter, current_player=player)

        return self.current_turn

    def add_action(self, action: Action) -> bool:
        """
        現在のターンに行動を追加

        Args:
            action: 追加する行動

        Returns:
            bool: 追加に成功したかどうか
        """
        if not self.current_turn:
            return False

        if self.current_turn.is_completed:
            return False

        self.current_turn.actions.append(action)

        # 特定の行動でターンが終了
        if action.action_type in [
            ActionType.DISCARD,
            ActionType.TSUMO,
            ActionType.RON,
            ActionType.KYUSHU,
            ActionType.RYUKYOKU,
        ]:
            self.complete_current_turn()

        return True

    def complete_current_turn(self):
        """現在のターンを完了"""
        if self.current_turn and not self.current_turn.is_completed:
            self.current_turn.end_time = time.time()
            self.current_turn.is_completed = True
            self.turn_history.append(self.current_turn)

    def get_current_turn(self) -> TurnState | None:
        """現在のターンを取得"""
        return self.current_turn

    def get_turn_history(self) -> list[TurnState]:
        """ターン履歴を取得"""
        return self.turn_history.copy()

    def get_last_action(self) -> Action | None:
        """最後の行動を取得"""
        if self.current_turn and self.current_turn.actions:
            return self.current_turn.actions[-1]
        elif self.turn_history:
            last_turn = self.turn_history[-1]
            if last_turn.actions:
                return last_turn.actions[-1]
        return None

    def get_actions_by_player(self, player: PlayerPosition) -> list[Action]:
        """指定プレイヤーの行動履歴を取得"""
        actions = []

        # 完了したターンから取得
        for turn in self.turn_history:
            for action in turn.actions:
                if action.player == player:
                    actions.append(action)

        # 現在のターンから取得
        if self.current_turn:
            for action in self.current_turn.actions:
                if action.player == player:
                    actions.append(action)

        return actions

    def get_actions_by_type(self, action_type: ActionType) -> list[Action]:
        """指定タイプの行動履歴を取得"""
        actions = []

        # 完了したターンから取得
        for turn in self.turn_history:
            for action in turn.actions:
                if action.action_type == action_type:
                    actions.append(action)

        # 現在のターンから取得
        if self.current_turn:
            for action in self.current_turn.actions:
                if action.action_type == action_type:
                    actions.append(action)

        return actions

    def get_recent_actions(self, count: int = 10) -> list[Action]:
        """最近の行動を取得"""
        all_actions = []

        # 完了したターンから取得
        for turn in self.turn_history:
            all_actions.extend(turn.actions)

        # 現在のターンから取得
        if self.current_turn:
            all_actions.extend(self.current_turn.actions)

        # 最新のものから指定数を返す
        return all_actions[-count:] if len(all_actions) > count else all_actions

    def find_action_by_frame(self, frame_number: int) -> Action | None:
        """フレーム番号で行動を検索"""
        # 完了したターンから検索
        for turn in self.turn_history:
            for action in turn.actions:
                if action.frame_number == frame_number:
                    return action

        # 現在のターンから検索
        if self.current_turn:
            for action in self.current_turn.actions:
                if action.frame_number == frame_number:
                    return action

        return None

    def get_turn_duration(self, turn_state: TurnState) -> float | None:
        """ターンの所要時間を取得"""
        if turn_state.end_time:
            return turn_state.end_time - turn_state.start_time
        elif turn_state == self.current_turn:
            return time.time() - turn_state.start_time
        return None

    def reset_for_new_round(self):
        """新しい局のためにリセット"""
        if self.current_turn and not self.current_turn.is_completed:
            self.complete_current_turn()

        self.current_turn = None
        self.turn_history = []
        self.turn_counter = 0

    def get_statistics(self) -> dict[str, Any]:
        """ターン統計を取得"""
        total_turns = len(self.turn_history)
        if self.current_turn:
            total_turns += 1

        action_counts = {}
        player_action_counts = dict.fromkeys(PlayerPosition, 0)

        all_actions = []
        for turn in self.turn_history:
            all_actions.extend(turn.actions)
        if self.current_turn:
            all_actions.extend(self.current_turn.actions)

        for action in all_actions:
            action_counts[action.action_type.value] = (
                action_counts.get(action.action_type.value, 0) + 1
            )
            player_action_counts[action.player] += 1

        return {
            "total_turns": total_turns,
            "total_actions": len(all_actions),
            "action_counts": action_counts,
            "player_action_counts": {
                pos.name: count for pos, count in player_action_counts.items()
            },
            "average_actions_per_turn": len(all_actions) / total_turns if total_turns > 0 else 0,
        }

    def __str__(self) -> str:
        """文字列表現"""
        current_info = ""
        if self.current_turn:
            current_info = f", Current: {self.current_turn.current_player.name}"
        return f"Turn(Total: {len(self.turn_history)}{current_info})"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"Turn(history_count={len(self.turn_history)}, "
            f"current_turn={self.current_turn is not None}, "
            f"turn_counter={self.turn_counter})"
        )
