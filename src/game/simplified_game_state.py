"""
シンプルなゲーム状態管理

手牌変化ベースのアクション検出に特化したシンプルなゲーム状態管理。
プレイヤー位置の推定や複雑な状態管理は行わない。
"""

from dataclasses import dataclass, field
from typing import Any

from ..utils.logger import LoggerMixin


@dataclass
class SimplifiedAction:
    """シンプルなアクション表現"""

    action_type: str  # "draw", "discard", "call", "riichi", "kan", etc.
    tile: str | None = None
    tiles: list[str] | None = None  # 鳴きの場合
    player_index: int = 0  # 0-3のプレイヤーインデックス
    frame_number: int | None = None
    timestamp: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


@dataclass
class SimplifiedGameState:
    """シンプルなゲーム状態"""

    actions: list[SimplifiedAction] = field(default_factory=list)
    current_player_index: int = 0
    turn_count: int = 0
    round_number: int = 0

    # 検出統計
    draw_count: int = 0
    discard_count: int = 0
    call_count: int = 0
    turn_change_count: int = 0


class SimplifiedGameStateManager(LoggerMixin):
    """シンプルなゲーム状態管理クラス"""

    def __init__(self):
        """初期化"""
        self.state = SimplifiedGameState()
        self.action_history: list[SimplifiedAction] = []

        self.logger.info("SimplifiedGameStateManager初期化完了")

    def add_action(
        self,
        action_type: str,
        tile: str | None = None,
        frame_number: int | None = None,
        confidence: float = 1.0,
    ) -> SimplifiedAction:
        """
        アクションを追加

        Args:
            action_type: アクションタイプ
            tile: 牌（オプション）
            frame_number: フレーム番号（オプション）
            confidence: 信頼度

        Returns:
            追加されたアクション
        """
        action = SimplifiedAction(
            action_type=action_type,
            tile=tile,
            player_index=self.state.current_player_index,
            frame_number=frame_number,
            confidence=confidence,
        )

        # アクションタイプに応じて処理
        if action_type == "turn_change":
            self._handle_turn_change()
        elif action_type == "draw":
            self.state.draw_count += 1
        elif action_type == "discard":
            self.state.discard_count += 1
            self.state.turn_count += 1
        elif action_type in ["pon", "chi", "kan"]:
            self.state.call_count += 1

        self.state.actions.append(action)
        self.action_history.append(action)

        return action

    def _handle_turn_change(self):
        """手番切り替えを処理"""
        self.state.current_player_index = (self.state.current_player_index + 1) % 4
        self.state.turn_change_count += 1

        # 4人分の手番が完了したら巡数を増加
        if self.state.current_player_index == 0:
            self.state.round_number += 1

    def get_current_player_actions(self) -> list[SimplifiedAction]:
        """現在のプレイヤーのアクションを取得"""
        return [
            action
            for action in self.state.actions
            if action.player_index == self.state.current_player_index
        ]

    def get_actions_by_type(self, action_type: str) -> list[SimplifiedAction]:
        """指定タイプのアクションを取得"""
        return [action for action in self.state.actions if action.action_type == action_type]

    def get_last_action(self) -> SimplifiedAction | None:
        """最後のアクションを取得"""
        return self.state.actions[-1] if self.state.actions else None

    def validate_action_sequence(self) -> list[str]:
        """
        アクションシーケンスの妥当性を検証

        Returns:
            エラーメッセージのリスト
        """
        errors = []

        # 基本的な検証
        draw_actions = self.get_actions_by_type("draw")
        discard_actions = self.get_actions_by_type("discard")

        # ツモと捨て牌の数が大きく異なる場合
        if abs(len(draw_actions) - len(discard_actions)) > 4:
            errors.append(f"ツモ({len(draw_actions)})と捨て牌({len(discard_actions)})の数が不均衡")

        # 連続した同じアクションをチェック
        for i in range(1, len(self.state.actions)):
            prev = self.state.actions[i - 1]
            curr = self.state.actions[i]

            # 同じプレイヤーが連続してツモ
            if (
                prev.action_type == "draw"
                and curr.action_type == "draw"
                and prev.player_index == curr.player_index
            ):
                errors.append(f"プレイヤー{prev.player_index}が連続してツモ")

        return errors

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        return {
            "total_actions": len(self.state.actions),
            "draw_count": self.state.draw_count,
            "discard_count": self.state.discard_count,
            "call_count": self.state.call_count,
            "turn_changes": self.state.turn_change_count,
            "current_player": self.state.current_player_index,
            "turn_count": self.state.turn_count,
            "round_number": self.state.round_number,
            "actions_per_player": self._get_actions_per_player(),
        }

    def _get_actions_per_player(self) -> dict[int, dict[str, int]]:
        """プレイヤーごとのアクション数を集計"""
        stats = {i: {"draw": 0, "discard": 0, "call": 0} for i in range(4)}

        for action in self.state.actions:
            if action.action_type == "draw":
                stats[action.player_index]["draw"] += 1
            elif action.action_type == "discard":
                stats[action.player_index]["discard"] += 1
            elif action.action_type in ["pon", "chi", "kan"]:
                stats[action.player_index]["call"] += 1

        return stats

    def export_to_tenhou_format(self) -> list[dict[str, Any]]:
        """
        天鳳形式にエクスポート

        Returns:
            天鳳形式のアクションリスト
        """
        tenhou_actions = []

        for action in self.state.actions:
            if action.action_type == "turn_change":
                continue

            tenhou_action = {
                "type": action.action_type,
                "player": action.player_index,
            }

            if action.tile:
                tenhou_action["tile"] = action.tile
            if action.tiles:
                tenhou_action["tiles"] = action.tiles
            if action.frame_number:
                tenhou_action["frame"] = action.frame_number

            tenhou_actions.append(tenhou_action)

        return tenhou_actions

    def reset(self):
        """状態をリセット"""
        self.state = SimplifiedGameState()
        self.action_history = []
        self.logger.info("ゲーム状態をリセットしました")
