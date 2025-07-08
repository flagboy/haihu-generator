"""
アクション推測モジュール

カメラ切り替えなどで欠落したアクションを、次巡の手牌から推測する。
"""

from dataclasses import dataclass
from typing import Any

from ..utils.logger import LoggerMixin


@dataclass
class InferredAction:
    """推測されたアクション"""

    action_type: str  # "draw", "discard"
    tile: str | None = None
    confidence: float = 0.0
    reason: str = ""  # 推測理由
    metadata: dict[str, Any] | None = None


class ActionInferencer(LoggerMixin):
    """アクション推測クラス"""

    def __init__(self):
        """初期化"""
        self.player_hands_history: list[dict[int, list[str]]] = []  # 各巡の4人の手牌
        self.inferred_actions: list[InferredAction] = []

        self.logger.info("ActionInferencer初期化完了")

    def record_player_hand(self, player_index: int, hand: list[str], turn_number: int):
        """
        プレイヤーの手牌を記録

        Args:
            player_index: プレイヤー番号（0-3）
            hand: 手牌
            turn_number: 巡目
        """
        # 必要に応じて履歴を拡張
        while len(self.player_hands_history) <= turn_number:
            self.player_hands_history.append({})

        self.player_hands_history[turn_number][player_index] = hand.copy()

        # 前巡との比較で欠落アクションを推測
        if turn_number >= 1:  # 巡1から推測開始
            self._infer_missing_actions(player_index, turn_number)

    def _infer_missing_actions(self, player_index: int, current_turn: int):
        """
        欠落したアクションを推測

        Args:
            player_index: 現在のプレイヤー番号
            current_turn: 現在の巡目
        """
        previous_turn = current_turn - 1  # 前巡

        self.logger.debug(
            f"推測チェック: player={player_index}, current_turn={current_turn}, previous_turn={previous_turn}"
        )

        if previous_turn < 0 or current_turn == 0:
            self.logger.debug("前巡がないためスキップ")
            return

        # 前巡の同じプレイヤーの手牌を取得
        if previous_turn >= len(self.player_hands_history):
            self.logger.debug("履歴範囲外のためスキップ")
            return

        previous_hands = self.player_hands_history[previous_turn]
        current_hands = self.player_hands_history[current_turn]

        self.logger.debug(f"前巡のプレイヤー: {list(previous_hands.keys())}")
        self.logger.debug(f"現巡のプレイヤー: {list(current_hands.keys())}")

        if player_index not in previous_hands or player_index not in current_hands:
            self.logger.debug(f"プレイヤー{player_index}のデータが不足")
            return

        prev_hand = previous_hands[player_index]
        curr_hand = current_hands[player_index]

        self.logger.debug(f"前巡の手牌: {prev_hand}")
        self.logger.debug(f"現巡の手牌: {curr_hand}")

        # 手牌の変化を分析
        self._analyze_hand_transition(
            player_index, prev_hand, curr_hand, previous_turn, current_turn
        )

    def _analyze_hand_transition(
        self,
        player_index: int,
        prev_hand: list[str],
        curr_hand: list[str],
        prev_turn: int,
        curr_turn: int,
    ):
        """
        手牌の変化を分析してアクションを推測

        Args:
            player_index: プレイヤー番号
            prev_hand: 前巡の手牌
            curr_hand: 現在の手牌
            prev_turn: 前巡の番号
            curr_turn: 現在の巡番号
        """
        prev_tiles = self._count_tiles(prev_hand)
        curr_tiles = self._count_tiles(curr_hand)

        # 増えた牌と減った牌を特定
        added_tiles = []
        removed_tiles = []

        all_tiles = set(prev_tiles.keys()) | set(curr_tiles.keys())

        for tile in all_tiles:
            prev_count = prev_tiles.get(tile, 0)
            curr_count = curr_tiles.get(tile, 0)

            if curr_count > prev_count:
                for _ in range(curr_count - prev_count):
                    added_tiles.append(tile)
            elif prev_count > curr_count:
                for _ in range(prev_count - curr_count):
                    removed_tiles.append(tile)

        # パターン分析
        if len(added_tiles) == 1 and len(removed_tiles) == 1:
            # 通常のツモ切り
            self._infer_draw_discard(
                player_index, added_tiles[0], removed_tiles[0], prev_turn, curr_turn
            )
        elif len(added_tiles) == 0 and len(removed_tiles) == 0:
            # 手牌に変化なし（リーチ後のツモ切りなど）
            self._infer_unchanged_hand(player_index, prev_turn, curr_turn)
        elif len(removed_tiles) > len(added_tiles):
            # 鳴きの可能性
            self._infer_call_action(player_index, added_tiles, removed_tiles, prev_turn, curr_turn)
        else:
            # その他の複雑なケース
            self.logger.debug(
                f"複雑な手牌変化: プレイヤー{player_index}, "
                f"追加: {added_tiles}, 削除: {removed_tiles}"
            )

    def _infer_draw_discard(
        self,
        player_index: int,
        drawn_tile: str,
        discarded_tile: str,
        prev_turn: int,
        curr_turn: int,
    ):
        """通常のツモ切りを推測"""
        # ツモアクション
        draw_action = InferredAction(
            action_type="draw",
            tile=drawn_tile,
            confidence=0.8,
            reason=f"次巡の手牌から推測（巡{prev_turn}→{curr_turn}）",
            metadata={
                "player_index": player_index,
                "turn": prev_turn,
                "inferred_from_turn": curr_turn,
            },
        )
        self.inferred_actions.append(draw_action)

        # 捨て牌アクション
        discard_action = InferredAction(
            action_type="discard",
            tile=discarded_tile,
            confidence=0.8,
            reason=f"次巡の手牌から推測（巡{prev_turn}→{curr_turn}）",
            metadata={
                "player_index": player_index,
                "turn": prev_turn,
                "inferred_from_turn": curr_turn,
            },
        )
        self.inferred_actions.append(discard_action)

        self.logger.info(
            f"アクション推測: プレイヤー{player_index} ツモ{drawn_tile}→切り{discarded_tile}"
        )

    def _infer_unchanged_hand(self, player_index: int, prev_turn: int, curr_turn: int):
        """手牌変化なしの場合の推測（リーチ後のツモ切りなど）"""
        # ツモ切りを推測（具体的な牌は不明）
        action = InferredAction(
            action_type="tsumo_giri",
            tile=None,
            confidence=0.6,
            reason="手牌変化なし（リーチ後のツモ切りの可能性）",
            metadata={
                "player_index": player_index,
                "turn": prev_turn,
                "inferred_from_turn": curr_turn,
            },
        )
        self.inferred_actions.append(action)

        self.logger.info(f"ツモ切り推測: プレイヤー{player_index}（手牌変化なし）")

    def _infer_call_action(
        self,
        player_index: int,
        added_tiles: list[str],
        removed_tiles: list[str],
        prev_turn: int,
        curr_turn: int,
    ):
        """鳴きアクションを推測"""
        removed_count = len(removed_tiles)

        if removed_count == 3:
            call_type = "pon_or_chi"
        elif removed_count == 4:
            call_type = "kan"
        else:
            call_type = "unknown_call"

        action = InferredAction(
            action_type="call",
            tile=None,
            confidence=0.7,
            reason=f"{removed_count}枚減少（{call_type}の可能性）",
            metadata={
                "player_index": player_index,
                "turn": prev_turn,
                "inferred_from_turn": curr_turn,
                "call_type": call_type,
                "removed_tiles": removed_tiles,
                "added_tiles": added_tiles,
            },
        )
        self.inferred_actions.append(action)

        self.logger.info(f"鳴き推測: プレイヤー{player_index} {call_type}")

    def _count_tiles(self, tiles: list[str]) -> dict[str, int]:
        """牌の枚数をカウント"""
        counts = {}
        for tile in tiles:
            counts[tile] = counts.get(tile, 0) + 1
        return counts

    def get_inferred_actions(self, player_index: int | None = None) -> list[InferredAction]:
        """
        推測されたアクションを取得

        Args:
            player_index: 特定のプレイヤーのみ取得する場合

        Returns:
            推測されたアクションのリスト
        """
        if player_index is None:
            return self.inferred_actions.copy()

        return [
            action
            for action in self.inferred_actions
            if action.metadata and action.metadata.get("player_index") == player_index
        ]

    def clear_history(self):
        """履歴をクリア"""
        self.player_hands_history.clear()
        self.inferred_actions.clear()
        self.logger.info("推測履歴をクリアしました")
