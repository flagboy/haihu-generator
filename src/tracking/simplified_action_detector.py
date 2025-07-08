"""
シンプルな手牌変化ベースのアクション検出器

手番プレイヤーの手牌変化のみを追跡してアクションを検出する。
プレイヤー位置の推定や他家の状態管理は行わない。
"""

from dataclasses import dataclass
from typing import Any

from ..utils.logger import LoggerMixin
from .action_inferencer import ActionInferencer


@dataclass
class HandChangeResult:
    """手牌変化の検出結果"""

    action_type: str  # "draw", "discard", "turn_change", "call", "unknown"
    tile: str | None = None  # 変化した牌
    confidence: float = 0.0
    hand_before: list[str] | None = None
    hand_after: list[str] | None = None
    metadata: dict[str, Any] | None = None


class SimplifiedActionDetector(LoggerMixin):
    """シンプルな手牌変化ベースのアクション検出器"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        初期化

        Args:
            config: 設定（オプション）
        """
        self.config = config or {}
        self.previous_hand: list[str] = []
        self.actions_sequence: list[dict[str, Any]] = []
        self.current_player = 0
        self.turn_number = 0

        # 設定パラメータ
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.min_hand_size = self.config.get("min_hand_size", 10)  # 手牌として認識する最小枚数
        self.max_hand_size = self.config.get("max_hand_size", 14)  # 手牌の最大枚数（カンを除く）
        self.enable_inference = self.config.get("enable_inference", True)  # 推測機能の有効化

        # アクション推測器
        self.inferencer = ActionInferencer() if self.enable_inference else None

        self.logger.info("SimplifiedActionDetector初期化完了")

    def detect_hand_change(
        self, current_hand: list[str], frame_number: int | None = None
    ) -> HandChangeResult:
        """
        手牌の変化からアクションを検出

        Args:
            current_hand: 現在検出された手牌
            frame_number: フレーム番号（オプション）

        Returns:
            手牌変化の検出結果
        """
        # 手牌の妥当性チェック
        if not self._is_valid_hand(current_hand):
            return HandChangeResult(
                action_type="unknown",
                confidence=0.0,
                hand_after=current_hand,
                metadata={"reason": "invalid_hand_size", "size": len(current_hand)},
            )

        # 初回検出時
        if not self.previous_hand:
            self.previous_hand = current_hand.copy()

            # 初回でも推測器に記録
            if self.inferencer:
                self.inferencer.record_player_hand(
                    self.current_player, current_hand, self.turn_number
                )

            return HandChangeResult(
                action_type="initial",
                confidence=1.0,
                hand_after=current_hand,
                metadata={"frame_number": frame_number},
            )

        # 手牌の類似度を計算
        similarity = self._calculate_hand_similarity(self.previous_hand, current_hand)

        # 完全に異なる手牌の場合は手番切り替え
        if similarity < self.similarity_threshold:
            result = HandChangeResult(
                action_type="turn_change",
                confidence=1.0 - similarity,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"similarity": similarity, "frame_number": frame_number},
            )

            # 手番切り替え
            self.current_player = (self.current_player + 1) % 4
            if self.current_player == 0:
                self.turn_number += 1

            # 新しいプレイヤーの手牌を記録
            if self.inferencer:
                self.inferencer.record_player_hand(
                    self.current_player, current_hand, self.turn_number
                )

            # 手牌を更新して早期リターン
            self.previous_hand = current_hand.copy()

            # アクションシーケンスに追加（initialは記録しない）
            if result.action_type != "initial":
                self.actions_sequence.append(
                    {
                        "action_type": result.action_type,
                        "tile": result.tile,
                        "confidence": result.confidence,
                        "frame_number": frame_number,
                        "player_index": self.current_player - 1,  # 切り替え前のプレイヤー
                    }
                )

            return result

        # 枚数変化を検出
        prev_count = len(self.previous_hand)
        curr_count = len(current_hand)
        diff = prev_count - curr_count

        if prev_count == 13 and curr_count == 14:
            # ツモ検出
            added_tile = self._find_added_tile(self.previous_hand, current_hand)
            result = HandChangeResult(
                action_type="draw",
                tile=added_tile,
                confidence=0.9 if added_tile else 0.5,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"frame_number": frame_number},
            )

        elif prev_count == 14 and curr_count == 13:
            # 捨て牌検出
            removed_tile = self._find_removed_tile(self.previous_hand, current_hand)
            result = HandChangeResult(
                action_type="discard",
                tile=removed_tile,
                confidence=0.9 if removed_tile else 0.5,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"frame_number": frame_number},
            )

        elif diff == 3:
            # ポン・チーの可能性
            result = HandChangeResult(
                action_type="call",
                confidence=0.7,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"call_type": "pon_or_chi", "frame_number": frame_number},
            )

        elif diff == 4:
            # カンの可能性（明槓・加槓）
            result = HandChangeResult(
                action_type="call",
                confidence=0.7,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"call_type": "kan", "frame_number": frame_number},
            )
        elif curr_count == prev_count and prev_count == 14:
            # 暗槓の可能性（枚数変化なし）
            result = HandChangeResult(
                action_type="call",
                confidence=0.6,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={"call_type": "ankan", "frame_number": frame_number},
            )

        else:
            # その他の変化
            result = HandChangeResult(
                action_type="unknown",
                confidence=0.3,
                hand_before=self.previous_hand.copy(),
                hand_after=current_hand.copy(),
                metadata={
                    "prev_count": prev_count,
                    "curr_count": curr_count,
                    "diff": diff,
                    "frame_number": frame_number,
                },
            )

        # 手牌を更新
        self.previous_hand = current_hand.copy()

        # アクションシーケンスに追加（initialは記録しない）
        if result.action_type != "initial":
            self.actions_sequence.append(
                {
                    "action_type": result.action_type,
                    "tile": result.tile,
                    "confidence": result.confidence,
                    "frame_number": frame_number,
                    "player_index": self.current_player,
                }
            )

        # 通常のアクションの場合、推測器に手牌を記録
        if result.action_type in ["draw", "discard", "call"] and self.inferencer:
            self.inferencer.record_player_hand(self.current_player, current_hand, self.turn_number)

        return result

    def _is_valid_hand(self, hand: list[str]) -> bool:
        """手牌の妥当性をチェック"""
        if not hand:
            return False

        hand_size = len(hand)
        # 手牌は通常10-14枚（鳴きがある場合は少なくなる）
        return self.min_hand_size <= hand_size <= self.max_hand_size

    def _calculate_hand_similarity(self, hand1: list[str], hand2: list[str]) -> float:
        """
        2つの手牌の類似度を計算

        Returns:
            0.0-1.0の類似度（1.0が完全一致）
        """
        if not hand1 or not hand2:
            return 0.0

        # 牌の集合として比較
        set1 = set(hand1)
        set2 = set(hand2)

        # Jaccard係数で類似度を計算
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _find_added_tile(self, prev_hand: list[str], curr_hand: list[str]) -> str | None:
        """追加された牌を特定"""
        prev_counts = self._count_tiles(prev_hand)
        curr_counts = self._count_tiles(curr_hand)

        for tile, count in curr_counts.items():
            if count > prev_counts.get(tile, 0):
                return tile

        return None

    def _find_removed_tile(self, prev_hand: list[str], curr_hand: list[str]) -> str | None:
        """削除された牌を特定"""
        prev_counts = self._count_tiles(prev_hand)
        curr_counts = self._count_tiles(curr_hand)

        for tile, count in prev_counts.items():
            if count > curr_counts.get(tile, 0):
                return tile

        return None

    def _count_tiles(self, tiles: list[str]) -> dict[str, int]:
        """牌の枚数をカウント"""
        counts = {}
        for tile in tiles:
            counts[tile] = counts.get(tile, 0) + 1
        return counts

    def get_action_sequence(self) -> list[dict[str, Any]]:
        """検出されたアクションシーケンスを取得"""
        return self.actions_sequence.copy()

    def get_inferred_actions(self) -> list[dict[str, Any]]:
        """
        推測されたアクションを取得

        Returns:
            推測されたアクションのリスト
        """
        if not self.inferencer:
            return []

        inferred = self.inferencer.get_inferred_actions()
        return [
            {
                "action_type": action.action_type,
                "tile": action.tile,
                "confidence": action.confidence,
                "reason": action.reason,
                "metadata": action.metadata,
            }
            for action in inferred
        ]

    def reset(self):
        """検出器をリセット"""
        self.previous_hand = []
        self.actions_sequence = []
        self.current_player = 0
        self.turn_number = 0
        if self.inferencer:
            self.inferencer.clear_history()
        self.logger.info("検出器をリセットしました")

    def convert_to_tenhou_format(self, include_inferred: bool = True) -> list[dict[str, Any]]:
        """
        アクションシーケンスを天鳳形式に変換

        Args:
            include_inferred: 推測されたアクションを含むか

        Returns:
            天鳳形式のアクションリスト
        """
        tenhou_actions = []
        current_player = 0  # 東家から開始

        # 通常のアクションを変換
        for action in self.actions_sequence:
            if action["action_type"] == "initial":
                # initialアクションはスキップ
                continue
            elif action["action_type"] == "turn_change":
                # 手番切り替え - 次のプレイヤーへ
                current_player = (current_player + 1) % 4
                continue

            if action["action_type"] in ["draw", "discard", "call"]:
                tenhou_action = {
                    "type": action["action_type"],
                    "player": action.get("player_index", current_player),
                    "tile": action["tile"],
                    "confidence": action["confidence"],
                }

                if action.get("frame_number"):
                    tenhou_action["frame"] = action["frame_number"]

                tenhou_actions.append(tenhou_action)

        # 推測されたアクションを追加
        if include_inferred and self.inferencer:
            inferred_actions = self.inferencer.get_inferred_actions()
            for inferred in inferred_actions:
                if inferred.action_type in ["draw", "discard"]:
                    tenhou_action = {
                        "type": inferred.action_type,
                        "player": inferred.metadata.get("player_index", 0),
                        "tile": inferred.tile,
                        "confidence": inferred.confidence,
                        "inferred": True,
                        "reason": inferred.reason,
                    }
                    tenhou_actions.append(tenhou_action)

        # フレーム番号またはターン番号でソート
        tenhou_actions.sort(key=lambda x: x.get("frame", x.get("turn", 0)))

        return tenhou_actions
