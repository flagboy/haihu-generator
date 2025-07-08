"""
行動検出クラス - SimplifiedActionDetectorへの移行

このモジュールは後方互換性のために残されています。
新しい実装はsimplified_action_detector.pyを参照してください。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..game.player import PlayerPosition
from ..game.turn import Action, ActionType
from ..utils.logger import LoggerMixin
from .simplified_action_detector import SimplifiedActionDetector


class DetectionConfidence(Enum):
    """検出信頼度レベル"""

    HIGH = "high"  # 高信頼度 (>0.8)
    MEDIUM = "medium"  # 中信頼度 (0.5-0.8)
    LOW = "low"  # 低信頼度 (<0.5)


@dataclass
class TileChange:
    """牌の変化情報（後方互換性のため維持）"""

    player: PlayerPosition
    tile: str
    change_type: str  # "added", "removed", "moved"
    location: str  # "hand", "discard", "call"
    confidence: float
    frame_number: int


@dataclass
class DetectionResult:
    """検出結果（後方互換性のため維持）"""

    actions: list[Action]
    tile_changes: list[TileChange]
    confidence_level: DetectionConfidence
    metadata: dict[str, Any]


class ActionDetector(LoggerMixin):
    """
    プレイヤーの行動検出クラス

    SimplifiedActionDetectorをラップして後方互換性を提供
    """

    def __init__(self):
        """行動検出クラスを初期化"""
        self.detector = SimplifiedActionDetector()
        self.previous_frame_data: dict[str, Any] | None = None
        self.detection_history: list[DetectionResult] = []
        self.current_player = 0

        # 検出パラメータ（後方互換性）
        self.min_confidence = 0.3
        self.hand_size_tolerance = 1
        self.discard_detection_threshold = 0.7

    def detect_actions(self, current_frame: dict[str, Any], frame_number: int) -> DetectionResult:
        """
        フレーム間の変化から行動を検出（SimplifiedActionDetectorを使用）

        Args:
            current_frame: 現在のフレームデータ
            frame_number: フレーム番号

        Returns:
            DetectionResult: 検出結果
        """
        actions = []
        tile_changes = []

        # 画面下部の手牌を取得（後方互換性のため）
        current_hand = []
        if "player_hands" in current_frame:
            # 手番プレイヤーの手牌は最初のエントリと仮定
            hands = current_frame["player_hands"]
            if hands:
                first_key = list(hands.keys())[0]
                current_hand = hands[first_key]
        elif "bottom_hand" in current_frame:
            current_hand = current_frame["bottom_hand"]

        # SimplifiedActionDetectorで手牌変化を検出
        result = self.detector.detect_hand_change(current_hand, frame_number)

        # 手番切り替えを検出したらプレイヤー番号を更新
        if result.action_type == "turn_change":
            self.current_player = (self.current_player + 1) % 4

        # HandChangeResultをActionに変換
        if result.action_type == "draw":
            action = Action(
                action_type=ActionType.DRAW,
                player=PlayerPosition(self.current_player),
                tile=result.tile,
                frame_number=frame_number,
                confidence=result.confidence,
                detected_by="hand_change",
            )
            actions.append(action)
        elif result.action_type == "discard":
            action = Action(
                action_type=ActionType.DISCARD,
                player=PlayerPosition(self.current_player),
                tile=result.tile,
                frame_number=frame_number,
                confidence=result.confidence,
                detected_by="hand_change",
            )
            actions.append(action)

        # 信頼度を計算
        if result.confidence > 0.8:
            confidence_level = DetectionConfidence.HIGH
        elif result.confidence > 0.5:
            confidence_level = DetectionConfidence.MEDIUM
        else:
            confidence_level = DetectionConfidence.LOW

        # 結果を作成
        detection_result = DetectionResult(
            actions=actions,
            tile_changes=tile_changes,
            confidence_level=confidence_level,
            metadata={
                "frame_number": frame_number,
                "detection_count": len(actions),
                "hand_change_type": result.action_type,
                "confidence": result.confidence,
            },
        )

        # 履歴に追加
        self.detection_history.append(detection_result)
        self.previous_frame_data = current_frame.copy()

        return detection_result

    def _detect_initial_deal(self, frame_data: dict[str, Any], frame_number: int) -> list[Action]:
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
        player_hands = frame_data.get("player_hands", {})

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
                            confidence=0.9,
                        )
                        actions.append(action)
            except (ValueError, KeyError):
                continue

        return actions

    def _analyze_tile_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[TileChange]:
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
        changes.extend(self._detect_hand_changes(previous_frame, current_frame, frame_number))

        # 捨て牌の変化を検出
        changes.extend(self._detect_discard_changes(previous_frame, current_frame, frame_number))

        # 鳴きの変化を検出
        changes.extend(self._detect_call_changes(previous_frame, current_frame, frame_number))

        return changes

    def _detect_hand_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[TileChange]:
        """手牌の変化を検出"""
        changes = []

        prev_hands = previous_frame.get("player_hands", {})
        curr_hands = current_frame.get("player_hands", {})

        for pos_str in set(prev_hands.keys()) | set(curr_hands.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_hands.get(pos_str, [])
                curr_tiles = curr_hands.get(pos_str, [])

                # 追加された牌
                added_tiles = [tile for tile in curr_tiles if tile not in prev_tiles]
                for tile in added_tiles:
                    changes.append(
                        TileChange(
                            player=pos,
                            tile=tile,
                            change_type="added",
                            location="hand",
                            confidence=0.8,
                            frame_number=frame_number,
                        )
                    )

                # 削除された牌
                removed_tiles = [tile for tile in prev_tiles if tile not in curr_tiles]
                for tile in removed_tiles:
                    changes.append(
                        TileChange(
                            player=pos,
                            tile=tile,
                            change_type="removed",
                            location="hand",
                            confidence=0.8,
                            frame_number=frame_number,
                        )
                    )

            except (ValueError, KeyError):
                continue

        return changes

    def _detect_discard_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[TileChange]:
        """捨て牌の変化を検出"""
        changes = []

        prev_discards = previous_frame.get("discarded_tiles", {})
        curr_discards = current_frame.get("discarded_tiles", {})

        for pos_str in set(prev_discards.keys()) | set(curr_discards.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_discards.get(pos_str, [])
                curr_tiles = curr_discards.get(pos_str, [])

                # 新しく捨てられた牌
                if len(curr_tiles) > len(prev_tiles):
                    new_discards = curr_tiles[len(prev_tiles) :]
                    for tile in new_discards:
                        changes.append(
                            TileChange(
                                player=pos,
                                tile=tile,
                                change_type="added",
                                location="discard",
                                confidence=0.9,
                                frame_number=frame_number,
                            )
                        )

            except (ValueError, KeyError):
                continue

        return changes

    def _detect_call_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[TileChange]:
        """鳴きの変化を検出"""
        changes = []

        # 鳴き情報の変化を検出（実装は簡略化）
        prev_calls = previous_frame.get("calls", {})
        curr_calls = current_frame.get("calls", {})

        for pos_str in set(curr_calls.keys()) - set(prev_calls.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                call_tiles = curr_calls.get(pos_str, [])

                for tile in call_tiles:
                    changes.append(
                        TileChange(
                            player=pos,
                            tile=tile,
                            change_type="added",
                            location="call",
                            confidence=0.7,
                            frame_number=frame_number,
                        )
                    )

            except (ValueError, KeyError):
                continue

        return changes

    def _infer_actions_from_changes(
        self, tile_changes: list[TileChange], frame_number: int
    ) -> list[Action]:
        """牌の変化から行動を推定"""
        actions = []

        # プレイヤーごとに変化を整理
        player_changes: dict[PlayerPosition, dict[str, Any]] = {}
        for change in tile_changes:
            if change.player not in player_changes:
                player_changes[change.player] = {"changes": []}
            player_changes[change.player]["changes"].append(change)

        # 各プレイヤーの行動を推定
        for player, player_data in player_changes.items():
            changes_list = player_data.get("changes", [])
            if changes_list:
                actions.extend(self._infer_player_actions(player, changes_list, frame_number))

        return actions

    def _infer_player_actions(
        self, player: PlayerPosition, changes: list[TileChange], frame_number: int
    ) -> list[Action]:
        """特定プレイヤーの行動を推定"""
        actions = []

        hand_added = [c for c in changes if c.location == "hand" and c.change_type == "added"]
        [c for c in changes if c.location == "hand" and c.change_type == "removed"]
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
                detected_by="change_analysis",
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
                detected_by="change_analysis",
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
                detected_by="change_analysis",
            )
            actions.append(action)

        return actions

    def _infer_call_type(self, tiles: list[str]) -> ActionType:
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

    def _calculate_confidence_level(
        self, actions: list[Action], tile_changes: list[TileChange]
    ) -> DetectionConfidence:
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

    def get_recent_detections(self, count: int = 10) -> list[DetectionResult]:
        """最近の検出結果を取得"""
        return (
            self.detection_history[-count:]
            if len(self.detection_history) > count
            else self.detection_history
        )

    def reset(self):
        """検出器をリセット"""
        self.detector.reset()
        self.previous_frame_data = None
        self.detection_history = []
        self.current_player = 0

    def __str__(self) -> str:
        """文字列表現"""
        return f"ActionDetector(history: {len(self.detection_history)})"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"ActionDetector(detection_count={len(self.detection_history)}, "
            f"min_confidence={self.min_confidence})"
        )
