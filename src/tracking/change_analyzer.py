"""
変化分析クラス
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..game.player import PlayerPosition
from ..utils.tile_definitions import TileDefinitions


class ChangeType(Enum):
    """変化の種類"""

    TILE_ADDED = "tile_added"  # 牌が追加された
    TILE_REMOVED = "tile_removed"  # 牌が削除された
    TILE_MOVED = "tile_moved"  # 牌が移動した
    POSITION_CHANGED = "position_changed"  # 位置が変更された
    COUNT_CHANGED = "count_changed"  # 枚数が変更された


class ChangeLocation(Enum):
    """変化の場所"""

    HAND = "hand"  # 手牌
    DISCARD = "discard"  # 捨て牌
    CALL = "call"  # 鳴き
    TABLE = "table"  # 卓上
    UNKNOWN = "unknown"  # 不明


@dataclass
class TileMovement:
    """牌の移動情報"""

    tile: str
    from_location: ChangeLocation
    to_location: ChangeLocation
    from_player: PlayerPosition | None = None
    to_player: PlayerPosition | None = None
    confidence: float = 1.0
    frame_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeEvent:
    """変化イベント"""

    change_type: ChangeType
    location: ChangeLocation
    player: PlayerPosition | None
    tiles: list[str]
    previous_state: dict[str, Any]
    current_state: dict[str, Any]
    confidence: float
    frame_number: int
    movements: list[TileMovement] = field(default_factory=list)

    def __post_init__(self):
        """初期化後の処理"""
        if not self.movements:
            self.movements = []


class ChangeAnalyzer:
    """牌の移動・変化の分析クラス"""

    def __init__(self):
        """変化分析クラスを初期化"""
        self.tile_definitions = TileDefinitions()

        # 分析履歴
        self.change_history: list[ChangeEvent] = []
        self.movement_history: list[TileMovement] = []

        # 分析パラメータ
        self.similarity_threshold = 0.8
        self.movement_confidence_threshold = 0.6
        self.max_history_size = 1000

        # 統計情報
        self.total_changes_analyzed = 0
        self.movements_detected = 0

    def analyze_frame_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[ChangeEvent]:
        """
        フレーム間の変化を分析

        Args:
            previous_frame: 前フレームのデータ
            current_frame: 現フレームのデータ
            frame_number: フレーム番号

        Returns:
            List[ChangeEvent]: 検出された変化イベントのリスト
        """
        change_events = []

        # 各プレイヤーの手牌変化を分析
        change_events.extend(
            self._analyze_hand_changes(previous_frame, current_frame, frame_number)
        )

        # 捨て牌の変化を分析
        change_events.extend(
            self._analyze_discard_changes(previous_frame, current_frame, frame_number)
        )

        # 鳴きの変化を分析
        change_events.extend(
            self._analyze_call_changes(previous_frame, current_frame, frame_number)
        )

        # 牌の移動を推定
        movements = self._infer_tile_movements(change_events, frame_number)

        # 移動情報を変化イベントに追加
        for event in change_events:
            event.movements = [m for m in movements if self._is_related_movement(event, m)]

        # 履歴に追加
        self.change_history.extend(change_events)
        self.movement_history.extend(movements)
        self.total_changes_analyzed += len(change_events)
        self.movements_detected += len(movements)

        # 履歴サイズを制限
        self._trim_history()

        return change_events

    def _analyze_hand_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[ChangeEvent]:
        """手牌の変化を分析"""
        changes = []

        prev_hands = previous_frame.get("player_hands", {})
        curr_hands = current_frame.get("player_hands", {})

        for pos_str in set(prev_hands.keys()) | set(curr_hands.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_hands.get(pos_str, [])
                curr_tiles = curr_hands.get(pos_str, [])

                # 変化を検出
                change_event = self._detect_tile_list_changes(
                    prev_tiles, curr_tiles, ChangeLocation.HAND, pos, frame_number
                )

                if change_event:
                    changes.append(change_event)

            except (ValueError, KeyError):
                continue

        return changes

    def _analyze_discard_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[ChangeEvent]:
        """捨て牌の変化を分析"""
        changes = []

        prev_discards = previous_frame.get("discarded_tiles", {})
        curr_discards = current_frame.get("discarded_tiles", {})

        for pos_str in set(prev_discards.keys()) | set(curr_discards.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                prev_tiles = prev_discards.get(pos_str, [])
                curr_tiles = curr_discards.get(pos_str, [])

                # 捨て牌は通常追加のみ
                if len(curr_tiles) > len(prev_tiles):
                    new_tiles = curr_tiles[len(prev_tiles) :]

                    change_event = ChangeEvent(
                        change_type=ChangeType.TILE_ADDED,
                        location=ChangeLocation.DISCARD,
                        player=pos,
                        tiles=new_tiles,
                        previous_state={"tiles": prev_tiles},
                        current_state={"tiles": curr_tiles},
                        confidence=0.9,
                        frame_number=frame_number,
                    )
                    changes.append(change_event)

            except (ValueError, KeyError):
                continue

        return changes

    def _analyze_call_changes(
        self, previous_frame: dict[str, Any], current_frame: dict[str, Any], frame_number: int
    ) -> list[ChangeEvent]:
        """鳴きの変化を分析"""
        changes = []

        prev_calls = previous_frame.get("calls", {})
        curr_calls = current_frame.get("calls", {})

        for pos_str in set(curr_calls.keys()) - set(prev_calls.keys()):
            try:
                pos = PlayerPosition(int(pos_str))
                call_tiles = curr_calls.get(pos_str, [])

                if call_tiles:
                    change_event = ChangeEvent(
                        change_type=ChangeType.TILE_ADDED,
                        location=ChangeLocation.CALL,
                        player=pos,
                        tiles=call_tiles,
                        previous_state={"calls": prev_calls.get(pos_str, [])},
                        current_state={"calls": call_tiles},
                        confidence=0.8,
                        frame_number=frame_number,
                    )
                    changes.append(change_event)

            except (ValueError, KeyError):
                continue

        return changes

    def _detect_tile_list_changes(
        self,
        prev_tiles: list[str],
        curr_tiles: list[str],
        location: ChangeLocation,
        player: PlayerPosition,
        frame_number: int,
    ) -> ChangeEvent | None:
        """牌リストの変化を検出"""
        if prev_tiles == curr_tiles:
            return None

        # 追加された牌
        added_tiles = []
        removed_tiles = []

        prev_counts = self._count_tiles(prev_tiles)
        curr_counts = self._count_tiles(curr_tiles)

        all_tiles = set(prev_counts.keys()) | set(curr_counts.keys())

        for tile in all_tiles:
            prev_count = prev_counts.get(tile, 0)
            curr_count = curr_counts.get(tile, 0)

            if curr_count > prev_count:
                added_tiles.extend([tile] * (curr_count - prev_count))
            elif prev_count > curr_count:
                removed_tiles.extend([tile] * (prev_count - curr_count))

        # 変化の種類を決定
        if added_tiles and not removed_tiles:
            change_type = ChangeType.TILE_ADDED
            tiles = added_tiles
        elif removed_tiles and not added_tiles:
            change_type = ChangeType.TILE_REMOVED
            tiles = removed_tiles
        else:
            change_type = ChangeType.TILE_MOVED
            tiles = added_tiles + removed_tiles

        # 信頼度を計算
        confidence = self._calculate_change_confidence(prev_tiles, curr_tiles, change_type)

        return ChangeEvent(
            change_type=change_type,
            location=location,
            player=player,
            tiles=tiles,
            previous_state={"tiles": prev_tiles, "counts": prev_counts},
            current_state={"tiles": curr_tiles, "counts": curr_counts},
            confidence=confidence,
            frame_number=frame_number,
        )

    def _count_tiles(self, tiles: list[str]) -> dict[str, int]:
        """牌の枚数をカウント"""
        counts: defaultdict[str, int] = defaultdict(int)
        for tile in tiles:
            if self.tile_definitions.is_valid_tile(tile):
                counts[tile] += 1
        return dict(counts)

    def _calculate_change_confidence(
        self, prev_tiles: list[str], curr_tiles: list[str], change_type: ChangeType
    ) -> float:
        """変化の信頼度を計算"""
        # 基本信頼度
        base_confidence = 0.7

        # 変化の大きさに基づく調整
        change_magnitude = abs(len(curr_tiles) - len(prev_tiles))

        if change_magnitude == 1:
            # 1枚の変化は通常の行動
            confidence_adjustment = 0.2
        elif change_magnitude <= 3:
            # 2-3枚の変化は鳴きなど
            confidence_adjustment = 0.1
        else:
            # 大きな変化は信頼度を下げる
            confidence_adjustment = -0.2

        return max(0.1, min(1.0, base_confidence + confidence_adjustment))

    def _infer_tile_movements(
        self, change_events: list[ChangeEvent], frame_number: int
    ) -> list[TileMovement]:
        """変化イベントから牌の移動を推定"""
        movements = []

        # 削除された牌と追加された牌をマッチング
        removed_tiles = []
        added_tiles = []

        for event in change_events:
            if event.change_type == ChangeType.TILE_REMOVED:
                for tile in event.tiles:
                    removed_tiles.append((tile, event.location, event.player))
            elif event.change_type == ChangeType.TILE_ADDED:
                for tile in event.tiles:
                    added_tiles.append((tile, event.location, event.player))

        # 同じ牌の削除と追加をマッチング
        for removed_tile, from_loc, from_player in removed_tiles:
            for added_tile, to_loc, to_player in added_tiles:
                if removed_tile == added_tile:
                    movement = TileMovement(
                        tile=removed_tile,
                        from_location=from_loc,
                        to_location=to_loc,
                        from_player=from_player,
                        to_player=to_player,
                        confidence=0.8,
                        frame_number=frame_number,
                    )
                    movements.append(movement)
                    break

        return movements

    def _is_related_movement(self, event: ChangeEvent, movement: TileMovement) -> bool:
        """変化イベントと移動が関連しているかチェック"""
        # 同じフレームで同じ牌が関わっている場合
        return movement.frame_number == event.frame_number and movement.tile in event.tiles

    def get_movement_patterns(self) -> dict[str, Any]:
        """移動パターンの統計を取得"""
        patterns: defaultdict[str, int] = defaultdict(int)
        tile_movements: defaultdict[str, int] = defaultdict(int)
        player_movements: defaultdict[str, int] = defaultdict(int)

        for movement in self.movement_history:
            # 移動パターン
            pattern = f"{movement.from_location.value}_to_{movement.to_location.value}"
            patterns[pattern] += 1

            # 牌別移動
            tile_movements[movement.tile] += 1

            # プレイヤー別移動
            if movement.from_player:
                player_movements[movement.from_player.name] += 1

        return {
            "movement_patterns": dict(patterns),
            "tile_movements": dict(tile_movements),
            "player_movements": dict(player_movements),
            "total_movements": len(self.movement_history),
        }

    def get_change_statistics(self) -> dict[str, Any]:
        """変化統計を取得"""
        change_types: defaultdict[str, int] = defaultdict(int)
        location_changes: defaultdict[str, int] = defaultdict(int)
        player_changes: defaultdict[str, int] = defaultdict(int)

        for change in self.change_history:
            change_types[change.change_type.value] += 1
            location_changes[change.location.value] += 1

            if change.player:
                player_changes[change.player.name] += 1

        return {
            "total_changes": self.total_changes_analyzed,
            "change_types": dict(change_types),
            "location_changes": dict(location_changes),
            "player_changes": dict(player_changes),
            "movements_detected": self.movements_detected,
        }

    def _trim_history(self):
        """履歴サイズを制限"""
        if len(self.change_history) > self.max_history_size:
            self.change_history = self.change_history[-self.max_history_size :]

        if len(self.movement_history) > self.max_history_size:
            self.movement_history = self.movement_history[-self.max_history_size :]

    def reset(self):
        """分析器をリセット"""
        self.change_history = []
        self.movement_history = []
        self.total_changes_analyzed = 0
        self.movements_detected = 0

    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"ChangeAnalyzer(Changes: {self.total_changes_analyzed}, "
            f"Movements: {self.movements_detected})"
        )

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"ChangeAnalyzer(changes={self.total_changes_analyzed}, "
            f"movements={self.movements_detected}, "
            f"history_size={len(self.change_history)})"
        )
