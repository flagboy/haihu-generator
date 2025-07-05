"""
状態追跡クラス
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..game.game_state import GameState
from ..game.player import PlayerPosition
from ..utils.tile_definitions import TileDefinitions
from .action_detector import ActionDetector, DetectionResult


class TrackingState(Enum):
    """追跡状態"""

    STABLE = "stable"  # 安定状態
    DETECTING = "detecting"  # 検出中
    INCONSISTENT = "inconsistent"  # 矛盾状態
    RECOVERING = "recovering"  # 回復中


@dataclass
class StateSnapshot:
    """状態スナップショット"""

    frame_number: int
    timestamp: float
    game_state: Any  # GameStateのコピー
    confidence: float
    tracking_state: TrackingState
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InconsistencyReport:
    """矛盾レポート"""

    frame_number: int
    inconsistency_type: str
    description: str
    affected_players: list[PlayerPosition]
    severity: str  # "low", "medium", "high"
    suggested_action: str


class StateTracker:
    """フレーム間での状態変化追跡クラス"""

    def __init__(self, game_state: GameState):
        """
        状態追跡クラスを初期化

        Args:
            game_state: 追跡対象のゲーム状態
        """
        self.game_state = game_state
        self.action_detector = ActionDetector()
        self.tile_definitions = TileDefinitions()

        # 追跡状態
        self.tracking_state = TrackingState.STABLE
        self.snapshots: list[StateSnapshot] = []
        self.inconsistencies: list[InconsistencyReport] = []

        # 追跡パラメータ
        self.max_snapshots = 100
        self.confidence_threshold = 0.7
        self.inconsistency_tolerance = 3
        self.recovery_frames = 5

        # 統計情報
        self.total_frames_processed = 0
        self.successful_updates = 0
        self.failed_updates = 0
        self.inconsistency_count = 0

    def update_from_frame(self, frame_data: dict[str, Any]) -> bool:
        """
        フレームデータから状態を更新

        Args:
            frame_data: フレーム検出データ

        Returns:
            bool: 更新に成功したかどうか
        """
        frame_number = frame_data.get("frame_number", self.total_frames_processed)
        timestamp = frame_data.get("timestamp", time.time())

        self.total_frames_processed += 1

        try:
            # 行動を検出
            detection_result = self.action_detector.detect_actions(frame_data, frame_number)

            # 矛盾をチェック
            inconsistencies = self._check_inconsistencies(frame_data, detection_result)

            if inconsistencies:
                self._handle_inconsistencies(inconsistencies, frame_number)
                return False

            # ゲーム状態を更新
            success = self.game_state.update_from_frame_detection(frame_data)

            if success:
                # 検出された行動を適用
                self.game_state.apply_pending_actions()

                # スナップショットを作成
                self._create_snapshot(frame_number, timestamp, detection_result)

                self.successful_updates += 1
                self.tracking_state = TrackingState.STABLE

                return True
            else:
                self.failed_updates += 1
                return False

        except Exception as e:
            print(f"Error updating state from frame {frame_number}: {e}")
            self.failed_updates += 1
            return False

    def _check_inconsistencies(
        self, frame_data: dict[str, Any], detection_result: DetectionResult
    ) -> list[InconsistencyReport]:
        """
        矛盾をチェック

        Args:
            frame_data: フレームデータ
            detection_result: 検出結果

        Returns:
            List[InconsistencyReport]: 検出された矛盾のリスト
        """
        inconsistencies = []
        frame_number = frame_data.get("frame_number", 0)

        # 手牌枚数の矛盾をチェック
        inconsistencies.extend(self._check_hand_size_inconsistencies(frame_data, frame_number))

        # 牌の総数の矛盾をチェック
        inconsistencies.extend(self._check_tile_count_inconsistencies(frame_data, frame_number))

        # 行動の論理的矛盾をチェック
        inconsistencies.extend(self._check_action_inconsistencies(detection_result, frame_number))

        return inconsistencies

    def _check_hand_size_inconsistencies(
        self, frame_data: dict[str, Any], frame_number: int
    ) -> list[InconsistencyReport]:
        """手牌枚数の矛盾をチェック"""
        inconsistencies = []
        player_hands = frame_data.get("player_hands", {})

        for pos_str, tiles in player_hands.items():
            try:
                pos = PlayerPosition(int(pos_str))
                hand_size = len(tiles)

                # 通常の手牌枚数は13枚または14枚
                if hand_size < 10 or hand_size > 15:
                    inconsistencies.append(
                        InconsistencyReport(
                            frame_number=frame_number,
                            inconsistency_type="hand_size",
                            description=f"Player {pos.name} has {hand_size} tiles (expected 13-14)",
                            affected_players=[pos],
                            severity="medium",
                            suggested_action="verify_detection",
                        )
                    )

            except (ValueError, KeyError):
                continue

        return inconsistencies

    def _check_tile_count_inconsistencies(
        self, frame_data: dict[str, Any], frame_number: int
    ) -> list[InconsistencyReport]:
        """牌の総数の矛盾をチェック"""
        inconsistencies = []

        # 全プレイヤーの牌を集計
        all_tiles = []
        player_hands = frame_data.get("player_hands", {})
        discarded_tiles = frame_data.get("discarded_tiles", {})

        for tiles in player_hands.values():
            all_tiles.extend(tiles)

        for tiles in discarded_tiles.values():
            all_tiles.extend(tiles)

        # 各牌の枚数をチェック
        tile_counts: dict[str, int] = {}
        for tile in all_tiles:
            if self.tile_definitions.is_valid_tile(tile):
                tile_counts[tile] = tile_counts.get(tile, 0) + 1

        for tile, count in tile_counts.items():
            max_count = 1 if self.tile_definitions.is_red_dora(tile) else 4

            if count > max_count:
                inconsistencies.append(
                    InconsistencyReport(
                        frame_number=frame_number,
                        inconsistency_type="tile_count",
                        description=f"Tile {tile} appears {count} times (max: {max_count})",
                        affected_players=list(PlayerPosition),
                        severity="high",
                        suggested_action="recheck_detection",
                    )
                )

        return inconsistencies

    def _check_action_inconsistencies(
        self, detection_result: DetectionResult, frame_number: int
    ) -> list[InconsistencyReport]:
        """行動の論理的矛盾をチェック"""
        inconsistencies = []

        # 同一フレームでの矛盾する行動をチェック
        player_actions: dict[Any, list[Any]] = {}
        for action in detection_result.actions:
            if action.player not in player_actions:
                player_actions[action.player] = []
            player_actions[action.player].append(action)

        for player, actions in player_actions.items():
            # 同一プレイヤーが同時に複数の行動を取ることはない
            if len(actions) > 2:  # ツモ+打牌は許可
                inconsistencies.append(
                    InconsistencyReport(
                        frame_number=frame_number,
                        inconsistency_type="multiple_actions",
                        description=f"Player {player.name} has {len(actions)} actions in one frame",
                        affected_players=[player],
                        severity="medium",
                        suggested_action="filter_actions",
                    )
                )

        return inconsistencies

    def _handle_inconsistencies(
        self, inconsistencies: list[InconsistencyReport], frame_number: int
    ):
        """矛盾を処理"""
        self.inconsistencies.extend(inconsistencies)
        self.inconsistency_count += len(inconsistencies)

        # 重大な矛盾の場合は追跡状態を変更
        high_severity_count = sum(1 for inc in inconsistencies if inc.severity == "high")

        if high_severity_count > 0:
            self.tracking_state = TrackingState.INCONSISTENT
        elif len(inconsistencies) > self.inconsistency_tolerance:
            self.tracking_state = TrackingState.DETECTING

        # 矛盾をログ出力
        for inconsistency in inconsistencies:
            print(f"Inconsistency detected at frame {frame_number}: {inconsistency.description}")

    def _create_snapshot(
        self, frame_number: int, timestamp: float, detection_result: DetectionResult
    ):
        """状態スナップショットを作成"""
        # ゲーム状態のコピーを作成
        game_state_copy = {
            "players": self.game_state.get_current_player_states(),
            "table": self.game_state.get_current_table_state(),
            "phase": self.game_state.phase,
        }

        # 信頼度を計算
        confidence = self._calculate_overall_confidence(detection_result)

        snapshot = StateSnapshot(
            frame_number=frame_number,
            timestamp=timestamp,
            game_state=game_state_copy,
            confidence=confidence,
            tracking_state=self.tracking_state,
            metadata={
                "actions_count": len(detection_result.actions),
                "changes_count": len(detection_result.tile_changes),
            },
        )

        self.snapshots.append(snapshot)

        # 古いスナップショットを削除
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def _calculate_overall_confidence(self, detection_result: DetectionResult) -> float:
        """全体的な信頼度を計算"""
        if not detection_result.actions and not detection_result.tile_changes:
            return 1.0  # 変化がない場合は高信頼度

        action_confidences = [action.confidence for action in detection_result.actions]
        change_confidences = [change.confidence for change in detection_result.tile_changes]

        all_confidences = action_confidences + change_confidences

        if not all_confidences:
            return 0.5

        return sum(all_confidences) / len(all_confidences)

    def get_current_confidence(self) -> float:
        """現在の追跡信頼度を取得"""
        if not self.snapshots:
            return 0.0

        recent_snapshots = self.snapshots[-5:]  # 最近5フレーム
        confidences = [snapshot.confidence for snapshot in recent_snapshots]

        return sum(confidences) / len(confidences)

    def get_tracking_statistics(self) -> dict[str, Any]:
        """追跡統計を取得"""
        success_rate = (
            self.successful_updates / self.total_frames_processed
            if self.total_frames_processed > 0
            else 0.0
        )

        return {
            "total_frames": self.total_frames_processed,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "success_rate": success_rate,
            "inconsistency_count": self.inconsistency_count,
            "current_confidence": self.get_current_confidence(),
            "tracking_state": self.tracking_state.value,
            "snapshots_count": len(self.snapshots),
        }

    def get_recent_inconsistencies(self, count: int = 10) -> list[InconsistencyReport]:
        """最近の矛盾を取得"""
        return (
            self.inconsistencies[-count:]
            if len(self.inconsistencies) > count
            else self.inconsistencies
        )

    def rollback_to_snapshot(self, frame_number: int) -> bool:
        """指定フレームのスナップショットに戻す"""
        target_snapshot = None

        for snapshot in reversed(self.snapshots):
            if snapshot.frame_number <= frame_number:
                target_snapshot = snapshot
                break

        if target_snapshot is None:
            return False

        try:
            # ゲーム状態を復元
            # 注意: 実際の実装では適切な復元処理が必要
            print(f"Rolling back to frame {target_snapshot.frame_number}")
            self.tracking_state = TrackingState.RECOVERING
            return True

        except Exception as e:
            print(f"Error rolling back to snapshot: {e}")
            return False

    def reset(self):
        """追跡器をリセット"""
        self.action_detector.reset()
        self.tracking_state = TrackingState.STABLE
        self.snapshots = []
        self.inconsistencies = []

        self.total_frames_processed = 0
        self.successful_updates = 0
        self.failed_updates = 0
        self.inconsistency_count = 0

    def __str__(self) -> str:
        """文字列表現"""
        return f"StateTracker({self.tracking_state.value}, {len(self.snapshots)} snapshots)"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"StateTracker(state={self.tracking_state.value}, "
            f"frames={self.total_frames_processed}, "
            f"success_rate={self.successful_updates / max(1, self.total_frames_processed):.2f})"
        )
