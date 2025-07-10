"""
シンプルなゲームパイプライン

手牌変化ベースのアクション検出に特化したシンプルなパイプライン。
複雑な状態追跡や履歴管理を省き、天鳳形式の牌譜生成に必要な最小限の機能のみを提供。
"""

import time
from dataclasses import dataclass
from typing import Any

from ..game.simplified_game_state import SimplifiedGameStateManager
from ..output.tenhou_json_formatter import TenhouJsonFormatter
from ..tracking.simplified_action_detector import SimplifiedActionDetector
from ..utils.logger import LoggerMixin


@dataclass
class SimplifiedProcessingResult:
    """シンプルな処理結果"""

    success: bool
    frame_number: int
    action_type: str | None = None
    tile: str | None = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: str | None = None


class SimplifiedGamePipeline(LoggerMixin):
    """シンプルなゲームパイプライン"""

    def __init__(self, game_id: str = "default"):
        """
        初期化

        Args:
            game_id: ゲームID
        """
        self.game_id = game_id
        self.detector = SimplifiedActionDetector()
        self.game_state = SimplifiedGameStateManager()
        self.formatter = TenhouJsonFormatter()

        # 統計情報
        self.frames_processed = 0
        self.actions_detected = 0
        self.start_time = time.time()

        self.logger.info(f"SimplifiedGamePipeline初期化完了: {game_id}")

    def process_frame(self, frame_data: dict[str, Any]) -> SimplifiedProcessingResult:
        """
        フレームを処理

        Args:
            frame_data: フレームデータ（手牌情報を含む）

        Returns:
            処理結果
        """
        start_time = time.time()
        frame_number = frame_data.get("frame_number", self.frames_processed)

        result = SimplifiedProcessingResult(success=False, frame_number=frame_number)

        try:
            # 手牌を取得（画面下部の手牌）
            hand_tiles = self._extract_hand_tiles(frame_data)

            if not hand_tiles:
                result.error = "手牌が検出されませんでした"
                return result

            # 手牌変化を検出
            detection = self.detector.detect_hand_change(hand_tiles, frame_number)

            # アクションをゲーム状態に追加
            if detection.action_type in ["draw", "discard", "call", "turn_change"]:
                self.game_state.add_action(
                    action_type=detection.action_type,
                    tile=detection.tile,
                    frame_number=frame_number,
                    confidence=detection.confidence,
                )

                result.success = True
                result.action_type = detection.action_type
                result.tile = detection.tile
                result.confidence = detection.confidence

                if detection.action_type != "turn_change":
                    self.actions_detected += 1

                self.logger.debug(
                    f"フレーム{frame_number}: {detection.action_type} "
                    f"(牌: {detection.tile}, 信頼度: {detection.confidence:.2f})"
                )
            else:
                result.error = f"不明なアクションタイプ: {detection.action_type}"

        except Exception as e:
            result.error = f"処理エラー: {str(e)}"
            self.logger.error(f"フレーム{frame_number}の処理エラー: {e}")

        finally:
            result.processing_time = time.time() - start_time
            self.frames_processed += 1

        return result

    def _extract_hand_tiles(self, frame_data: dict[str, Any]) -> list[str]:
        """
        フレームデータから手牌を抽出

        Args:
            frame_data: フレームデータ

        Returns:
            手牌のリスト
        """
        # 複数の形式に対応
        if "bottom_hand" in frame_data:
            return frame_data["bottom_hand"]

        if "current_player_hand" in frame_data:
            return frame_data["current_player_hand"]

        if "hand_tiles" in frame_data:
            return frame_data["hand_tiles"]

        # player_hands形式（互換性のため）
        if "player_hands" in frame_data:
            hands = frame_data["player_hands"]
            if isinstance(hands, dict) and hands:
                # 最初のエントリが現在の手番プレイヤーと仮定
                first_key = list(hands.keys())[0]
                return hands[first_key]

        return []

    def process_batch(self, frames: list[dict[str, Any]]) -> list[SimplifiedProcessingResult]:
        """
        複数フレームをバッチ処理

        Args:
            frames: フレームデータのリスト

        Returns:
            処理結果のリスト
        """
        results = []

        for frame in frames:
            result = self.process_frame(frame)
            results.append(result)

        return results

    def export_to_tenhou_json(self) -> str:
        """
        天鳳JSON形式でエクスポート

        Returns:
            天鳳JSON形式の文字列
        """
        # シンプルなアクションリストを天鳳形式に変換
        tenhou_actions = self.game_state.export_to_tenhou_format()

        # 天鳳形式のデータ構造を作成
        game_data = {
            "title": f"Game_{self.game_id}_{int(time.time())}",
            "name": ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"],
            "rule": {"disp": "東風戦", "aka": 1, "kuitan": 1, "tonnan": 0},
            "log": self._convert_actions_to_log(tenhou_actions),
            "sc": [25000, 25000, 25000, 25000],  # 初期点数
            "owari": {
                "順位": [1, 2, 3, 4],
                "得点": [25000, 25000, 25000, 25000],
                "ウマ": [15, 5, -5, -15],
            },
        }

        return self.formatter.format_game_data(game_data)

    def _convert_actions_to_log(self, actions: list[dict[str, Any]]) -> list[list[Any]]:
        """
        アクションを天鳳ログ形式に変換

        Args:
            actions: アクションリスト

        Returns:
            天鳳ログ形式のリスト
        """
        log = []

        for action in actions:
            player = action["player"]
            action_type = action["type"]

            if action_type == "draw":
                log.append([f"T{player}", action.get("tile", "")])
            elif action_type == "discard":
                log.append([f"D{player}", action.get("tile", "")])
            elif action_type in ["pon", "chi", "kan"]:
                log.append([f"N{player}", action_type, action.get("tiles", [])])

        return log

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        elapsed = time.time() - self.start_time

        return {
            "game_id": self.game_id,
            "frames_processed": self.frames_processed,
            "actions_detected": self.actions_detected,
            "detection_rate": self.actions_detected / max(1, self.frames_processed),
            "elapsed_time": elapsed,
            "fps": self.frames_processed / max(1, elapsed),
            "game_state": self.game_state.get_statistics(),
            "validation_errors": self.game_state.validate_action_sequence(),
        }

    def reset(self):
        """パイプラインをリセット"""
        self.detector.reset()
        self.game_state.reset()
        self.frames_processed = 0
        self.actions_detected = 0
        self.start_time = time.time()
        self.logger.info("パイプラインをリセットしました")
