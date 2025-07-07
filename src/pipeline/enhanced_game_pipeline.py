"""
拡張ゲーム統合パイプライン

シーン検出、点数読み取り、プレイヤー検出機能を統合した
高精度な牌譜作成パイプライン
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..detection import (
    PlayerDetector,
    SceneDetector,
    SceneType,
    ScoreReader,
    TileDetector,
)
from ..detection import (
    PlayerPosition as DetectorPlayerPosition,
)
from ..pipeline.game_pipeline import GamePipeline, ProcessingResult
from ..tracking.action_detector import ActionDetector


@dataclass
class EnhancedProcessingResult(ProcessingResult):
    """拡張処理結果"""

    scene_type: SceneType | None = None
    player_scores: dict[str, int] | None = None
    active_player: str | None = None
    dealer_position: str | None = None
    round_info: dict[str, Any] | None = None
    # 検出結果を保持
    scene_result: Any = None
    score_result: Any = None
    player_result: Any = None


class EnhancedGamePipeline(GamePipeline):
    """拡張ゲーム統合パイプラインクラス"""

    def __init__(
        self,
        game_id: str = "default_game",
        enable_scene_detection: bool = True,
        enable_score_reading: bool = True,
        enable_player_detection: bool = True,
    ):
        """
        拡張パイプラインを初期化

        Args:
            game_id: ゲームID
            enable_scene_detection: シーン検出を有効化
            enable_score_reading: 点数読み取りを有効化
            enable_player_detection: プレイヤー検出を有効化
        """
        super().__init__(game_id)

        # 拡張機能の初期化
        self.scene_detector = SceneDetector() if enable_scene_detection else None
        self.score_reader = ScoreReader() if enable_score_reading else None
        self.player_detector = PlayerDetector() if enable_player_detection else None

        # 追加のコンポーネント
        from ..utils.config import ConfigManager

        config_manager = ConfigManager()
        self.tile_detector = TileDetector(config_manager)
        self.action_detector = ActionDetector()

        # 拡張状態管理
        self.current_scene_type = SceneType.UNKNOWN
        self.last_valid_scores: dict[str, int] | None = None
        self.round_boundaries: list[tuple[int, float, SceneType]] = []  # (frame, timestamp, type)

        self.logger.info("拡張ゲームパイプライン初期化完了")

    def process_frame(self, frame_data: dict[str, Any]) -> ProcessingResult:
        """
        フレームデータを処理（親クラスのインターフェース）

        Args:
            frame_data: フレーム検出データ

        Returns:
            ProcessingResult: 処理結果
        """
        # フレームデータから必要な情報を抽出
        frame = frame_data.get("frame")
        frame_number = frame_data.get("frame_number", self.total_frames_processed)
        timestamp = frame_data.get("timestamp", frame_number / 30.0)

        if frame is None:
            # フレームがない場合は親クラスのメソッドを使用
            return super().process_frame(frame_data)

        # 拡張処理を実行
        enhanced_result = self.process_frame_enhanced(frame, frame_number, timestamp)

        # 拡張結果を通常の結果に変換
        return ProcessingResult(
            success=enhanced_result.success,
            frame_number=enhanced_result.frame_number,
            actions_detected=enhanced_result.actions_detected,
            confidence=enhanced_result.confidence,
            processing_time=enhanced_result.processing_time,
            errors=enhanced_result.errors,
            warnings=enhanced_result.warnings,
            metadata=enhanced_result.__dict__,
        )

    def process_frame_enhanced(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float | None = None,
    ) -> EnhancedProcessingResult:
        """
        フレームを処理（拡張版）

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            拡張処理結果
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = frame_number / 30.0  # デフォルト30fps

        result = EnhancedProcessingResult(
            success=False,
            frame_number=frame_number,
            actions_detected=0,
            confidence=0.0,
            processing_time=0.0,
        )

        try:
            # 1. シーン検出
            scene_info = None
            if self.scene_detector:
                scene_info = self.scene_detector.detect_scene(frame, frame_number, timestamp)
                result.scene_type = scene_info.scene_type

                # ゲーム境界の記録
                if scene_info.is_game_boundary():
                    self.round_boundaries.append((frame_number, timestamp, scene_info.scene_type))
                    self.logger.info(
                        f"ゲーム境界検出: {scene_info.scene_type.value} at frame {frame_number}"
                    )

                # ゲームプレイ中でない場合はスキップ
                if scene_info.scene_type not in [SceneType.GAME_PLAY, SceneType.ROUND_START]:
                    result.warnings.append("ゲームプレイ中ではありません")
                    result.success = True
                    result.processing_time = time.time() - start_time
                    return result

            # 2. プレイヤー検出
            player_info = None
            if self.player_detector:
                player_info = self.player_detector.detect_players(frame, frame_number, timestamp)
                result.active_player = (
                    player_info.active_position.value if player_info.active_position else None
                )
                result.dealer_position = player_info.dealer_position.value
                result.round_info = {
                    "wind": player_info.round_wind,
                    "dealer": player_info.dealer_position.value,
                }

            # 3. 点数読み取り
            score_info = None
            if self.score_reader:
                score_info = self.score_reader.read_scores(frame, frame_number, timestamp)
                if score_info.scores:
                    scores = {s.player_position: s.score for s in score_info.scores}

                    # 妥当性チェック
                    if self._validate_scores(scores):
                        result.player_scores = scores
                        self.last_valid_scores = scores
                    else:
                        result.warnings.append("読み取った点数が無効です")
                        result.player_scores = self.last_valid_scores

            # 4. 牌検出
            detection_result = self.tile_detector.detect_tiles(frame)

            # 5. アクション検出
            if detection_result and detection_result.detections:
                # 検出結果をフレームデータ形式に変換
                frame_data = {
                    "detections": detection_result.detections,
                    "timestamp": timestamp,
                }
                action_result = self.action_detector.detect_actions(frame_data, frame_number)
                actions = action_result.actions if action_result else []

                # 6. ゲーム状態更新
                for action in actions:
                    # 現在の手番情報を使用してアクションを補正
                    if player_info and player_info.active_position:
                        action.player = self._position_to_player(player_info.active_position)

                    # アクションを適用
                    if self.state_tracker.apply_action(action, frame_number):
                        result.actions_detected += 1
                        self.logger.debug(
                            f"アクション適用: {action.action_type.name} by {action.player.name}"
                        )
                    else:
                        result.warnings.append(f"アクション適用失敗: {action.action_type.name}")

            # 7. 履歴記録
            self._record_enhanced_history(
                frame_number, timestamp, scene_info, player_info, score_info
            )

            # 統計更新
            self.total_frames_processed += 1
            self.successful_frames += 1
            self.consecutive_failures = 0

            result.success = True
            result.confidence = self._calculate_overall_confidence(
                scene_info, player_info, score_info, detection_result
            )

        except Exception as e:
            self.logger.error(f"フレーム処理エラー: {e}")
            result.errors.append(str(e))
            self.failed_frames += 1
            self.consecutive_failures += 1

        result.processing_time = time.time() - start_time
        self.processing_results.append(result)

        return result

    def process_video(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int | None = None,
        skip_frames: int = 1,
    ) -> dict[str, Any]:
        """
        動画全体を処理（拡張版）

        Args:
            video_path: 動画ファイルパス
            start_frame: 開始フレーム
            end_frame: 終了フレーム
            skip_frames: スキップフレーム数

        Returns:
            処理結果の統計
        """
        # まずシーン境界を検出
        if self.scene_detector:
            self.logger.info("シーン境界を検出中...")
            boundaries = self.scene_detector.detect_game_boundaries(video_path, sample_interval=30)

            # ゲーム開始・終了を自動設定
            for boundary in boundaries:
                if boundary.scene_type == SceneType.GAME_START and start_frame == 0:
                    start_frame = boundary.frame_number
                    self.logger.info(f"ゲーム開始を検出: frame {start_frame}")
                elif boundary.scene_type == SceneType.GAME_END and end_frame is None:
                    end_frame = boundary.frame_number
                    self.logger.info(f"ゲーム終了を検出: frame {end_frame}")

        # 動画処理の実装
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"動画を開けません: {video_path}")
            return {"success": False, "error": "Failed to open video"}

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if end_frame is None:
                end_frame = total_frames

            frame_count = 0
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count >= start_frame and frame_count % (skip_frames + 1) == 0:
                    timestamp = frame_count / fps
                    self.process_frame_enhanced(frame, frame_count, timestamp)

                frame_count += 1

        finally:
            cap.release()

        return self.get_enhanced_statistics()

    def _validate_scores(self, scores: dict[str, int]) -> bool:
        """点数の妥当性をチェック"""
        if len(scores) != 4:
            return False

        total = sum(scores.values())
        # 4人麻雀の標準的な点数範囲
        if not (80000 <= total <= 120000):
            return False

        # 各プレイヤーの点数が妥当な範囲内か
        return all(-50000 <= score <= 150000 for score in scores.values())

    def _position_to_player(self, position: DetectorPlayerPosition):
        """プレイヤー位置からプレイヤーオブジェクトを取得"""
        from ..game.player import PlayerPosition

        # DetectorPlayerPositionをPlayerPositionに変換
        position_map = {
            DetectorPlayerPosition.EAST: PlayerPosition.EAST,
            DetectorPlayerPosition.SOUTH: PlayerPosition.SOUTH,
            DetectorPlayerPosition.WEST: PlayerPosition.WEST,
            DetectorPlayerPosition.NORTH: PlayerPosition.NORTH,
        }
        game_position = position_map.get(position)
        if game_position:
            return self.game_state.players.get(game_position)
        return None

    def _calculate_overall_confidence(
        self, scene_info, player_info, score_info, detection_result
    ) -> float:
        """全体的な信頼度を計算"""
        confidences = []

        if scene_info:
            confidences.append(scene_info.confidence)

        if player_info and player_info.players:
            player_conf = sum(p.confidence for p in player_info.players) / len(player_info.players)
            confidences.append(player_conf)

        if score_info:
            confidences.append(score_info.total_confidence)

        if detection_result and detection_result.detections:
            det_conf = sum(d.confidence for d in detection_result.detections) / len(
                detection_result.detections
            )
            confidences.append(det_conf)

        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0

    def _record_enhanced_history(
        self, frame_number: int, timestamp: float, scene_info, player_info, score_info
    ):
        """拡張履歴情報を記録"""
        history_data: dict[str, Any] = {
            "frame_number": frame_number,
            "timestamp": timestamp,
        }

        if scene_info:
            history_data["scene"] = {
                "type": scene_info.scene_type.value,
                "confidence": scene_info.confidence,
            }

        if player_info:
            history_data["players"] = {
                "active": player_info.active_position.value
                if player_info.active_position
                else None,
                "dealer": player_info.dealer_position.value,
                "round_wind": player_info.round_wind,
            }

        if score_info and score_info.scores:
            history_data["scores"] = {s.player_position: s.score for s in score_info.scores}

        # 履歴データを保存（HistoryManagerにカスタムイベントメソッドがない場合は別の方法で保存）
        if hasattr(self.history_manager, "add_custom_event"):
            self.history_manager.add_custom_event("enhanced_frame_data", history_data)
        else:
            # 代替方法：メタデータとして保存
            if self.history_manager.current_game:
                self.history_manager.current_game.metadata[f"frame_{frame_number}"] = history_data

    def get_enhanced_statistics(self) -> dict[str, Any]:
        """拡張統計情報を取得"""
        # 基本統計情報
        base_stats = {
            "game_id": self.game_id,
            "total_frames": self.total_frames_processed,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "success_rate": self.successful_frames / max(self.total_frames_processed, 1),
            "pipeline_state": self.pipeline_state.value,
        }

        # 拡張情報を追加
        base_stats["enhanced"] = {
            "scene_boundaries": len(self.round_boundaries),
            "current_scene": self.current_scene_type.value,
            "last_valid_scores": self.last_valid_scores,
            "rounds_detected": self._count_rounds(),
        }

        return base_stats

    def _count_rounds(self) -> int:
        """検出された局数をカウント"""
        round_starts = [b for b in self.round_boundaries if b[2] == SceneType.ROUND_START]
        return len(round_starts)
