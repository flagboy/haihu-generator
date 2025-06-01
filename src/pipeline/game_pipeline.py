"""
ゲーム状態統合パイプライン
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging

from ..game.game_state import GameState, GamePhase
from ..game.player import PlayerPosition
from ..tracking.state_tracker import StateTracker
from ..tracking.history_manager import HistoryManager
from ..utils.tile_definitions import TileDefinitions
from ..utils.logger import get_logger


class PipelineState(Enum):
    """パイプライン状態"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ProcessingResult:
    """処理結果"""
    success: bool
    frame_number: int
    actions_detected: int
    confidence: float
    processing_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GamePipeline:
    """ゲーム状態統合パイプラインクラス"""
    
    def __init__(self, game_id: str = "default_game"):
        """
        パイプラインを初期化
        
        Args:
            game_id: ゲームID
        """
        self.game_id = game_id
        self.logger = get_logger(__name__)
        self.tile_definitions = TileDefinitions()
        
        # コアコンポーネント
        self.game_state = GameState()
        self.state_tracker = StateTracker(self.game_state)
        self.history_manager = HistoryManager()
        
        # パイプライン状態
        self.pipeline_state = PipelineState.IDLE
        self.processing_results: List[ProcessingResult] = []
        
        # 統計情報
        self.total_frames_processed = 0
        self.successful_frames = 0
        self.failed_frames = 0
        self.start_time: Optional[float] = None
        
        # 設定
        self.auto_recovery = True
        self.max_consecutive_failures = 10
        self.consecutive_failures = 0
        
        # プレイヤー名の設定
        self.player_names = {
            PlayerPosition.EAST: "Player1",
            PlayerPosition.SOUTH: "Player2", 
            PlayerPosition.WEST: "Player3",
            PlayerPosition.NORTH: "Player4"
        }
    
    def initialize_game(self, player_names: Optional[Dict[PlayerPosition, str]] = None) -> bool:
        """
        ゲームを初期化
        
        Args:
            player_names: プレイヤー名の辞書
            
        Returns:
            bool: 初期化に成功したかどうか
        """
        try:
            # プレイヤー名を設定
            if player_names:
                self.player_names = player_names
            
            # 履歴管理を開始
            self.history_manager.start_new_game(self.game_id, self.player_names)
            
            # ゲーム状態を初期化
            self.game_state.set_phase(GamePhase.WAITING)
            
            # 統計をリセット
            self.total_frames_processed = 0
            self.successful_frames = 0
            self.failed_frames = 0
            self.consecutive_failures = 0
            self.start_time = time.time()
            
            self.pipeline_state = PipelineState.IDLE
            self.logger.info(f"Game pipeline initialized for game: {self.game_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize game pipeline: {e}")
            self.pipeline_state = PipelineState.ERROR
            return False
    
    def process_frame(self, frame_data: Dict[str, Any]) -> ProcessingResult:
        """
        フレームデータを処理
        
        Args:
            frame_data: フレーム検出データ
            
        Returns:
            ProcessingResult: 処理結果
        """
        start_time = time.time()
        frame_number = frame_data.get('frame_number', self.total_frames_processed)
        
        self.pipeline_state = PipelineState.PROCESSING
        self.total_frames_processed += 1
        
        result = ProcessingResult(
            success=False,
            frame_number=frame_number,
            actions_detected=0,
            confidence=0.0,
            processing_time=0.0
        )
        
        try:
            # フレームデータの前処理
            processed_data = self._preprocess_frame_data(frame_data)
            
            # 状態追跡器で処理
            tracking_success = self.state_tracker.update_from_frame(processed_data)
            
            if tracking_success:
                # 検出された行動を履歴に追加
                self._update_history()
                
                # 結果を更新
                result.success = True
                result.confidence = self.state_tracker.get_current_confidence()
                result.actions_detected = len(self.game_state.pending_actions)
                
                self.successful_frames += 1
                self.consecutive_failures = 0
                
                self.logger.debug(f"Frame {frame_number} processed successfully")
                
            else:
                # 処理失敗
                result.errors.append("State tracking failed")
                self.failed_frames += 1
                self.consecutive_failures += 1
                
                self.logger.warning(f"Frame {frame_number} processing failed")
                
                # 自動回復を試行
                if self.auto_recovery and self.consecutive_failures >= 3:
                    self._attempt_recovery()
            
            # 連続失敗が多い場合はエラー状態に
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.pipeline_state = PipelineState.ERROR
                result.errors.append("Too many consecutive failures")
            else:
                self.pipeline_state = PipelineState.IDLE
            
        except Exception as e:
            result.errors.append(f"Processing exception: {str(e)}")
            self.failed_frames += 1
            self.consecutive_failures += 1
            self.pipeline_state = PipelineState.ERROR
            
            self.logger.error(f"Exception processing frame {frame_number}: {e}")
        
        # 処理時間を記録
        result.processing_time = time.time() - start_time
        
        # メタデータを追加
        result.metadata = {
            'game_phase': self.game_state.phase.value,
            'tracking_state': self.state_tracker.tracking_state.value,
            'total_processed': self.total_frames_processed,
            'success_rate': self.successful_frames / self.total_frames_processed
        }
        
        # 結果を履歴に追加
        self.processing_results.append(result)
        
        return result
    
    def _preprocess_frame_data(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        フレームデータの前処理
        
        Args:
            frame_data: 生のフレームデータ
            
        Returns:
            Dict[str, Any]: 前処理済みデータ
        """
        processed_data = frame_data.copy()
        
        # タイムスタンプの正規化
        if 'timestamp' not in processed_data:
            processed_data['timestamp'] = time.time()
        
        # フレーム番号の正規化
        if 'frame_number' not in processed_data:
            processed_data['frame_number'] = self.total_frames_processed
        
        # プレイヤー手牌データの検証
        if 'player_hands' in processed_data:
            processed_data['player_hands'] = self._validate_player_hands(
                processed_data['player_hands']
            )
        
        # 捨て牌データの検証
        if 'discarded_tiles' in processed_data:
            processed_data['discarded_tiles'] = self._validate_discarded_tiles(
                processed_data['discarded_tiles']
            )
        
        return processed_data
    
    def _validate_player_hands(self, player_hands: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """プレイヤー手牌データを検証"""
        validated_hands = {}
        
        for pos_str, tiles in player_hands.items():
            try:
                # プレイヤー位置の検証
                pos = PlayerPosition(int(pos_str))
                
                # 牌の検証
                valid_tiles = [
                    tile for tile in tiles 
                    if self.tile_definitions.is_valid_tile(tile)
                ]
                
                validated_hands[pos_str] = valid_tiles
                
            except (ValueError, KeyError):
                continue
        
        return validated_hands
    
    def _validate_discarded_tiles(self, discarded_tiles: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """捨て牌データを検証"""
        validated_discards = {}
        
        for pos_str, tiles in discarded_tiles.items():
            try:
                # プレイヤー位置の検証
                pos = PlayerPosition(int(pos_str))
                
                # 牌の検証
                valid_tiles = [
                    tile for tile in tiles 
                    if self.tile_definitions.is_valid_tile(tile)
                ]
                
                validated_discards[pos_str] = valid_tiles
                
            except (ValueError, KeyError):
                continue
        
        return validated_discards
    
    def _update_history(self):
        """履歴を更新"""
        # 適用された行動を履歴に追加
        applied_actions = self.game_state.apply_pending_actions()
        
        for action in applied_actions:
            self.history_manager.add_action(action)
    
    def _attempt_recovery(self):
        """自動回復を試行"""
        self.logger.info("Attempting automatic recovery...")
        
        try:
            # 状態追跡器をリセット
            self.state_tracker.reset()
            
            # 最近のスナップショットに戻す
            recent_snapshots = self.state_tracker.snapshots[-5:]
            if recent_snapshots:
                stable_snapshot = None
                for snapshot in reversed(recent_snapshots):
                    if snapshot.confidence > 0.8:
                        stable_snapshot = snapshot
                        break
                
                if stable_snapshot:
                    self.state_tracker.rollback_to_snapshot(stable_snapshot.frame_number)
                    self.consecutive_failures = 0
                    self.logger.info(f"Recovered to frame {stable_snapshot.frame_number}")
        
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
    
    def start_new_round(self, round_number: int, round_name: str, 
                       dealer: PlayerPosition) -> bool:
        """
        新しい局を開始
        
        Args:
            round_number: 局番号
            round_name: 局名
            dealer: 親のプレイヤー
            
        Returns:
            bool: 開始に成功したかどうか
        """
        try:
            # ゲーム状態をリセット
            self.game_state.reset_for_new_round()
            self.game_state.set_phase(GamePhase.DEALING)
            
            # 履歴管理で新しい局を開始
            self.history_manager.start_new_round(round_number, round_name, dealer)
            
            # 状態追跡器をリセット
            self.state_tracker.reset()
            
            self.logger.info(f"Started new round: {round_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start new round: {e}")
            return False
    
    def complete_current_round(self, result: Optional[Dict[str, Any]] = None,
                             scores: Optional[Dict[PlayerPosition, int]] = None) -> bool:
        """
        現在の局を完了
        
        Args:
            result: 局の結果
            scores: 各プレイヤーの点数
            
        Returns:
            bool: 完了に成功したかどうか
        """
        try:
            # 履歴管理で局を完了
            self.history_manager.complete_current_round(result, scores)
            
            # ゲーム状態を更新
            self.game_state.set_phase(GamePhase.FINISHED)
            
            self.logger.info("Round completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete round: {e}")
            return False
    
    def export_tenhou_json_record(self) -> Dict[str, Any]:
        """
        天鳳JSON形式でゲーム記録をエクスポート（専用最適化版）
        
        Returns:
            Dict[str, Any]: 天鳳JSON形式の牌譜データ
        """
        try:
            # 天鳳JSON形式で牌譜データを生成
            tenhou_data = self.history_manager.export_to_tenhou_json_format()
            
            # 統計情報を追加
            pipeline_stats = self.get_pipeline_statistics()
            tenhou_data['processing_metadata'] = {
                'pipeline_statistics': pipeline_stats,
                'export_timestamp': time.time(),
                'format_version': '1.0',
                'system_info': {
                    'total_frames_processed': self.total_frames_processed,
                    'success_rate': self.successful_frames / max(1, self.total_frames_processed),
                    'processing_time': time.time() - self.start_time if self.start_time else 0
                }
            }
            
            return tenhou_data
                
        except Exception as e:
            self.logger.error(f"天鳳JSON牌譜のエクスポートに失敗しました: {e}")
            return {}
    
    def export_game_record(self) -> str:
        """
        ゲーム記録を天鳳JSON形式でエクスポート
        
        Returns:
            str: 天鳳JSON形式の牌譜データ
        """
        try:
            tenhou_data = self.export_tenhou_json_record()
            return json.dumps(tenhou_data, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"ゲーム記録のエクスポートに失敗しました: {e}")
            return "{}"
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """パイプライン統計を取得"""
        processing_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'game_id': self.game_id,
            'pipeline_state': self.pipeline_state.value,
            'total_frames': self.total_frames_processed,
            'successful_frames': self.successful_frames,
            'failed_frames': self.failed_frames,
            'success_rate': self.successful_frames / max(1, self.total_frames_processed),
            'consecutive_failures': self.consecutive_failures,
            'processing_time': processing_time,
            'average_frame_time': processing_time / max(1, self.total_frames_processed),
            'game_statistics': self.game_state.get_game_summary(),
            'tracking_statistics': self.state_tracker.get_tracking_statistics(),
            'history_statistics': self.history_manager.get_game_summary()
        }
    
    def reset(self):
        """パイプラインをリセット"""
        self.game_state.reset_for_new_round()
        self.state_tracker.reset()
        self.history_manager.clear_history()
        
        self.pipeline_state = PipelineState.IDLE
        self.processing_results = []
        self.total_frames_processed = 0
        self.successful_frames = 0
        self.failed_frames = 0
        self.consecutive_failures = 0
        self.start_time = None
        
        self.logger.info("Pipeline reset completed")
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"GamePipeline({self.game_id}, {self.pipeline_state.value})"
    
    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (f"GamePipeline(game_id='{self.game_id}', "
                f"state={self.pipeline_state.value}, "
                f"frames={self.total_frames_processed}, "
                f"success_rate={self.successful_frames/max(1, self.total_frames_processed):.2f})")