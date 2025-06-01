"""
システム統合クラス - 天鳳JSON形式特化版
全コンポーネントを統合し、天鳳JSON形式での牌譜出力に最適化されたエンドツーエンド処理を提供
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.file_io import FileIOHelper
from ..video.video_processor import VideoProcessor
from ..pipeline.ai_pipeline import AIPipeline
from ..pipeline.game_pipeline import GamePipeline
from .orchestrator import VideoProcessingOrchestrator, ProcessingOptions, ProcessingResult
from .result_processor import ResultProcessor
from .statistics_collector import StatisticsCollector


@dataclass
class ProcessingProgress:
    """処理進捗"""
    current_step: str
    progress_percentage: float
    estimated_remaining_time: float
    current_frame: int
    total_frames: int
    processing_speed: float  # frames per second


@dataclass
class IntegrationResult:
    """統合処理結果"""
    success: bool
    output_path: str
    processing_time: float
    quality_score: Optional[float]
    frame_count: int
    detection_count: int
    classification_count: int
    error_messages: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]


class SystemIntegrator:
    """システム統合クラス - リファクタリング版"""
    
    def __init__(self, config_manager: ConfigManager,
                 video_processor: VideoProcessor,
                 ai_pipeline: AIPipeline,
                 game_pipeline: GamePipeline):
        """
        初期化
        
        Args:
            config_manager: 設定管理オブジェクト
            video_processor: 動画処理オブジェクト
            ai_pipeline: AIパイプラインオブジェクト
            game_pipeline: ゲームパイプラインオブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)
        
        # コアコンポーネント
        self.video_processor = video_processor
        self.ai_pipeline = ai_pipeline
        self.game_pipeline = game_pipeline
        
        # リファクタリングされたコンポーネント
        self.orchestrator = VideoProcessingOrchestrator(
            config_manager, video_processor, ai_pipeline, game_pipeline
        )
        self.result_processor = ResultProcessor(config_manager)
        self.statistics_collector = StatisticsCollector(config_manager)
        
        # 統合設定
        self.integration_config = self._load_integration_config()
        
        # 進捗追跡
        self.current_progress: Optional[ProcessingProgress] = None
        self.progress_callbacks: List[callable] = []
        
        self.logger.info("SystemIntegrator initialized with refactored components")
    
    def _load_integration_config(self) -> Dict[str, Any]:
        """天鳳JSON形式特化の統合設定を読み込み"""
        system_config = self.config.get_config().get('system', {})
        tenhou_config = self.config.get_config().get('tenhou_json', {})
        performance_config = self.config.get_config().get('performance', {})
        
        return {
            'max_workers': system_config.get('max_workers', mp.cpu_count()),
            'batch_size': performance_config.get('processing', {}).get('chunk_size', 8),
            'enable_progress_tracking': True,
            'auto_optimization': True,
            'quality_threshold': 70.0,
            'retry_failed_frames': True,
            'max_retries': 3,
            'frame_skip_threshold': 0.3,  # 信頼度が低い場合のフレームスキップ閾値
            # 天鳳JSON形式専用設定
            'output_format': 'tenhou_json',  # 固定値
            'tenhou_optimization': tenhou_config.get('optimization', {}),
            'intermediate_save': True,  # 中間結果の保存
            'cleanup_temp_files': True,
            # パフォーマンス最適化
            'enable_parallel_processing': performance_config.get('processing', {}).get('enable_parallel_processing', True),
            'enable_batch_optimization': performance_config.get('processing', {}).get('enable_batch_optimization', True)
        }
    
    def add_progress_callback(self, callback: callable):
        """進捗コールバックを追加"""
        self.progress_callbacks.append(callback)
    
    def _update_progress(self, step: str, percentage: float, 
                        current_frame: int = 0, total_frames: int = 0,
                        processing_speed: float = 0.0):
        """進捗を更新"""
        if not self.integration_config['enable_progress_tracking']:
            return
        
        # 残り時間を推定
        if processing_speed > 0 and current_frame < total_frames:
            remaining_frames = total_frames - current_frame
            estimated_remaining = remaining_frames / processing_speed
        else:
            estimated_remaining = 0.0
        
        self.current_progress = ProcessingProgress(
            current_step=step,
            progress_percentage=percentage,
            estimated_remaining_time=estimated_remaining,
            current_frame=current_frame,
            total_frames=total_frames,
            processing_speed=processing_speed
        )
        
        # コールバック実行
        for callback in self.progress_callbacks:
            try:
                callback(self.current_progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def process_video_complete(self, video_path: str, output_path: str,
                             enable_optimization: bool = True,
                             enable_validation: bool = True) -> IntegrationResult:
        """
        動画を完全処理（天鳳JSON形式専用エンドツーエンド処理）
        リファクタリング版：責務を分離したコンポーネントを使用
        
        Args:
            video_path: 入力動画パス
            output_path: 出力パス（.json拡張子）
            enable_optimization: 最適化を有効にするか
            enable_validation: 検証を有効にするか
            
        Returns:
            統合処理結果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting complete video processing: {video_path}")
            
            # 処理オプションを設定
            options = ProcessingOptions(
                enable_optimization=enable_optimization,
                enable_validation=enable_validation,
                enable_gpu=self.integration_config.get('enable_gpu', True),
                batch_size=self.integration_config.get('batch_size', 32),
                max_workers=self.integration_config.get('max_workers', 4)
            )
            
            # 1. オーケストレーターで動画処理を実行
            self._update_progress("動画処理実行中", 0.0)
            processing_result = self.orchestrator.process_video(video_path, options)
            
            if not processing_result.success:
                return IntegrationResult(
                    success=False,
                    output_path="",
                    processing_time=time.time() - start_time,
                    quality_score=None,
                    frame_count=0,
                    detection_count=0,
                    classification_count=0,
                    error_messages=processing_result.errors,
                    warnings=processing_result.warnings,
                    statistics={}
                )
            
            # 2. 結果を天鳳JSON形式で保存
            self._update_progress("天鳳JSON保存中", 70.0)
            
            game_data = processing_result.game_data
            metadata = {
                'video_path': video_path,
                'processing_time': processing_result.processing_time,
                'frame_count': processing_result.frame_count,
                'detected_tiles': processing_result.detected_tiles
            }
            
            self.result_processor.save_results(game_data, output_path, metadata)
            
            # 3. 品質検証（有効な場合）
            quality_score = None
            if enable_validation:
                self._update_progress("品質検証中", 85.0)
                try:
                    from ..validation.quality_validator import QualityValidator
                    validator = QualityValidator(self.config)
                    validation_result = validator.validate_record_file(output_path)
                    quality_score = validation_result.overall_score
                except Exception as e:
                    self.logger.warning(f"Quality validation failed: {e}")
                    processing_result.warnings.append(f"Quality validation failed: {e}")
            
            # 4. 統計情報収集
            self._update_progress("統計情報収集中", 95.0)
            statistics = self.statistics_collector.collect_statistics(
                processing_result,
                ai_results=None,  # オーケストレーター内部で処理済み
                game_results=game_data
            )
            
            # 統計情報をエクスポート
            self.result_processor.export_statistics(statistics, output_path)
            
            processing_time = time.time() - start_time
            
            # 結果作成
            result = IntegrationResult(
                success=True,
                output_path=output_path,
                processing_time=processing_time,
                quality_score=quality_score,
                frame_count=processing_result.frame_count,
                detection_count=processing_result.detected_tiles,
                classification_count=processing_result.detected_tiles,  # 検出と分類は同数
                error_messages=processing_result.errors,
                warnings=processing_result.warnings,
                statistics=statistics
            )
            
            self._update_progress("処理完了", 100.0)
            self.logger.info(f"Complete processing finished in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Complete video processing failed: {e}")
            return IntegrationResult(
                success=False,
                output_path="",
                processing_time=time.time() - start_time,
                quality_score=None,
                frame_count=0,
                detection_count=0,
                classification_count=0,
                error_messages=[str(e)],
                warnings=[],
                statistics={}
            )
    
    def process_batch(self, video_files: List[str], output_directory: str,
                     max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        バッチ処理（天鳳JSON形式専用）
        
        Args:
            video_files: 動画ファイルリスト
            output_directory: 出力ディレクトリ
            max_workers: 最大並列数
            
        Returns:
            バッチ処理結果
        """
        start_time = time.time()
        
        if max_workers is None:
            max_workers = self.integration_config['max_workers']
        
        try:
            self.logger.info(f"Starting batch processing: {len(video_files)} files")
            
            # 出力ディレクトリ作成
            Path(output_directory).mkdir(parents=True, exist_ok=True)
            
            results = []
            successful_count = 0
            failed_count = 0
            
            # 並列処理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # タスク投入
                future_to_video = {}
                for video_file in video_files:
                    video_name = Path(video_file).stem
                    output_path = os.path.join(
                        output_directory,
                        f"{video_name}_tenhou_record.json"  # 天鳳JSON形式固定
                    )
                    
                    future = executor.submit(
                        self.process_video_complete,
                        video_file, output_path
                    )
                    future_to_video[future] = video_file
                
                # 結果収集
                for future in as_completed(future_to_video):
                    video_file = future_to_video[future]
                    try:
                        result = future.result()
                        results.append({
                            'video_file': video_file,
                            'result': result
                        })
                        
                        if result.success:
                            successful_count += 1
                        else:
                            failed_count += 1
                            
                        self.logger.info(f"Completed: {video_file} ({'Success' if result.success else 'Failed'})")
                        
                    except Exception as e:
                        failed_count += 1
                        results.append({
                            'video_file': video_file,
                            'result': IntegrationResult(
                                success=False,
                                output_path="",
                                processing_time=0.0,
                                quality_score=None,
                                frame_count=0,
                                detection_count=0,
                                classification_count=0,
                                error_messages=[str(e)],
                                warnings=[],
                                statistics={}
                            )
                        })
                        self.logger.error(f"Failed: {video_file} - {e}")
            
            processing_time = time.time() - start_time
            success_rate = successful_count / len(video_files) if video_files else 0
            
            batch_result = {
                'success': True,
                'total_files': len(video_files),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'success_rate': success_rate,
                'processing_time': processing_time,
                'average_time_per_file': processing_time / len(video_files) if video_files else 0,
                'results': results
            }
            
            self.logger.info(f"Batch processing completed: {successful_count}/{len(video_files)} successful")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _convert_ai_to_game_data(self, ai_result) -> Dict[str, Any]:
        """AI結果をゲームデータに変換"""
        try:
            # 検出結果を手牌・捨て牌・鳴き牌に分類
            player_hands = {}
            discarded_tiles = {}
            called_tiles = {}
            
            # エリア別に牌を分類
            for area, detections in ai_result.tile_areas.items():
                tiles = []
                for detection in detections:
                    # 対応する分類結果を検索
                    for det, cls in ai_result.classifications:
                        if det == detection:
                            tiles.append(cls.tile_name)
                            break
                
                if area == 'hand_tiles':
                    player_hands['0'] = tiles  # プレイヤー0の手牌として仮設定
                elif area == 'discarded_tiles':
                    discarded_tiles['0'] = tiles
                elif area == 'called_tiles':
                    called_tiles['0'] = tiles
            
            return {
                'frame_number': ai_result.frame_id,
                'timestamp': time.time(),
                'player_hands': player_hands,
                'discarded_tiles': discarded_tiles,
                'called_tiles': called_tiles,
                'confidence': ai_result.confidence_scores.get('combined_confidence', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to convert AI result to game data: {e}")
            return {
                'frame_number': ai_result.frame_id,
                'timestamp': time.time(),
                'player_hands': {},
                'discarded_tiles': {},
                'called_tiles': {},
                'confidence': 0.0
            }
    
    def _save_tenhou_json_record(self, record_data: Dict[str, Any], output_path: str) -> None:
        """天鳳JSON牌譜データを保存 - リファクタリング版
        
        Note: このメソッドは互換性のために残されていますが、
              実際の処理はResultProcessorに委譲されます。
        """
        self.result_processor.save_results(record_data, output_path)
    
    def _convert_mock_to_serializable(self, data: Any) -> Any:
        """MockオブジェクトをJSONシリアライズ可能な形式に変換"""
        if hasattr(data, '_mock_name'):
            # Mockオブジェクトの場合は文字列表現に変換
            return f"Mock({data._mock_name})"
        elif isinstance(data, dict):
            return {key: self._convert_mock_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_mock_to_serializable(item) for item in data]
        else:
            return data
    
    # _remove_empty_fieldsメソッドはResultProcessorに移動済み
    
    # _compress_redundant_dataメソッドはResultProcessorに移動済み
    
    def _collect_statistics(self, ai_results: List[Any], game_results: List[Any]) -> Dict[str, Any]:
        """統計情報を収集 - リファクタリング版
        
        Note: このメソッドは互換性のために残されていますが、
              実際の処理はStatisticsCollectorに委譲されます。
        """
        # ダミーのProcessingResultを作成
        from .orchestrator import ProcessingResult
        dummy_result = ProcessingResult(
            success=True,
            video_path="",
            processing_time=0.0
        )
        
        return self.statistics_collector.collect_statistics(
            dummy_result,
            ai_results=ai_results,
            game_results=game_results
        )
    
    def _merge_consecutive_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """連続する同じアクションをマージ"""
        if not actions:
            return actions
        
        merged_actions = []
        current_action = actions[0].copy()
        
        for action in actions[1:]:
            # 同じタイプのアクションで連続している場合はマージ
            if (action.get('type') == current_action.get('type') and
                action.get('player') == current_action.get('player')):
                # カウントを増やすか、データをマージ
                if 'count' in current_action:
                    current_action['count'] += action.get('count', 1)
                else:
                    current_action['count'] = 2
            else:
                merged_actions.append(current_action)
                current_action = action.copy()
        
        merged_actions.append(current_action)
        return merged_actions
    
    def _collect_statistics(self, ai_results: List, game_results: List) -> Dict[str, Any]:
        """統計情報を収集"""
        try:
            # AI統計
            total_detections = sum(len(r.detections) for r in ai_results)
            total_classifications = sum(len(r.classifications) for r in ai_results)
            avg_confidence = np.mean([
                r.confidence_scores.get('combined_confidence', 0.0) 
                for r in ai_results
            ]) if ai_results else 0.0
            
            # ゲーム統計
            successful_frames = sum(1 for r in game_results if r.success)
            avg_processing_time = np.mean([r.processing_time for r in ai_results]) if ai_results else 0.0
            
            return {
                'ai_statistics': {
                    'total_frames': len(ai_results),
                    'total_detections': total_detections,
                    'total_classifications': total_classifications,
                    'average_detections_per_frame': total_detections / len(ai_results) if ai_results else 0,
                    'average_confidence': avg_confidence
                },
                'game_statistics': {
                    'total_frames': len(game_results),
                    'successful_frames': successful_frames,
                    'success_rate': successful_frames / len(game_results) if game_results else 0,
                    'average_processing_time': avg_processing_time
                },
                'performance_statistics': {
                    'frames_per_second': len(ai_results) / sum(r.processing_time for r in ai_results) if ai_results else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect statistics: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        try:
            import psutil
            
            return {
                'cpu_count': mp.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'integration_config': self.integration_config,
                'component_status': {
                    'video_processor': self.video_processor is not None,
                    'ai_pipeline': self.ai_pipeline is not None,
                    'game_pipeline': self.game_pipeline is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}
    
    def cleanup_temp_files(self, temp_directory: str = None):
        """一時ファイルをクリーンアップ"""
        if not self.integration_config['cleanup_temp_files']:
            return
        
        try:
            if temp_directory is None:
                temp_directory = self.config.get_config()['directories']['temp']
            
            temp_path = Path(temp_directory)
            if temp_path.exists():
                for file_path in temp_path.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                        self.logger.debug(f"Cleaned up temp file: {file_path}")
            
            self.logger.info("Temporary files cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    def get_current_progress(self) -> Optional[ProcessingProgress]:
        """現在の進捗を取得"""
        return self.current_progress
    
    def estimate_processing_time(self, video_path: str) -> Dict[str, Any]:
        """処理時間を推定"""
        try:
            # 動画情報を取得
            video_info = self.video_processor.get_video_info(video_path)
            
            if not video_info['success']:
                return {'success': False, 'error': 'Failed to get video info'}
            
            # フレーム数から推定
            total_frames = video_info['frame_count']
            
            # 経験的な処理速度（フレーム/秒）
            # 設定から推定FPSを取得
            estimated_fps = self.config.get_config().get('system', {}).get('constants', {}).get('estimated_fps', 2.0)
            
            estimated_time = total_frames / estimated_fps
            
            return {
                'success': True,
                'estimated_time_seconds': estimated_time,
                'estimated_time_minutes': estimated_time / 60,
                'total_frames': total_frames,
                'estimated_processing_speed': estimated_fps
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate processing time: {e}")
            return {'success': False, 'error': str(e)}