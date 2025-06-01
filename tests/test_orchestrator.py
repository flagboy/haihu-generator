"""
VideoProcessingOrchestratorのテスト
"""

import pytest
import tempfile
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import time

from src.integration.orchestrator import (
    VideoProcessingOrchestrator, 
    ProcessingOptions, 
    ProcessingResult
)
from src.utils.config import ConfigManager


class TestVideoProcessingOrchestrator:
    """VideoProcessingOrchestratorのテストクラス"""
    
    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのモック"""
        config_manager = Mock(spec=ConfigManager)
        config_manager.config = {
            'system': {
                'max_workers': 4,
                'constants': {
                    'default_batch_size': 32
                }
            },
            'performance': {
                'processing': {
                    'chunk_size': 8
                }
            }
        }
        return config_manager
    
    @pytest.fixture
    def mock_components(self):
        """モックコンポーネント"""
        # VideoProcessor
        video_processor = Mock()
        video_processor.extract_frames.return_value = {
            'success': True,
            'frames': [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
        }
        
        # AIPipeline
        ai_pipeline = Mock()
        ai_result = Mock()
        ai_result.frame_results = [
            {
                'frame_id': i,
                'classifications': [
                    (Mock(bbox=[10, 10, 50, 50]), Mock(label='1m', confidence=0.9))
                    for _ in range(3)
                ]
            }
            for i in range(10)
        ]
        ai_pipeline.process_frames_batch.return_value = ai_result
        
        # GamePipeline
        game_pipeline = Mock()
        game_pipeline.process_game_data.return_value = Mock(
            get_statistics=lambda: {'rounds': 1, 'actions': 20}
        )
        
        return video_processor, ai_pipeline, game_pipeline
    
    @pytest.fixture
    def orchestrator(self, config_manager, mock_components):
        """オーケストレーターのフィクスチャ"""
        video_processor, ai_pipeline, game_pipeline = mock_components
        return VideoProcessingOrchestrator(
            config_manager, video_processor, ai_pipeline, game_pipeline
        )
    
    def test_initialization(self, orchestrator):
        """初期化テスト"""
        assert orchestrator is not None
        assert orchestrator.video_processor is not None
        assert orchestrator.ai_pipeline is not None
        assert orchestrator.game_pipeline is not None
        assert orchestrator.system_config is not None
        assert orchestrator.performance_config is not None
    
    def test_process_video_success(self, orchestrator):
        """正常な動画処理テスト"""
        options = ProcessingOptions(
            enable_optimization=True,
            enable_validation=True,
            batch_size=5
        )
        
        # 処理実行
        result = orchestrator.process_video('/path/to/video.mp4', options)
        
        # 結果検証
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.frame_count == 10
        assert result.detected_tiles > 0
        assert result.processing_time > 0
        assert result.game_data is not None
        assert 'total_detections' in result.statistics
        
        # コンポーネントが呼び出されたことを確認
        orchestrator.video_processor.extract_frames.assert_called_once()
        orchestrator.ai_pipeline.process_frames_batch.assert_called()
        orchestrator.game_pipeline.process_game_data.assert_called()
    
    def test_process_video_frame_extraction_failure(self, orchestrator):
        """フレーム抽出失敗時のテスト"""
        # フレーム抽出を失敗させる
        orchestrator.video_processor.extract_frames.return_value = {
            'success': False,
            'error': 'Cannot open video file'
        }
        
        options = ProcessingOptions()
        result = orchestrator.process_video('/path/to/invalid.mp4', options)
        
        # 結果検証
        assert result.success is False
        assert len(result.errors) > 0
        assert 'Cannot open video file' in result.errors[0]
        assert result.frame_count == 0
        assert result.detected_tiles == 0
    
    def test_process_frames_with_ai(self, orchestrator):
        """AI処理のテスト"""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]
        options = ProcessingOptions(batch_size=5)
        
        # AI処理実行
        results = orchestrator._process_frames_with_ai(frames, options)
        
        # バッチ処理が適切に呼び出されたか確認
        assert orchestrator.ai_pipeline.process_frames_batch.call_count == 4  # 20フレーム / 5バッチサイズ
        assert len(results) == 20
    
    def test_process_game_state(self, orchestrator):
        """ゲーム状態処理のテスト"""
        # AI結果のモック
        ai_results = [
            {
                'frame_id': i,
                'classifications': [
                    (Mock(bbox=[10, 10, 50, 50]), Mock(label='1m', confidence=0.9))
                    for _ in range(2)
                ]
            }
            for i in range(5)
        ]
        
        options = ProcessingOptions()
        game_results = orchestrator._process_game_state(ai_results, options)
        
        # GamePipelineが呼び出されたことを確認
        orchestrator.game_pipeline.process_game_data.assert_called_once()
        assert game_results is not None
    
    def test_convert_ai_to_game_data(self, orchestrator):
        """AIデータからゲームデータへの変換テスト"""
        ai_result = {
            'frame_id': 42,
            'timestamp': 123.456,
            'classifications': [
                (Mock(bbox=[100, 200, 150, 250]), Mock(label='5m', confidence=0.95)),
                (Mock(bbox=[500, 600, 550, 650]), Mock(label='1p', confidence=0.85))
            ]
        }
        
        game_data = orchestrator._convert_ai_to_game_data(ai_result)
        
        # 変換結果の検証
        assert game_data['frame_id'] == 42
        assert game_data['timestamp'] == 123.456
        assert len(game_data['tiles']) == 2
        assert game_data['tiles'][0]['type'] == '5m'
        assert game_data['tiles'][0]['confidence'] == 0.95
        assert game_data['tiles'][0]['area'] in ['player_top', 'player_bottom', 'player_left', 'player_right', 'center']
    
    def test_classify_tile_area(self, orchestrator):
        """牌領域分類のテスト"""
        # 各領域のテストケース
        test_cases = [
            ([320, 100, 370, 150], 'player_top'),      # 上部
            ([320, 800, 370, 850], 'player_bottom'),   # 下部
            ([100, 400, 150, 450], 'player_left'),     # 左側
            ([1400, 400, 1450, 450], 'player_right'),  # 右側
            ([960, 540, 1010, 590], 'center')          # 中央
        ]
        
        for bbox, expected_area in test_cases:
            area = orchestrator._classify_tile_area(bbox)
            assert area == expected_area
    
    def test_collect_statistics(self, orchestrator):
        """統計情報収集のテスト"""
        ai_results = [
            {
                'frame_id': i,
                'classifications': [
                    (Mock(), Mock(confidence=0.8 + i * 0.02))
                    for _ in range(2)
                ]
            }
            for i in range(5)
        ]
        
        game_results = Mock()
        game_results.get_statistics.return_value = {
            'rounds': 2,
            'actions': 30
        }
        
        stats = orchestrator._collect_statistics(ai_results, game_results)
        
        # 統計情報の検証
        assert stats['total_frames'] == 5
        assert stats['frames_with_detections'] == 5
        assert stats['total_detections'] == 10
        assert 'average_confidence' in stats
        assert stats['average_confidence'] > 0
        assert 'game_statistics' in stats
        assert stats['game_statistics']['rounds'] == 2
    
    def test_processing_options(self):
        """ProcessingOptionsのテスト"""
        # デフォルト値
        options1 = ProcessingOptions()
        assert options1.enable_optimization is True
        assert options1.enable_validation is True
        assert options1.enable_gpu is True
        assert options1.batch_size is None
        assert options1.max_workers is None
        
        # カスタム値
        options2 = ProcessingOptions(
            enable_optimization=False,
            enable_validation=False,
            enable_gpu=False,
            batch_size=16,
            max_workers=2
        )
        assert options2.enable_optimization is False
        assert options2.enable_validation is False
        assert options2.enable_gpu is False
        assert options2.batch_size == 16
        assert options2.max_workers == 2
    
    def test_processing_result(self):
        """ProcessingResultのテスト"""
        # 成功結果
        result1 = ProcessingResult(
            success=True,
            video_path='/path/to/video.mp4',
            frame_count=100,
            detected_tiles=250,
            processing_time=45.6
        )
        assert result1.success is True
        assert result1.video_path == '/path/to/video.mp4'
        assert result1.frame_count == 100
        assert result1.detected_tiles == 250
        assert result1.processing_time == 45.6
        assert result1.statistics == {}  # デフォルト値
        assert result1.errors == []      # デフォルト値
        assert result1.warnings == []    # デフォルト値
        
        # 失敗結果
        result2 = ProcessingResult(
            success=False,
            video_path='/path/to/invalid.mp4',
            errors=['File not found', 'Invalid format']
        )
        assert result2.success is False
        assert len(result2.errors) == 2
        assert 'File not found' in result2.errors
    
    def test_error_handling_in_process_video(self, orchestrator):
        """process_videoのエラーハンドリングテスト"""
        # AI処理でエラーを発生させる
        orchestrator.ai_pipeline.process_frames_batch.side_effect = Exception("AI processing error")
        
        options = ProcessingOptions()
        result = orchestrator.process_video('/path/to/video.mp4', options)
        
        # エラーが適切に処理されることを確認
        assert result.success is False
        assert len(result.errors) > 0
        assert "AI processing error" in str(result.errors[0])
        assert result.processing_time > 0