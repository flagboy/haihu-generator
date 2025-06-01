"""
AIパイプライン機能のテスト
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.pipeline.ai_pipeline import AIPipeline, PipelineResult, BatchProcessingConfig
from src.detection.tile_detector import DetectionResult
from src.classification.tile_classifier import ClassificationResult
from src.utils.config import ConfigManager


class TestAIPipeline:
    """AIPipelineクラスのテスト"""
    
    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        config = {
            'ai': {
                'detection': {
                    'model_type': 'cnn',
                    'confidence_threshold': 0.5
                },
                'classification': {
                    'model_type': 'cnn',
                    'num_classes': 37
                },
                'training': {
                    'batch_size': 8
                }
            },
            'system': {
                'max_workers': 4,
                'gpu_enabled': False
            }
        }
        
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_config.return_value = config
        return mock_config
        
    @pytest.fixture
    def pipeline(self, config_manager):
        """AIPipelineのフィクスチャ"""
        with patch('src.pipeline.ai_pipeline.TileDetector'), \
             patch('src.pipeline.ai_pipeline.TileClassifier'):
            return AIPipeline(config_manager)
            
    @pytest.fixture
    def sample_frame(self):
        """サンプルフレームのフィクスチャ"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    @pytest.fixture
    def sample_frames(self):
        """複数のサンプルフレームのフィクスチャ"""
        frames = []
        for _ in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        return frames
        
    @pytest.fixture
    def mock_detections(self):
        """モック検出結果のフィクスチャ"""
        return [
            DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"),
            DetectionResult(bbox=(200, 200, 250, 250), confidence=0.7, class_id=0, class_name="tile")
        ]
        
    @pytest.fixture
    def mock_classifications(self):
        """モック分類結果のフィクスチャ"""
        return [
            ClassificationResult(tile_name='1m', confidence=0.9, class_id=0, probabilities={'1m': 0.9}),
            ClassificationResult(tile_name='2p', confidence=0.8, class_id=1, probabilities={'2p': 0.8})
        ]
        
    def test_pipeline_initialization(self, pipeline):
        """パイプライン初期化テスト"""
        assert pipeline.detector is not None
        assert pipeline.classifier is not None
        assert isinstance(pipeline.batch_config, BatchProcessingConfig)
        assert pipeline.stats['total_frames'] == 0
        
    def test_setup_batch_config(self, pipeline):
        """バッチ設定初期化テスト"""
        config = pipeline.batch_config
        
        assert config.batch_size == 8
        assert config.max_workers == 4
        assert config.enable_parallel is True
        assert config.confidence_filter == 0.5
        
    def test_process_frame(self, pipeline, sample_frame, mock_detections, mock_classifications):
        """単一フレーム処理テスト"""
        # モックの設定
        pipeline.detector.detect_tiles.return_value = mock_detections
        pipeline.detector.classify_tile_areas.return_value = {
            'hand_tiles': mock_detections[:1],
            'discarded_tiles': mock_detections[1:],
            'called_tiles': []
        }
        pipeline.classifier.classify_tiles_batch.return_value = mock_classifications
        
        # フレーム処理実行
        result = pipeline.process_frame(sample_frame, frame_id=1)
        
        # 結果の型チェック
        assert isinstance(result, PipelineResult)
        assert result.frame_id == 1
        assert len(result.detections) <= len(mock_detections)
        assert len(result.classifications) <= len(mock_classifications)
        assert result.processing_time > 0
        assert isinstance(result.tile_areas, dict)
        assert isinstance(result.confidence_scores, dict)
        
    def test_process_frame_no_detections(self, pipeline, sample_frame):
        """検出結果なしのフレーム処理テスト"""
        # 検出結果なしの設定
        pipeline.detector.detect_tiles.return_value = []
        pipeline.detector.classify_tile_areas.return_value = {
            'hand_tiles': [],
            'discarded_tiles': [],
            'called_tiles': []
        }
        
        # フレーム処理実行
        result = pipeline.process_frame(sample_frame)
        
        # 結果チェック
        assert len(result.detections) == 0
        assert len(result.classifications) == 0
        assert result.processing_time > 0
        
    def test_process_frames_batch(self, pipeline, sample_frames):
        """バッチフレーム処理テスト"""
        # モックの設定
        pipeline.detector.detect_tiles.return_value = []
        pipeline.detector.classify_tile_areas.return_value = {
            'hand_tiles': [],
            'discarded_tiles': [],
            'called_tiles': []
        }
        
        # バッチ処理実行
        results = pipeline.process_frames_batch(sample_frames)
        
        # 結果の数が入力と一致することを確認
        assert len(results) == len(sample_frames)
        
        # 各結果の型チェック
        for i, result in enumerate(results):
            assert isinstance(result, PipelineResult)
            assert result.frame_id == i
            
    def test_process_frames_batch_empty(self, pipeline):
        """空のバッチ処理テスト"""
        results = pipeline.process_frames_batch([])
        assert results == []
        
    def test_extract_tile_images(self, pipeline, sample_frame, mock_detections):
        """牌画像切り出しテスト"""
        tile_images = pipeline._extract_tile_images(sample_frame, mock_detections)
        
        # 結果の数が検出結果と一致することを確認
        assert len(tile_images) == len(mock_detections)
        
        # 各画像の型チェック
        for image in tile_images:
            assert isinstance(image, np.ndarray)
            assert image.ndim == 3  # カラー画像
            assert image.shape[2] == 3  # BGR
            
    def test_extract_tile_images_invalid_bbox(self, pipeline, sample_frame):
        """無効なバウンディングボックスの切り出しテスト"""
        # 画像範囲外のバウンディングボックス
        invalid_detections = [
            DetectionResult(bbox=(-10, -10, 5, 5), confidence=0.8, class_id=0, class_name="tile"),
            DetectionResult(bbox=(1000, 1000, 1050, 1050), confidence=0.7, class_id=0, class_name="tile")
        ]
        
        tile_images = pipeline._extract_tile_images(sample_frame, invalid_detections)
        
        # ダミー画像が生成されることを確認
        assert len(tile_images) == len(invalid_detections)
        for image in tile_images:
            assert isinstance(image, np.ndarray)
            
    def test_calculate_confidence_scores(self, pipeline, mock_detections, mock_classifications):
        """信頼度スコア計算テスト"""
        classifications = list(zip(mock_detections, mock_classifications))
        
        scores = pipeline._calculate_confidence_scores(classifications)
        
        # スコアの構造チェック
        assert 'avg_detection_confidence' in scores
        assert 'avg_classification_confidence' in scores
        assert 'min_detection_confidence' in scores
        assert 'min_classification_confidence' in scores
        assert 'combined_confidence' in scores
        
        # 値の範囲チェック
        for score in scores.values():
            assert 0 <= score <= 1
            
    def test_calculate_confidence_scores_empty(self, pipeline):
        """空の信頼度スコア計算テスト"""
        scores = pipeline._calculate_confidence_scores([])
        assert scores == {}
        
    def test_filter_results(self, pipeline, mock_detections, mock_classifications):
        """結果フィルタリングテスト"""
        classifications = list(zip(mock_detections, mock_classifications))
        confidence_scores = {'combined_confidence': 0.7}
        
        filtered_detections, filtered_classifications = pipeline._filter_results(
            mock_detections, classifications, confidence_scores
        )
        
        # フィルタリング後の結果チェック
        assert len(filtered_detections) <= len(mock_detections)
        assert len(filtered_classifications) <= len(classifications)
        
        # 信頼度閾値以上の結果のみ残ることを確認
        for detection in filtered_detections:
            assert detection.confidence >= pipeline.batch_config.confidence_filter
            
    def test_post_process_results(self, pipeline):
        """結果後処理テスト"""
        # ダミー結果を作成
        results = [
            PipelineResult(
                frame_id=i,
                detections=[DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile")],
                classifications=[(
                    DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"),
                    ClassificationResult(tile_name='1m', confidence=0.9, class_id=0, probabilities={'1m': 0.9})
                )],
                processing_time=0.1,
                tile_areas={'hand_tiles': [], 'discarded_tiles': [], 'called_tiles': []},
                confidence_scores={}
            )
            for i in range(3)
        ]
        
        processed = pipeline.post_process_results(results)
        
        # 処理結果の構造チェック
        assert 'summary' in processed
        assert 'tile_statistics' in processed
        assert 'area_statistics' in processed
        assert 'confidence_statistics' in processed
        
        # サマリー情報チェック
        summary = processed['summary']
        assert summary['total_frames'] == 3
        assert summary['total_detections'] == 3
        assert summary['total_classifications'] == 3
        assert summary['total_processing_time'] > 0
        
    def test_post_process_results_empty(self, pipeline):
        """空の結果後処理テスト"""
        processed = pipeline.post_process_results([])
        assert processed == {}
        
    def test_update_stats(self, pipeline):
        """統計更新テスト"""
        initial_stats = pipeline.get_statistics()
        
        pipeline._update_stats(1, 2, 2, 0.1)
        
        updated_stats = pipeline.get_statistics()
        
        assert updated_stats['total_frames'] == initial_stats['total_frames'] + 1
        assert updated_stats['total_detections'] == initial_stats['total_detections'] + 2
        assert updated_stats['total_classifications'] == initial_stats['total_classifications'] + 2
        assert updated_stats['total_processing_time'] > initial_stats['total_processing_time']
        
    def test_get_statistics(self, pipeline):
        """統計取得テスト"""
        stats = pipeline.get_statistics()
        
        # 統計の構造チェック
        assert 'total_frames' in stats
        assert 'total_detections' in stats
        assert 'total_classifications' in stats
        assert 'total_processing_time' in stats
        assert 'average_processing_time' in stats
        
        # 初期値チェック
        assert stats['total_frames'] == 0
        assert stats['total_detections'] == 0
        assert stats['total_classifications'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['average_processing_time'] == 0.0
        
    def test_reset_statistics(self, pipeline):
        """統計リセットテスト"""
        # 統計を更新
        pipeline._update_stats(1, 2, 2, 0.1)
        
        # リセット実行
        pipeline.reset_statistics()
        
        # リセット後の統計チェック
        stats = pipeline.get_statistics()
        assert stats['total_frames'] == 0
        assert stats['total_detections'] == 0
        assert stats['total_classifications'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['average_processing_time'] == 0.0
        
    def test_visualize_results(self, pipeline, sample_frame, mock_detections, mock_classifications):
        """結果可視化テスト"""
        # ダミー結果を作成
        result = PipelineResult(
            frame_id=1,
            detections=mock_detections,
            classifications=list(zip(mock_detections, mock_classifications)),
            processing_time=0.1,
            tile_areas={},
            confidence_scores={}
        )
        
        vis_frame = pipeline.visualize_results(sample_frame, result)
        
        # 画像サイズが変わらないことを確認
        assert vis_frame.shape == sample_frame.shape
        assert vis_frame.dtype == sample_frame.dtype
        
    def test_save_results_json(self, pipeline, tmp_path):
        """JSON形式での結果保存テスト"""
        # ダミー結果を作成
        results = [
            PipelineResult(
                frame_id=1,
                detections=[DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile")],
                classifications=[(
                    DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"),
                    ClassificationResult(tile_name='1m', confidence=0.9, class_id=0, probabilities={'1m': 0.9})
                )],
                processing_time=0.1,
                tile_areas={},
                confidence_scores={}
            )
        ]
        
        output_path = tmp_path / "results.json"
        pipeline.save_results(results, str(output_path), format='json')
        
        # ファイルが作成されることを確認
        assert output_path.exists()
        
        # ファイル内容の確認
        import json
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
            
        assert len(saved_data) == 1
        assert saved_data[0]['frame_id'] == 1


class TestPipelineResult:
    """PipelineResultデータクラスのテスト"""
    
    def test_pipeline_result_creation(self):
        """PipelineResult作成テスト"""
        detections = [DetectionResult(bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile")]
        classifications = [(detections[0], ClassificationResult(tile_name='1m', confidence=0.9, class_id=0, probabilities={'1m': 0.9}))]
        
        result = PipelineResult(
            frame_id=1,
            detections=detections,
            classifications=classifications,
            processing_time=0.1,
            tile_areas={'hand_tiles': []},
            confidence_scores={'avg_confidence': 0.8}
        )
        
        assert result.frame_id == 1
        assert len(result.detections) == 1
        assert len(result.classifications) == 1
        assert result.processing_time == 0.1
        assert 'hand_tiles' in result.tile_areas
        assert 'avg_confidence' in result.confidence_scores


class TestBatchProcessingConfig:
    """BatchProcessingConfigデータクラスのテスト"""
    
    def test_batch_config_creation(self):
        """BatchProcessingConfig作成テスト"""
        config = BatchProcessingConfig(
            batch_size=16,
            max_workers=8,
            enable_parallel=False,
            confidence_filter=0.7
        )
        
        assert config.batch_size == 16
        assert config.max_workers == 8
        assert config.enable_parallel is False
        assert config.confidence_filter == 0.7
        
    def test_batch_config_defaults(self):
        """BatchProcessingConfigデフォルト値テスト"""
        config = BatchProcessingConfig()
        
        assert config.batch_size == 8
        assert config.max_workers == 4
        assert config.enable_parallel is True
        assert config.confidence_filter == 0.5


if __name__ == "__main__":
    pytest.main([__file__])