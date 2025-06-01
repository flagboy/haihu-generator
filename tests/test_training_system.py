"""
教師データ作成システムのテスト
"""

import pytest
import tempfile
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.training.annotation_data import (
    BoundingBox, TileAnnotation, FrameAnnotation, VideoAnnotation, AnnotationData
)
from src.training.dataset_manager import DatasetManager
from src.training.frame_extractor import FrameExtractor, FrameQualityAnalyzer
from src.training.semi_auto_labeler import SemiAutoLabeler, PredictionResult
from src.utils.config import ConfigManager


class TestBoundingBox:
    """BoundingBoxクラスのテスト"""
    
    def test_bbox_creation(self):
        """バウンディングボックスの作成テスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 120
    
    def test_bbox_normalization(self):
        """座標の正規化テスト"""
        # 逆順の座標が正規化されることを確認
        bbox = BoundingBox(100, 120, 10, 20)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 120
    
    def test_bbox_properties(self):
        """プロパティの計算テスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        assert bbox.width == 90
        assert bbox.height == 100
        assert bbox.area == 9000
        assert bbox.center == (55.0, 70.0)
    
    def test_yolo_format_conversion(self):
        """YOLO形式変換テスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        center_x, center_y, width, height = bbox.to_yolo_format(200, 200)
        
        assert abs(center_x - 0.275) < 0.001  # (10+100)/2 / 200
        assert abs(center_y - 0.35) < 0.001   # (20+120)/2 / 200
        assert abs(width - 0.45) < 0.001      # 90 / 200
        assert abs(height - 0.5) < 0.001      # 100 / 200
    
    def test_from_yolo_format(self):
        """YOLO形式からの変換テスト"""
        bbox = BoundingBox.from_yolo_format(0.5, 0.5, 0.4, 0.6, 200, 200)
        assert bbox.x1 == 60   # (0.5 * 200) - (0.4 * 200 / 2)
        assert bbox.y1 == 40   # (0.5 * 200) - (0.6 * 200 / 2)
        assert bbox.x2 == 140  # 60 + (0.4 * 200)
        assert bbox.y2 == 160  # 40 + (0.6 * 200)


class TestTileAnnotation:
    """TileAnnotationクラスのテスト"""
    
    def test_tile_annotation_creation(self):
        """牌アノテーションの作成テスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        tile = TileAnnotation(
            tile_id="1m",
            bbox=bbox,
            confidence=0.9,
            area_type="hand"
        )
        
        assert tile.tile_id == "1m"
        assert tile.bbox == bbox
        assert tile.confidence == 0.9
        assert tile.area_type == "hand"
        assert tile.is_face_up is True
        assert tile.is_occluded is False
    
    def test_tile_annotation_validation(self):
        """バリデーションテスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        
        # 不正な信頼度
        with pytest.raises(ValueError):
            TileAnnotation(tile_id="1m", bbox=bbox, confidence=1.5)
        
        # 不正な遮蔽率
        with pytest.raises(ValueError):
            TileAnnotation(tile_id="1m", bbox=bbox, occlusion_ratio=1.5)
        
        # 不正なエリアタイプ
        with pytest.raises(ValueError):
            TileAnnotation(tile_id="1m", bbox=bbox, area_type="invalid")


class TestFrameAnnotation:
    """FrameAnnotationクラスのテスト"""
    
    def test_frame_annotation_creation(self):
        """フレームアノテーションの作成テスト"""
        bbox = BoundingBox(10, 20, 100, 120)
        tile = TileAnnotation(tile_id="1m", bbox=bbox)
        
        frame = FrameAnnotation(
            frame_id="test_frame",
            image_path="/path/to/image.jpg",
            image_width=1920,
            image_height=1080,
            timestamp=10.5,
            tiles=[tile]
        )
        
        assert frame.frame_id == "test_frame"
        assert frame.tile_count == 1
        assert frame.tile_types == ["1m"]
        assert len(frame.get_tiles_by_area("unknown")) == 1
        assert len(frame.get_high_confidence_tiles(0.8)) == 1


class TestAnnotationData:
    """AnnotationDataクラスのテスト"""
    
    def test_annotation_data_creation(self):
        """アノテーションデータの作成テスト"""
        annotation_data = AnnotationData()
        assert len(annotation_data.video_annotations) == 0
    
    def test_create_video_annotation(self):
        """動画アノテーション作成テスト"""
        annotation_data = AnnotationData()
        video_info = {
            "duration": 120.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080
        }
        
        video_id = annotation_data.create_video_annotation("/path/to/video.mp4", video_info)
        assert video_id in annotation_data.video_annotations
        
        video_annotation = annotation_data.video_annotations[video_id]
        assert video_annotation.duration == 120.0
        assert video_annotation.fps == 30.0
    
    def test_json_serialization(self):
        """JSON保存・読み込みテスト"""
        annotation_data = AnnotationData()
        video_info = {"duration": 120.0, "fps": 30.0, "width": 1920, "height": 1080}
        video_id = annotation_data.create_video_annotation("/path/to/video.mp4", video_info)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存テスト
            assert annotation_data.save_to_json(temp_path)
            
            # 読み込みテスト
            new_annotation_data = AnnotationData()
            assert new_annotation_data.load_from_json(temp_path)
            assert video_id in new_annotation_data.video_annotations
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDatasetManager:
    """DatasetManagerクラスのテスト"""
    
    @pytest.fixture
    def temp_config(self):
        """テスト用設定を作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                "training": {
                    "database_path": f"{temp_dir}/test.db",
                    "dataset_root": temp_dir
                }
            }
            
            config_manager = Mock(spec=ConfigManager)
            config_manager.get_config.return_value = config_data
            
            yield config_manager, temp_dir
    
    def test_dataset_manager_initialization(self, temp_config):
        """DatasetManagerの初期化テスト"""
        config_manager, temp_dir = temp_config
        
        dataset_manager = DatasetManager(config_manager)
        assert dataset_manager.dataset_root == Path(temp_dir)
        assert dataset_manager.db_path.exists()
    
    def test_save_and_load_annotation_data(self, temp_config):
        """アノテーションデータの保存・読み込みテスト"""
        config_manager, temp_dir = temp_config
        dataset_manager = DatasetManager(config_manager)
        
        # テストデータを作成
        annotation_data = AnnotationData()
        video_info = {"duration": 120.0, "fps": 30.0, "width": 1920, "height": 1080}
        video_id = annotation_data.create_video_annotation("/path/to/video.mp4", video_info)
        
        # 保存テスト
        assert dataset_manager.save_annotation_data(annotation_data)
        
        # 読み込みテスト
        loaded_data = dataset_manager.load_annotation_data(video_id)
        assert video_id in loaded_data.video_annotations


class TestFrameQualityAnalyzer:
    """FrameQualityAnalyzerクラスのテスト"""
    
    def test_quality_analysis(self):
        """品質分析テスト"""
        analyzer = FrameQualityAnalyzer()
        
        # テスト画像を作成（ランダムノイズ）
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        scores = analyzer.analyze_frame_quality(test_image)
        
        assert "sharpness" in scores
        assert "brightness" in scores
        assert "contrast" in scores
        assert "noise" in scores
        assert "overall" in scores
        
        # スコアが0-1の範囲内であることを確認
        for score in scores.values():
            assert 0.0 <= score <= 1.0


class TestFrameExtractor:
    """FrameExtractorクラスのテスト"""
    
    @pytest.fixture
    def temp_config(self):
        """テスト用設定を作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                "training": {
                    "frame_extraction": {
                        "min_quality_score": 0.5,
                        "max_frames_per_video": 100,
                        "frame_interval_seconds": 1.0,
                        "diversity_threshold": 0.3
                    },
                    "output_dir": temp_dir
                },
                "video": {
                    "frame_extraction": {"fps": 1, "output_format": "jpg", "quality": 95},
                    "preprocessing": {"target_width": 640, "target_height": 480, "normalize": True, "denoise": False}
                },
                "directories": {"temp": f"{temp_dir}/temp"}
            }
            
            config_manager = Mock(spec=ConfigManager)
            config_manager.get_config.return_value = config_data
            config_manager.get_video_config.return_value = config_data["video"]
            config_manager.get_directories.return_value = config_data["directories"]
            
            yield config_manager, temp_dir
    
    def test_frame_extractor_initialization(self, temp_config):
        """FrameExtractorの初期化テスト"""
        config_manager, temp_dir = temp_config
        
        extractor = FrameExtractor(config_manager)
        assert extractor.min_quality_score == 0.5
        assert extractor.max_frames_per_video == 100
        assert extractor.output_dir == Path(temp_dir)
    
    def test_frame_similarity_calculation(self, temp_config):
        """フレーム類似度計算テスト"""
        config_manager, temp_dir = temp_config
        extractor = FrameExtractor(config_manager)
        
        # 同じ画像
        image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image2 = image1.copy()
        
        similarity = extractor._calculate_frame_similarity(image1, image2)
        assert similarity > 0.9  # 同じ画像なので高い類似度
        
        # 異なる画像
        image3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        similarity = extractor._calculate_frame_similarity(image1, image3)
        assert 0.0 <= similarity <= 1.0


class TestSemiAutoLabeler:
    """SemiAutoLabelerクラスのテスト"""
    
    @pytest.fixture
    def temp_config(self):
        """テスト用設定を作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_data = {
                "training": {
                    "semi_auto_labeling": {
                        "confidence_threshold": 0.5,
                        "auto_area_classification": True,
                        "enable_occlusion_detection": True
                    },
                    "labeling_output_dir": temp_dir
                },
                "ai": {
                    "detection": {
                        "model_type": "cnn",
                        "model_path": "test_model.pt",
                        "confidence_threshold": 0.5,
                        "input_size": [640, 640]
                    },
                    "classification": {
                        "model_type": "cnn",
                        "model_path": "test_classifier.pt",
                        "confidence_threshold": 0.8,
                        "input_size": [224, 224],
                        "num_classes": 37
                    }
                },
                "system": {"gpu_enabled": False}
            }
            
            config_manager = Mock(spec=ConfigManager)
            config_manager.get_config.return_value = config_data
            
            yield config_manager, temp_dir
    
    @patch('src.training.semi_auto_labeler.TileDetector')
    @patch('src.training.semi_auto_labeler.TileClassifier')
    def test_semi_auto_labeler_initialization(self, mock_classifier, mock_detector, temp_config):
        """SemiAutoLabelerの初期化テスト"""
        config_manager, temp_dir = temp_config
        
        labeler = SemiAutoLabeler(config_manager)
        assert labeler.confidence_threshold == 0.5
        assert labeler.auto_area_classification is True
        assert labeler.output_dir == Path(temp_dir)
    
    def test_occlusion_detection(self, temp_config):
        """遮蔽検出テスト"""
        config_manager, temp_dir = temp_config
        
        with patch('src.training.semi_auto_labeler.TileDetector'), \
             patch('src.training.semi_auto_labeler.TileClassifier'):
            labeler = SemiAutoLabeler(config_manager)
            
            # テスト画像（エッジが少ない = 遮蔽されている可能性）
            low_edge_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
            is_occluded, occlusion_ratio = labeler._detect_occlusion(low_edge_image)
            
            assert isinstance(is_occluded, bool)
            assert 0.0 <= occlusion_ratio <= 1.0
    
    def test_prediction_statistics_calculation(self, temp_config):
        """予測統計計算テスト"""
        config_manager, temp_dir = temp_config
        
        with patch('src.training.semi_auto_labeler.TileDetector'), \
             patch('src.training.semi_auto_labeler.TileClassifier'):
            labeler = SemiAutoLabeler(config_manager)
            
            # テスト用予測結果を作成
            bbox = BoundingBox(10, 20, 100, 120)
            tile = TileAnnotation(tile_id="1m", bbox=bbox, confidence=0.8)
            frame = FrameAnnotation(
                frame_id="test",
                image_path="/test.jpg",
                image_width=640,
                image_height=480,
                timestamp=1.0,
                tiles=[tile]
            )
            
            prediction_result = PredictionResult(frame, [], [])
            stats = labeler._calculate_prediction_statistics([prediction_result])
            
            assert stats["total_frames"] == 1
            assert stats["total_tiles"] == 1
            assert "avg_confidence" in stats
            assert "tile_types" in stats


class TestIntegration:
    """統合テスト"""
    
    @pytest.fixture
    def temp_workspace(self):
        """テスト用ワークスペースを作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用画像を作成
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            image_path = Path(temp_dir) / "test_image.jpg"
            cv2.imwrite(str(image_path), test_image)
            
            yield temp_dir, str(image_path)
    
    def test_end_to_end_workflow(self, temp_workspace):
        """エンドツーエンドワークフローテスト"""
        temp_dir, image_path = temp_workspace
        
        # 1. アノテーションデータの作成
        annotation_data = AnnotationData()
        video_info = {"duration": 120.0, "fps": 30.0, "width": 640, "height": 480}
        video_id = annotation_data.create_video_annotation("/test/video.mp4", video_info)
        
        # 2. フレームアノテーションの追加
        frame_annotation = FrameAnnotation(
            frame_id="test_frame",
            image_path=image_path,
            image_width=640,
            image_height=480,
            timestamp=1.0,
            tiles=[]
        )
        
        annotation_data.add_frame_annotation(video_id, frame_annotation)
        
        # 3. JSON保存・読み込み
        json_path = Path(temp_dir) / "test_annotations.json"
        assert annotation_data.save_to_json(str(json_path))
        
        new_annotation_data = AnnotationData()
        assert new_annotation_data.load_from_json(str(json_path))
        
        # 4. 統計情報の確認
        stats = new_annotation_data.get_all_statistics()
        assert stats["video_count"] == 1
        assert stats["total_frames"] == 1


if __name__ == "__main__":
    pytest.main([__file__])