"""
AI分類機能のテスト
"""

import pytest
import numpy as np
import cv2
import torch
from unittest.mock import Mock, patch

from src.classification.tile_classifier import TileClassifier, ClassificationResult, TileClassificationCNN, TileResNet
from src.utils.config import ConfigManager


class TestTileClassifier:
    """TileClassifierクラスのテスト"""
    
    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        config = {
            'ai': {
                'classification': {
                    'model_type': 'cnn',
                    'model_path': 'models/tile_classifier.pt',
                    'confidence_threshold': 0.8,
                    'input_size': [224, 224],
                    'num_classes': 37
                }
            },
            'system': {
                'gpu_enabled': False
            }
        }
        
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_config.return_value = config
        return mock_config
        
    @pytest.fixture
    def classifier(self, config_manager):
        """TileClassifierのフィクスチャ"""
        return TileClassifier(config_manager)
        
    @pytest.fixture
    def sample_tile_image(self):
        """サンプル牌画像のフィクスチャ"""
        # 224x224のダミー牌画像を作成
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return image
        
    @pytest.fixture
    def sample_tile_images(self):
        """複数のサンプル牌画像のフィクスチャ"""
        images = []
        for _ in range(5):
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            images.append(image)
        return images
        
    def test_classifier_initialization(self, classifier):
        """分類器の初期化テスト"""
        assert classifier.model_type == 'cnn'
        assert classifier.confidence_threshold == 0.8
        assert classifier.input_size == (224, 224)
        assert classifier.num_classes == 37
        assert len(classifier.class_names) == 37
        
    def test_create_class_mapping(self, classifier):
        """クラスマッピング作成テスト"""
        class_names = classifier.class_names
        
        # 37種類の牌が含まれることを確認
        assert len(class_names) == 37
        
        # 萬子が含まれることを確認
        manzu_tiles = [tile for tile in class_names if tile.endswith('m')]
        assert len(manzu_tiles) >= 9
        
        # 筒子が含まれることを確認
        pinzu_tiles = [tile for tile in class_names if tile.endswith('p')]
        assert len(pinzu_tiles) >= 9
        
        # 索子が含まれることを確認
        souzu_tiles = [tile for tile in class_names if tile.endswith('s')]
        assert len(souzu_tiles) >= 9
        
    def test_load_model_cnn(self, classifier):
        """CNNモデル読み込みテスト"""
        result = classifier.load_model()
        assert result is True
        assert classifier.model is not None
        assert isinstance(classifier.model, TileClassificationCNN)
        
    def test_load_model_resnet(self, config_manager):
        """ResNetモデル読み込みテスト"""
        config_manager.get_config.return_value['ai']['classification']['model_type'] = 'resnet'
        classifier = TileClassifier(config_manager)
        
        result = classifier.load_model()
        assert result is True
        assert classifier.model is not None
        assert isinstance(classifier.model, TileResNet)
        
    def test_classify_tile(self, classifier, sample_tile_image):
        """牌分類テスト"""
        # モデルを読み込み
        classifier.load_model()
        
        # 分類実行
        result = classifier.classify_tile(sample_tile_image)
        
        # 結果の型チェック
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.tile_name, str)
        assert 0 <= result.confidence <= 1
        assert result.class_id >= -1
        assert isinstance(result.probabilities, dict)
        
    def test_classify_tiles_batch(self, classifier, sample_tile_images):
        """バッチ分類テスト"""
        # モデルを読み込み
        classifier.load_model()
        
        # バッチ分類実行
        results = classifier.classify_tiles_batch(sample_tile_images)
        
        # 結果の数が入力と一致することを確認
        assert len(results) == len(sample_tile_images)
        
        # 各結果の型チェック
        for result in results:
            assert isinstance(result, ClassificationResult)
            assert isinstance(result.tile_name, str)
            assert 0 <= result.confidence <= 1
            
    def test_classify_tiles_batch_empty(self, classifier):
        """空のバッチ分類テスト"""
        results = classifier.classify_tiles_batch([])
        assert results == []
        
    def test_handle_occlusion(self, classifier, sample_tile_image):
        """遮蔽処理テスト"""
        # モデルを読み込み
        classifier.load_model()
        
        # 遮蔽マスクを作成
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:100, 50:100] = 255  # 一部を遮蔽
        
        # 遮蔽処理実行
        result = classifier.handle_occlusion(sample_tile_image, mask)
        
        # 結果の型チェック
        assert isinstance(result, ClassificationResult)
        
    def test_confidence_scoring(self, classifier):
        """信頼度スコア計算テスト"""
        # ダミー分類結果を作成
        probabilities = {
            '1m': 0.7,
            '2m': 0.2,
            '3m': 0.1
        }
        
        result = ClassificationResult(
            tile_name='1m',
            confidence=0.7,
            class_id=0,
            probabilities=probabilities
        )
        
        scores = classifier.confidence_scoring(result)
        
        # スコアの構造チェック
        assert 'primary_confidence' in scores
        assert 'entropy' in scores
        assert 'top_k_confidence' in scores
        assert 'margin' in scores
        
        # 値の範囲チェック
        assert scores['primary_confidence'] == 0.7
        assert scores['entropy'] > 0
        assert scores['top_k_confidence'] <= 1.0
        assert scores['margin'] >= 0
        
    def test_calculate_entropy(self, classifier):
        """エントロピー計算テスト"""
        probabilities = {'1m': 0.5, '2m': 0.3, '3m': 0.2}
        entropy = classifier._calculate_entropy(probabilities)
        
        assert entropy > 0
        assert entropy < 10  # 合理的な範囲
        
    def test_calculate_top_k_confidence(self, classifier):
        """Top-K信頼度計算テスト"""
        probabilities = {'1m': 0.5, '2m': 0.3, '3m': 0.2}
        top_k = classifier._calculate_top_k_confidence(probabilities, k=2)
        
        assert top_k == 0.8  # 0.5 + 0.3
        
    def test_calculate_margin(self, classifier):
        """マージン計算テスト"""
        probabilities = {'1m': 0.5, '2m': 0.3, '3m': 0.2}
        margin = classifier._calculate_margin(probabilities)
        
        assert margin == 0.2  # 0.5 - 0.3
        
    def test_get_tile_features(self, classifier, sample_tile_image):
        """特徴量抽出テスト"""
        # モデルを読み込み
        classifier.load_model()
        
        # 特徴量抽出実行
        features = classifier.get_tile_features(sample_tile_image)
        
        # 結果の型チェック
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1  # 1次元ベクトル
        assert len(features) > 0
        
    def test_preprocess_occluded_image(self, classifier, sample_tile_image):
        """遮蔽画像前処理テスト"""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:100, 50:100] = 255
        
        processed = classifier._preprocess_occluded_image(sample_tile_image, mask)
        
        # 画像サイズが変わらないことを確認
        assert processed.shape == sample_tile_image.shape
        
    def test_apply_enhancement(self, classifier, sample_tile_image):
        """画像強化テスト"""
        methods = ['enhance_contrast', 'denoise', 'sharpen']
        
        for method in methods:
            enhanced = classifier._apply_enhancement(sample_tile_image, method)
            
            # 画像サイズが変わらないことを確認
            assert enhanced.shape == sample_tile_image.shape


class TestTileClassificationCNN:
    """TileClassificationCNNモデルのテスト"""
    
    @pytest.fixture
    def model(self):
        """TileClassificationCNNモデルのフィクスチャ"""
        return TileClassificationCNN(num_classes=37)
        
    def test_model_initialization(self, model):
        """モデル初期化テスト"""
        assert isinstance(model, TileClassificationCNN)
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
        
    def test_model_forward(self, model):
        """モデル順伝播テスト"""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        output = model(input_tensor)
        
        # 出力形状チェック
        assert output.shape == (batch_size, 37)
        
    def test_model_parameters(self, model):
        """モデルパラメータテスト"""
        params = list(model.parameters())
        assert len(params) > 0
        
        for param in params:
            assert param.requires_grad


class TestTileResNet:
    """TileResNetモデルのテスト"""
    
    @pytest.fixture
    def model(self):
        """TileResNetモデルのフィクスチャ"""
        return TileResNet(num_classes=37)
        
    def test_model_initialization(self, model):
        """モデル初期化テスト"""
        assert isinstance(model, TileResNet)
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        assert hasattr(model, 'layer4')
        assert hasattr(model, 'fc')
        
    def test_model_forward(self, model):
        """モデル順伝播テスト"""
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        output = model(input_tensor)
        
        # 出力形状チェック
        assert output.shape == (batch_size, 37)
        
    def test_make_layer(self, model):
        """レイヤー作成テスト"""
        layer = model._make_layer(64, 128, 2, stride=2)
        
        # レイヤーが正しく作成されることを確認
        assert isinstance(layer, torch.nn.Sequential)
        assert len(layer) == 2


class TestClassificationResult:
    """ClassificationResultデータクラスのテスト"""
    
    def test_classification_result_creation(self):
        """ClassificationResult作成テスト"""
        probabilities = {'1m': 0.7, '2m': 0.2, '3m': 0.1}
        
        result = ClassificationResult(
            tile_name='1m',
            confidence=0.7,
            class_id=0,
            probabilities=probabilities
        )
        
        assert result.tile_name == '1m'
        assert result.confidence == 0.7
        assert result.class_id == 0
        assert result.probabilities == probabilities
        
    def test_classification_result_equality(self):
        """ClassificationResult等価性テスト"""
        probabilities = {'1m': 0.7, '2m': 0.2, '3m': 0.1}
        
        result1 = ClassificationResult(
            tile_name='1m',
            confidence=0.7,
            class_id=0,
            probabilities=probabilities
        )
        
        result2 = ClassificationResult(
            tile_name='1m',
            confidence=0.7,
            class_id=0,
            probabilities=probabilities
        )
        
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__])