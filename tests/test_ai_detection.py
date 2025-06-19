"""
AI検出機能のテスト
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Optional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.detection.tile_detector import DetectionResult, TileDetector
from src.utils.config import ConfigManager

if TORCH_AVAILABLE:
    from src.detection.tile_detector import SimpleCNN


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTileDetector:
    """TileDetectorクラスのテスト"""

    @pytest.fixture
    def config_manager(self):
        """設定管理オブジェクトのフィクスチャ"""
        config = {
            "ai": {
                "detection": {
                    "model_type": "cnn",
                    "model_path": "models/tile_detector.pt",
                    "confidence_threshold": 0.5,
                    "nms_threshold": 0.4,
                    "input_size": [640, 640],
                }
            },
            "system": {"gpu_enabled": False},
        }

        mock_config = Mock(spec=ConfigManager)
        mock_config.get_config.return_value = config
        return mock_config

    @pytest.fixture
    def detector(self, config_manager):
        """TileDetectorのフィクスチャ"""
        return TileDetector(config_manager)

    @pytest.fixture
    def sample_image(self):
        """サンプル画像のフィクスチャ"""
        # 640x480のダミー画像を作成
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image

    def test_detector_initialization(self, detector):
        """検出器の初期化テスト"""
        assert detector.model_type == "cnn"
        assert detector.confidence_threshold == 0.5
        assert detector.input_size == (640, 640)
        assert detector.device.type in ["cpu", "cuda"]

    def test_setup_device_cpu(self, config_manager):
        """CPUデバイス設定テスト"""
        detector = TileDetector(config_manager)
        assert detector.device.type == "cpu"

    @patch("src.utils.device_utils.get_device_info")
    @patch("torch.cuda.is_available")
    def test_setup_device_gpu(self, mock_cuda_available, mock_get_device_info, config_manager):
        """GPUデバイス設定テスト"""
        mock_cuda_available.return_value = True
        # get_device_infoのモックを設定
        mock_get_device_info.return_value = {
            "cuda": {
                "available": True,
                "device_count": 1,
                "current_device": 0,
                "devices": [
                    {
                        "name": "Test GPU",
                        "total_memory": 8589934592,  # 8GB
                        "index": 0,
                        "major": 8,
                        "minor": 0,
                        "multi_processor_count": 30,
                    }
                ],
            },
            "mps": {"available": False},
            "cpu": {"available": True, "threads": 4},
        }
        config_manager.get_config.return_value["system"]["gpu_enabled"] = True

        detector = TileDetector(config_manager)
        # GPUが利用可能な場合
        assert detector.device.type == "cuda"

    def test_load_model_cnn(self, detector):
        """CNNモデル読み込みテスト"""
        # モデルファイルが存在しない場合でも初期化されることを確認
        result = detector.load_model()
        assert result is True
        assert detector.model is not None
        assert isinstance(detector.model, SimpleCNN)

    def test_detect_tiles_no_model(self, detector, sample_image):
        """モデル未読み込み時の検出テスト"""
        detector.model = None

        with patch.object(detector, "load_model", return_value=False):
            detections = detector.detect_tiles(sample_image)
            assert detections == []

    def test_detect_tiles_with_cnn(self, detector, sample_image):
        """CNN検出テスト"""
        # モデルを読み込み
        detector.load_model()

        # 検出実行
        detections = detector.detect_tiles(sample_image)

        # 結果の型チェック
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, DetectionResult)
            assert len(detection.bbox) == 4
            assert 0 <= detection.confidence <= 1

    def test_classify_tile_areas(self, detector, sample_image):
        """牌エリア分類テスト"""
        # ダミー検出結果を作成
        detections = [
            DetectionResult(
                bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"
            ),
            DetectionResult(
                bbox=(200, 200, 250, 250), confidence=0.7, class_id=0, class_name="tile"
            ),
            DetectionResult(
                bbox=(300, 400, 350, 450), confidence=0.9, class_id=0, class_name="tile"
            ),
        ]

        areas = detector.classify_tile_areas(sample_image, detections)

        # 結果の構造チェック
        assert "hand_tiles" in areas
        assert "discarded_tiles" in areas
        assert "called_tiles" in areas

        # 全ての検出結果がいずれかのエリアに分類されることを確認
        total_classified = sum(len(area_detections) for area_detections in areas.values())
        assert total_classified == len(detections)

    def test_track_tile_movements(self, detector):
        """牌移動追跡テスト"""
        prev_detections = [
            DetectionResult(
                bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"
            ),
            DetectionResult(
                bbox=(200, 200, 250, 250), confidence=0.7, class_id=0, class_name="tile"
            ),
        ]

        curr_detections = [
            DetectionResult(
                bbox=(110, 110, 160, 160), confidence=0.8, class_id=0, class_name="tile"
            ),
            DetectionResult(
                bbox=(300, 300, 350, 350), confidence=0.9, class_id=0, class_name="tile"
            ),
        ]

        movements = detector.track_tile_movements(prev_detections, curr_detections)

        # 結果の型チェック
        assert isinstance(movements, list)
        for prev_det, curr_det in movements:
            assert isinstance(prev_det, DetectionResult)
            assert isinstance(curr_det, DetectionResult)

    def test_get_bbox_center(self, detector):
        """バウンディングボックス中心点計算テスト"""
        bbox = (100, 100, 200, 200)
        center = detector._get_bbox_center(bbox)

        assert center == (150.0, 150.0)

    def test_visualize_detections(self, detector, sample_image):
        """検出結果可視化テスト"""
        detections = [
            DetectionResult(
                bbox=(100, 100, 150, 150), confidence=0.8, class_id=0, class_name="tile"
            )
        ]

        vis_image = detector.visualize_detections(sample_image, detections)

        # 画像サイズが変わらないことを確認
        assert vis_image.shape == sample_image.shape
        assert vis_image.dtype == sample_image.dtype


class TestSimpleCNN:
    """SimpleCNNモデルのテスト"""

    @pytest.fixture
    def model(self):
        """SimpleCNNモデルのフィクスチャ"""
        return SimpleCNN(num_classes=1)

    def test_model_initialization(self, model):
        """モデル初期化テスト"""
        assert isinstance(model, SimpleCNN)
        assert hasattr(model, "features")
        assert hasattr(model, "detection_head")

    def test_model_forward(self, model):
        """モデル順伝播テスト"""
        # ダミー入力
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 640, 640)

        # 順伝播実行
        bbox, confidence, class_logits = model(input_tensor)

        # 出力形状チェック
        assert bbox.shape == (batch_size, 4)
        assert confidence.shape == (batch_size, 1)
        assert class_logits.shape == (batch_size, 1)

        # 信頼度が0-1の範囲内であることを確認
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)

    def test_model_parameters(self, model):
        """モデルパラメータテスト"""
        # パラメータが存在することを確認
        params = list(model.parameters())
        assert len(params) > 0

        # 勾配計算が有効であることを確認
        for param in params:
            assert param.requires_grad

    def test_model_eval_mode(self, model):
        """モデル評価モードテスト"""
        model.eval()

        # 評価モードでの推論
        with torch.no_grad():
            input_tensor = torch.randn(1, 3, 640, 640)
            bbox, confidence, class_logits = model(input_tensor)

        # 出力が有効であることを確認
        assert not torch.isnan(bbox).any()
        assert not torch.isnan(confidence).any()
        assert not torch.isnan(class_logits).any()


class TestDetectionResult:
    """DetectionResultデータクラスのテスト"""

    def test_detection_result_creation(self):
        """DetectionResult作成テスト"""
        result = DetectionResult(
            bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="tile"
        )

        assert result.bbox == (100, 100, 200, 200)
        assert result.confidence == 0.8
        assert result.class_id == 0
        assert result.class_name == "tile"

    def test_detection_result_equality(self):
        """DetectionResult等価性テスト"""
        result1 = DetectionResult(
            bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="tile"
        )

        result2 = DetectionResult(
            bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, class_name="tile"
        )

        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__])
