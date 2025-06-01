"""
麻雀牌検出モジュール
YOLOv8ベースの物体検出を使用して麻雀牌を検出する
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


@dataclass
class DetectionResult:
    """検出結果を格納するデータクラス"""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class SimpleCNN(nn.Module):
    """基本的なCNNモデル（YOLOv8の代替として使用）"""

    def __init__(self, num_classes: int = 1):
        super().__init__()

        # 特徴抽出層
        self.features = nn.Sequential(
            # 第1ブロック
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第2ブロック
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第3ブロック
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第4ブロック
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 検出ヘッド（簡易版）
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # バウンディングボックス (4) + 信頼度 (1) + クラス (num_classes)
            nn.Linear(256, 4 + 1 + num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.detection_head(features)

        # 出力を分割
        bbox = output[:, :4]  # バウンディングボックス
        confidence = torch.sigmoid(output[:, 4:5])  # 信頼度
        class_logits = output[:, 5:]  # クラス分類

        return bbox, confidence, class_logits


class TileDetector:
    """麻雀牌検出クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # AI設定を取得
        self.ai_config = self.config.get_config().get("ai", {})
        self.detection_config = self.ai_config.get("detection", {})

        # モデル設定
        self.model_type = self.detection_config.get("model_type", "cnn")
        self.model_path = self.detection_config.get("model_path", "models/tile_detector.pt")
        self.confidence_threshold = self.detection_config.get("confidence_threshold", 0.5)
        self.nms_threshold = self.detection_config.get("nms_threshold", 0.4)
        self.input_size = tuple(self.detection_config.get("input_size", [640, 640]))

        # デバイス設定
        self.device = self._setup_device()

        # モデル初期化
        self.model = None
        self.transform = self._setup_transform()

        self.logger.info(f"TileDetector initialized with model_type: {self.model_type}")

    def _setup_device(self) -> torch.device:
        """デバイス設定"""
        gpu_enabled = self.config.get_config().get("system", {}).get("gpu_enabled", False)

        if gpu_enabled and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info("Using GPU for detection")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU for detection")

        return device

    def _setup_transform(self) -> transforms.Compose:
        """画像変換の設定"""
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self) -> bool:
        """モデルの読み込み"""
        try:
            if self.model_type == "yolo":
                # YOLOv8モデルの読み込み（ultralytics使用）
                try:
                    from ultralytics import YOLO

                    self.model = YOLO(self.model_path)
                    self.logger.info("YOLOv8 model loaded successfully")
                    return True
                except ImportError:
                    self.logger.warning("ultralytics not available, falling back to CNN")
                    self.model_type = "cnn"

            if self.model_type == "cnn":
                # 基本CNNモデルの読み込み
                self.model = SimpleCNN(num_classes=1)  # 牌検出のみ
                self.model.to(self.device)

                # 学習済みモデルがあれば読み込み
                try:
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint)
                    self.logger.info("Pre-trained CNN model loaded successfully")
                except FileNotFoundError:
                    self.logger.warning(f"Model file not found: {self.model_path}")
                    self.logger.info("Using randomly initialized model")

                self.model.eval()
                return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def detect_tiles(self, image: np.ndarray) -> list[DetectionResult]:
        """
        画像から麻雀牌を検出

        Args:
            image: 入力画像 (BGR format)

        Returns:
            検出結果のリスト
        """
        if self.model is None and not self.load_model():
            self.logger.error("Model not loaded")
            return []

        try:
            if self.model_type == "yolo":
                return self._detect_with_yolo(image)
            else:
                return self._detect_with_cnn(image)

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def _detect_with_yolo(self, image: np.ndarray) -> list[DetectionResult]:
        """YOLOv8を使用した検出"""
        results = self.model(image, conf=self.confidence_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    detection = DetectionResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(confidence),
                        class_id=class_id,
                        class_name="tile",  # 簡易版では全て"tile"
                    )
                    detections.append(detection)

        return detections

    def _detect_with_cnn(self, image: np.ndarray) -> list[DetectionResult]:
        """基本CNNを使用した検出"""
        # 画像前処理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            bbox, confidence, class_logits = self.model(input_tensor)

        # 結果の後処理
        detections = []
        confidence_val = confidence[0, 0].cpu().numpy()

        if confidence_val > self.confidence_threshold:
            # バウンディングボックスを画像サイズに変換
            h, w = image.shape[:2]
            bbox_normalized = bbox[0].cpu().numpy()

            x1 = int(bbox_normalized[0] * w)
            y1 = int(bbox_normalized[1] * h)
            x2 = int(bbox_normalized[2] * w)
            y2 = int(bbox_normalized[3] * h)

            detection = DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence_val),
                class_id=0,
                class_name="tile",
            )
            detections.append(detection)

        return detections

    def classify_tile_areas(
        self, image: np.ndarray, detections: list[DetectionResult]
    ) -> dict[str, list[DetectionResult]]:
        """
        検出された牌を手牌・捨て牌・鳴き牌に分類

        Args:
            image: 入力画像
            detections: 検出結果

        Returns:
            エリア別の検出結果
        """
        h, w = image.shape[:2]

        # 簡易的な位置ベース分類
        areas = {
            "hand_tiles": [],  # 手牌
            "discarded_tiles": [],  # 捨て牌
            "called_tiles": [],  # 鳴き牌
        }

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_y = (y1 + y2) / 2
            (x1 + x2) / 2

            # 画面下部を手牌エリアとする
            if center_y > h * 0.7:
                areas["hand_tiles"].append(detection)
            # 画面中央を捨て牌エリアとする
            elif center_y > h * 0.3:
                areas["discarded_tiles"].append(detection)
            # 画面上部を鳴き牌エリアとする
            else:
                areas["called_tiles"].append(detection)

        return areas

    def track_tile_movements(
        self, prev_detections: list[DetectionResult], curr_detections: list[DetectionResult]
    ) -> list[tuple[DetectionResult, DetectionResult]]:
        """
        牌の移動を追跡

        Args:
            prev_detections: 前フレームの検出結果
            curr_detections: 現フレームの検出結果

        Returns:
            移動ペアのリスト
        """
        movements = []

        for curr_det in curr_detections:
            best_match = None
            min_distance = float("inf")

            for prev_det in prev_detections:
                # 中心点間の距離を計算
                curr_center = self._get_bbox_center(curr_det.bbox)
                prev_center = self._get_bbox_center(prev_det.bbox)

                distance = np.sqrt(
                    (curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    best_match = prev_det

            # 閾値以下の場合のみマッチとする
            if best_match and min_distance < 50:  # ピクセル単位
                movements.append((best_match, curr_det))

        return movements

    def _get_bbox_center(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """バウンディングボックスの中心点を取得"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def visualize_detections(
        self, image: np.ndarray, detections: list[DetectionResult]
    ) -> np.ndarray:
        """
        検出結果を可視化

        Args:
            image: 入力画像
            detections: 検出結果

        Returns:
            可視化された画像
        """
        vis_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence

            # バウンディングボックスを描画
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 信頼度を表示
            label = f"tile: {confidence:.2f}"
            cv2.putText(
                vis_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        return vis_image
