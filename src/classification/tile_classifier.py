"""
麻雀牌分類モジュール
CNNベースの牌分類を行い、37種類の牌を識別する
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.tile_definitions import TileDefinitions


@dataclass
class ClassificationResult:
    """分類結果を格納するデータクラス"""

    tile_name: str
    confidence: float
    class_id: int
    probabilities: dict[str, float]


class TileClassificationCNN(nn.Module):
    """麻雀牌分類用CNNモデル"""

    def __init__(self, num_classes: int = 37):
        super(TileClassificationCNN, self).__init__()

        # 特徴抽出層
        self.features = nn.Sequential(
            # 第1ブロック
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第2ブロック
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第3ブロック
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第4ブロック
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


class ResNetBlock(nn.Module):
    """ResNetブロック"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TileResNet(nn.Module):
    """麻雀牌分類用ResNetモデル"""

    def __init__(self, num_classes: int = 37):
        super(TileResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetブロック
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class TileClassifier:
    """麻雀牌分類クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # 牌定義を初期化
        self.tile_definitions = TileDefinitions()

        # AI設定を取得
        self.ai_config = self.config.get_config().get("ai", {})
        self.classification_config = self.ai_config.get("classification", {})

        # モデル設定
        self.model_type = self.classification_config.get("model_type", "cnn")
        self.model_path = self.classification_config.get("model_path", "models/tile_classifier.pt")
        self.confidence_threshold = self.classification_config.get("confidence_threshold", 0.8)
        self.input_size = tuple(self.classification_config.get("input_size", [224, 224]))
        self.num_classes = self.classification_config.get("num_classes", 37)

        # デバイス設定
        self.device = self._setup_device()

        # モデル初期化
        self.model = None
        self.transform = self._setup_transform()

        # クラス名マッピング
        self.class_names = self._create_class_mapping()

        self.logger.info(f"TileClassifier initialized with model_type: {self.model_type}")

    def _setup_device(self) -> torch.device:
        """デバイス設定"""
        gpu_enabled = self.config.get_config().get("system", {}).get("gpu_enabled", False)

        if gpu_enabled and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info("Using GPU for classification")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU for classification")

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

    def _create_class_mapping(self) -> list[str]:
        """クラスIDから牌名へのマッピングを作成"""
        return self.tile_definitions.get_all_tiles()

    def load_model(self) -> bool:
        """モデルの読み込み"""
        try:
            if self.model_type == "resnet":
                self.model = TileResNet(num_classes=self.num_classes)
            else:  # cnn
                self.model = TileClassificationCNN(num_classes=self.num_classes)

            self.model.to(self.device)

            # 学習済みモデルがあれば読み込み
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.logger.info(f"Pre-trained {self.model_type} model loaded successfully")
            except FileNotFoundError:
                self.logger.warning(f"Model file not found: {self.model_path}")
                self.logger.info("Using randomly initialized model")

            self.model.eval()
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def classify_tile(self, image: np.ndarray) -> ClassificationResult:
        """
        牌画像を分類

        Args:
            image: 牌画像 (BGR format)

        Returns:
            分類結果
        """
        if self.model is None:
            if not self.load_model():
                self.logger.error("Model not loaded")
                return ClassificationResult("unknown", 0.0, -1, {})

        try:
            # 画像前処理
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            # 結果の後処理
            class_id = predicted.item()
            confidence_val = confidence.item()

            if class_id < len(self.class_names):
                tile_name = self.class_names[class_id]
            else:
                tile_name = "unknown"

            # 全クラスの確率を取得
            prob_dict = {}
            probs = probabilities[0].cpu().numpy()
            for i, prob in enumerate(probs):
                if i < len(self.class_names):
                    prob_dict[self.class_names[i]] = float(prob)

            return ClassificationResult(
                tile_name=tile_name,
                confidence=confidence_val,
                class_id=class_id,
                probabilities=prob_dict,
            )

        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return ClassificationResult("unknown", 0.0, -1, {})

    def classify_tiles_batch(self, images: list[np.ndarray]) -> list[ClassificationResult]:
        """
        複数の牌画像をバッチ処理で分類

        Args:
            images: 牌画像のリスト

        Returns:
            分類結果のリスト
        """
        if not images:
            return []

        if self.model is None:
            if not self.load_model():
                self.logger.error("Model not loaded")
                return [ClassificationResult("unknown", 0.0, -1, {}) for _ in images]

        try:
            # バッチ用テンソルを準備
            batch_tensors = []
            for image in images:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb_image)
                batch_tensors.append(tensor)

            batch_input = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_input)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)

            # 結果の後処理
            results = []
            for i in range(len(images)):
                class_id = predicted[i].item()
                confidence_val = confidences[i].item()

                if class_id < len(self.class_names):
                    tile_name = self.class_names[class_id]
                else:
                    tile_name = "unknown"

                # 全クラスの確率を取得
                prob_dict = {}
                probs = probabilities[i].cpu().numpy()
                for j, prob in enumerate(probs):
                    if j < len(self.class_names):
                        prob_dict[self.class_names[j]] = float(prob)

                results.append(
                    ClassificationResult(
                        tile_name=tile_name,
                        confidence=confidence_val,
                        class_id=class_id,
                        probabilities=prob_dict,
                    )
                )

            return results

        except Exception as e:
            self.logger.error(f"Batch classification failed: {e}")
            return [ClassificationResult("unknown", 0.0, -1, {}) for _ in images]

    def handle_occlusion(
        self, image: np.ndarray, occlusion_mask: np.ndarray | None = None
    ) -> ClassificationResult:
        """
        重なり牌の処理

        Args:
            image: 牌画像
            occlusion_mask: 遮蔽マスク（オプション）

        Returns:
            分類結果
        """
        # 基本的な前処理で遮蔽を軽減
        processed_image = self._preprocess_occluded_image(image, occlusion_mask)

        # 通常の分類を実行
        result = self.classify_tile(processed_image)

        # 信頼度が低い場合は複数の前処理を試行
        if result.confidence < self.confidence_threshold:
            alternative_results = []

            # 異なる前処理を試行
            for method in ["enhance_contrast", "denoise", "sharpen"]:
                alt_image = self._apply_enhancement(image, method)
                alt_result = self.classify_tile(alt_image)
                alternative_results.append(alt_result)

            # 最も信頼度の高い結果を選択
            best_result = max([result] + alternative_results, key=lambda x: x.confidence)
            return best_result

        return result

    def _preprocess_occluded_image(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """遮蔽された画像の前処理"""
        processed = image.copy()

        if mask is not None:
            # マスクされた領域をインペインティング
            processed = cv2.inpaint(processed, mask, 3, cv2.INPAINT_TELEA)

        # コントラスト強化
        processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=10)

        return processed

    def _apply_enhancement(self, image: np.ndarray, method: str) -> np.ndarray:
        """画像強化の適用"""
        if method == "enhance_contrast":
            return cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        elif method == "denoise":
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif method == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        else:
            return image

    def confidence_scoring(self, result: ClassificationResult) -> dict[str, float]:
        """
        信頼度スコアの詳細分析

        Args:
            result: 分類結果

        Returns:
            詳細な信頼度情報
        """
        scores = {
            "primary_confidence": result.confidence,
            "entropy": self._calculate_entropy(result.probabilities),
            "top_k_confidence": self._calculate_top_k_confidence(result.probabilities, k=3),
            "margin": self._calculate_margin(result.probabilities),
        }

        return scores

    def _calculate_entropy(self, probabilities: dict[str, float]) -> float:
        """確率分布のエントロピーを計算"""
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy

    def _calculate_top_k_confidence(self, probabilities: dict[str, float], k: int = 3) -> float:
        """上位k個の確率の合計を計算"""
        sorted_probs = sorted(probabilities.values(), reverse=True)
        return sum(sorted_probs[:k])

    def _calculate_margin(self, probabilities: dict[str, float]) -> float:
        """最高確率と2番目の確率の差を計算"""
        sorted_probs = sorted(probabilities.values(), reverse=True)
        if len(sorted_probs) >= 2:
            return sorted_probs[0] - sorted_probs[1]
        return sorted_probs[0] if sorted_probs else 0.0

    def get_tile_features(self, image: np.ndarray) -> np.ndarray:
        """
        牌画像から特徴量を抽出

        Args:
            image: 牌画像

        Returns:
            特徴量ベクトル
        """
        if self.model is None:
            if not self.load_model():
                return np.array([])

        try:
            # 画像前処理
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

            # 特徴量抽出（最後の全結合層の前まで）
            with torch.no_grad():
                if hasattr(self.model, "features"):
                    features = self.model.features(input_tensor)
                    features = torch.flatten(features, 1)
                else:
                    # ResNetの場合
                    x = input_tensor
                    x = F.relu(self.model.bn1(self.model.conv1(x)))
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    x = self.model.avgpool(x)
                    features = torch.flatten(x, 1)

            return features.cpu().numpy().flatten()

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([])
