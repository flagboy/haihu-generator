"""
対局画面分類器

麻雀動画のフレームが対局画面かどうかを判定するCNNベースの分類器
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from ....utils.logger import LoggerMixin


class GameSceneClassifierModel(nn.Module):
    """対局画面分類用のCNNモデル"""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        初期化

        Args:
            num_classes: 分類クラス数（デフォルト: 2 - 対局/非対局）
            pretrained: 事前学習済みモデルを使用するか
        """
        super().__init__()

        # バックボーンモデル（EfficientNet-B0）
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # 最終層を置き換え
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        return self.backbone(x)


class GameSceneClassifier(LoggerMixin):
    """対局画面分類器"""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.8,
    ):
        """
        初期化

        Args:
            model_path: 学習済みモデルのパス
            device: 使用デバイス（None で自動選択）
            confidence_threshold: 信頼度閾値
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        # デバイス設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # モデル初期化
        self.model = GameSceneClassifierModel(num_classes=2, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # 学習済みモデルがあれば読み込み
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        # 画像前処理
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.logger.info(f"GameSceneClassifier初期化完了 (device: {self.device})")

    def load_model(self, model_path: str):
        """モデルを読み込み"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"モデルを読み込みました: {model_path}")
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")

    def save_model(self, save_path: str):
        """モデルを保存"""
        try:
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"モデルを保存しました: {save_path}")
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")

    def classify_frame(self, frame: np.ndarray | str) -> tuple[bool, float]:
        """
        フレームが対局画面かどうかを分類

        Args:
            frame: 画像データまたは画像ファイルパス

        Returns:
            (is_game_scene, confidence): 対局画面かどうかと信頼度
        """
        try:
            # 画像読み込み
            if isinstance(frame, str):
                image = cv2.imread(frame)
                if image is None:
                    raise ValueError(f"画像を読み込めません: {frame}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = frame
                if len(image.shape) == 2:  # グレースケール
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # BGRA
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3 and isinstance(frame, np.ndarray):
                    # OpenCVのBGRをRGBに変換
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 前処理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 推論
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)

                # クラス0: 非対局, クラス1: 対局
                game_scene_prob = probabilities[0, 1].item()
                is_game_scene = game_scene_prob >= self.confidence_threshold

            return is_game_scene, game_scene_prob

        except Exception as e:
            self.logger.error(f"フレーム分類エラー: {e}")
            return False, 0.0

    def classify_batch(self, frames: list[np.ndarray | str]) -> list[tuple[bool, float]]:
        """
        複数フレームをバッチ処理

        Args:
            frames: 画像データまたは画像ファイルパスのリスト

        Returns:
            分類結果のリスト
        """
        results = []

        # バッチサイズ
        batch_size = 32

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_tensors = []

            # バッチの前処理
            for frame in batch_frames:
                try:
                    if isinstance(frame, str):
                        image = cv2.imread(frame)
                        if image is None:
                            batch_tensors.append(None)
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image = frame
                        if len(image.shape) == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        elif image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                        elif image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    self.logger.error(f"前処理エラー: {e}")
                    batch_tensors.append(None)

            # 有効なテンソルのみ処理
            valid_indices = [j for j, t in enumerate(batch_tensors) if t is not None]
            if not valid_indices:
                results.extend([(False, 0.0)] * len(batch_frames))
                continue

            # バッチ推論
            valid_tensors = torch.stack([batch_tensors[j] for j in valid_indices])
            valid_tensors = valid_tensors.to(self.device)

            with torch.no_grad():
                outputs = self.model(valid_tensors)
                probabilities = F.softmax(outputs, dim=1)

            # 結果を整理
            batch_results = [(False, 0.0)] * len(batch_frames)
            for idx, valid_idx in enumerate(valid_indices):
                game_scene_prob = probabilities[idx, 1].item()
                is_game_scene = game_scene_prob >= self.confidence_threshold
                batch_results[valid_idx] = (is_game_scene, game_scene_prob)

            results.extend(batch_results)

        return results

    def extract_features(self, frame: np.ndarray | str) -> np.ndarray | None:
        """
        フレームから特徴量を抽出（中間層の出力）

        Args:
            frame: 画像データまたは画像ファイルパス

        Returns:
            特徴量ベクトル
        """
        try:
            # 画像読み込みと前処理
            if isinstance(frame, str):
                image = cv2.imread(frame)
                if image is None:
                    return None
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = frame
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 特徴量抽出（最終層の前の層の出力を取得）
            features = []

            def hook(module, input, output):
                features.append(output.detach())

            # フックを登録
            handle = self.backbone.features.register_forward_hook(hook)

            with torch.no_grad():
                _ = self.model(input_tensor)

            handle.remove()

            if features:
                # Global Average Pooling
                feature_vector = F.adaptive_avg_pool2d(features[0], 1)
                feature_vector = feature_vector.squeeze().cpu().numpy()
                return feature_vector
            else:
                return None

        except Exception as e:
            self.logger.error(f"特徴量抽出エラー: {e}")
            return None
