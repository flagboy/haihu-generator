"""
データローダーファクトリ

データローダーの作成責務を分離し、異なるタイプのデータローダーを
一貫した方法で作成する
"""

from typing import Any

try:
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None
    transforms = None

from ....utils.logger import LoggerMixin
from ...annotation_data import AnnotationData


class DataLoaderFactory(LoggerMixin):
    """データローダーファクトリクラス"""

    def __init__(self, device: Any = None):
        """
        初期化

        Args:
            device: 使用するデバイス
        """
        self.device = device or (torch.device("cpu") if TORCH_AVAILABLE else None)

    def create_dataloader(
        self,
        annotation_data: AnnotationData,
        config: Any,
        is_training: bool,
        batch_size: int | None = None,
    ) -> DataLoader | None:
        """
        データローダーを作成

        Args:
            annotation_data: アノテーションデータ
            config: 設定オブジェクト
            is_training: 訓練用かどうか
            batch_size: バッチサイズ（オプション）

        Returns:
            データローダー
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorchが利用できません")
            return None

        # 画像変換を定義
        transform = self._create_transform(config, is_training)

        # データセットを作成（循環インポート回避のため動的インポート）
        from ..model_trainer import TileDataset

        dataset = TileDataset(
            annotation_data=annotation_data,
            model_type=config.model_type,
            transform=transform,
            augment=is_training and config.use_data_augmentation,
        )

        # バッチサイズの決定
        actual_batch_size = batch_size if batch_size is not None else config.batch_size

        # データローダーを作成
        return DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=is_training,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda" if self.device else False,
        )

    def _create_transform(self, config: Any, is_training: bool) -> Any:
        """
        画像変換を作成

        Args:
            config: 設定オブジェクト
            is_training: 訓練用かどうか

        Returns:
            変換オブジェクト
        """
        if not TORCH_AVAILABLE:
            return None

        transform_list = [transforms.ToTensor()]

        if (
            is_training
            and hasattr(config, "augmentation")
            and config.augmentation.get("enabled", False)
        ):
            # データ拡張を追加
            aug_config = config.augmentation

            if aug_config.get("random_horizontal_flip", False):
                transform_list.append(transforms.RandomHorizontalFlip())

            if aug_config.get("random_rotation", False):
                max_angle = aug_config.get("rotation_range", 15)
                transform_list.append(transforms.RandomRotation(max_angle))

            if aug_config.get("color_jitter", False):
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=aug_config.get("brightness_range", 0.2),
                        contrast=aug_config.get("contrast_range", 0.2),
                        saturation=aug_config.get("saturation_range", 0.2),
                        hue=aug_config.get("hue_range", 0.1),
                    )
                )

        # 正規化を追加
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        return transforms.Compose(transform_list)
