"""
高度なデータ拡張機能

Albumentationsを使用して、1枚の画像から20枚以上の
多様な拡張画像を生成する
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np


class AdvancedAugmentor:
    """20倍以上のデータ拡張を実現する高度な拡張クラス"""

    def __init__(self, augmentation_factor: int = 20):
        """
        Args:
            augmentation_factor: 1枚の画像から生成する画像数
        """
        self.augmentation_factor = augmentation_factor
        self.pipelines = self._create_augmentation_pipelines()

    def _create_augmentation_pipelines(self) -> list[A.Compose]:
        """多様な拡張パイプラインを作成"""

        # 基本的な幾何学的変換
        geometric_light = A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Transpose(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 中程度の幾何学的変換
        geometric_medium = A.Compose(
            [
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 透視変換（カメラアングルのシミュレーション）
        perspective = A.Compose(
            [
                A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.7),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-10, 10),
                    p=0.7,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 照明条件のシミュレーション
        lighting = A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.7
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 影のシミュレーション
        shadows = A.Compose(
            [
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.5),
                A.RandomToneCurve(scale=0.1, p=0.3),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # ノイズとブラー（実環境のシミュレーション）
        noise_blur = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussNoise(p=1),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=1),
                        A.MedianBlur(blur_limit=5, p=1),
                        A.GaussianBlur(blur_limit=(3, 7), p=1),
                        # DefocusBlurは一部のバージョンで利用できない可能性があるため除外
                    ],
                    p=0.5,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 画質劣化のシミュレーション
        quality_degradation = A.Compose(
            [
                A.Downscale(scale_range=(0.5, 0.9), p=0.3),
                A.ImageCompression(quality_range=(60, 95), p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # 天候条件のシミュレーション（オプション）
        weather = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomRain(
                            slant_range=(-10, 10),
                            drop_length=20,
                            drop_width=1,
                            drop_color=(200, 200, 200),
                            blur_value=3,
                            brightness_coefficient=0.7,
                            rain_type="default",
                            p=1,
                        ),
                        A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.08, p=1),
                    ],
                    p=0.1,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        )

        # パイプラインの組み合わせ
        return [
            # 軽度の変換（高頻度）
            A.Compose(
                [geometric_light, lighting, noise_blur],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            ),
            # 中程度の変換
            A.Compose(
                [geometric_medium, lighting, shadows],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            ),
            # 強度の変換
            A.Compose(
                [perspective, lighting, quality_degradation],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            ),
            # 特殊条件
            A.Compose(
                [geometric_light, weather, noise_blur],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
            ),
        ]

    def augment_single_image(
        self, image: np.ndarray, bboxes: list[list[float]], class_ids: list[int]
    ) -> list[dict[str, Any]]:
        """
        1枚の画像から複数の拡張画像を生成

        Args:
            image: 入力画像 (H, W, C)
            bboxes: バウンディングボックスのリスト [[x1, y1, x2, y2], ...]
            class_ids: 各バウンディングボックスのクラスID

        Returns:
            拡張された画像とアノテーションのリスト
        """
        augmented_samples = []

        # 元画像も含める
        augmented_samples.append(
            {
                "image": image.copy(),
                "bboxes": bboxes.copy(),
                "class_ids": class_ids.copy(),
                "augmentation_info": {"pipeline": "original", "iteration": 0},
            }
        )

        # 各パイプラインで複数回拡張
        remaining_samples = self.augmentation_factor - 1
        samples_per_pipeline = remaining_samples // len(self.pipelines)
        extra_samples = remaining_samples % len(self.pipelines)

        for pipeline_idx, pipeline in enumerate(self.pipelines):
            # このパイプラインで生成するサンプル数
            n_samples = samples_per_pipeline + (1 if pipeline_idx < extra_samples else 0)

            for i in range(n_samples):
                try:
                    # Albumentationsで変換
                    transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_ids)

                    # 有効なバウンディングボックスのみを保持
                    valid_bboxes = []
                    valid_class_ids = []

                    for bbox, class_id in zip(
                        transformed["bboxes"], transformed["class_labels"], strict=False
                    ):
                        # バウンディングボックスが画像内に収まっているか確認
                        x1, y1, x2, y2 = bbox
                        if (
                            0 <= x1 < transformed["image"].shape[1]
                            and 0 <= y1 < transformed["image"].shape[0]
                            and 0 < x2 <= transformed["image"].shape[1]
                            and 0 < y2 <= transformed["image"].shape[0]
                            and x2 > x1
                            and y2 > y1
                        ):
                            valid_bboxes.append(list(bbox))
                            valid_class_ids.append(class_id)

                    if valid_bboxes:  # 有効なバウンディングボックスがある場合のみ追加
                        augmented_samples.append(
                            {
                                "image": transformed["image"],
                                "bboxes": valid_bboxes,
                                "class_ids": valid_class_ids,
                                "augmentation_info": {
                                    "pipeline": f"pipeline_{pipeline_idx}",
                                    "iteration": i,
                                },
                            }
                        )

                except Exception as e:
                    print(f"警告: パイプライン {pipeline_idx} でエラーが発生しました: {e}")
                    continue

        # 目標数に達していない場合は、成功したサンプルを複製
        while len(augmented_samples) < self.augmentation_factor:
            source = augmented_samples[np.random.randint(1, len(augmented_samples))]
            augmented_samples.append(
                {
                    "image": source["image"].copy(),
                    "bboxes": source["bboxes"].copy(),
                    "class_ids": source["class_ids"].copy(),
                    "augmentation_info": {**source["augmentation_info"], "duplicated": True},
                }
            )

        return augmented_samples[: self.augmentation_factor]

    def create_balanced_dataset(
        self, original_data: dict[str, list[dict]], target_per_class: int = 1000
    ) -> dict[str, list[dict]]:
        """
        クラスバランスを考慮したデータセット作成

        Args:
            original_data: クラス名をキーとする元データ
            target_per_class: 各クラスの目標サンプル数

        Returns:
            バランスの取れたデータセット
        """
        balanced_data = {}

        for class_name, samples in original_data.items():
            if len(samples) >= target_per_class:
                # 十分なデータがある場合はランダムサンプリング
                indices = np.random.choice(len(samples), target_per_class, replace=False)
                balanced_data[class_name] = [samples[i] for i in indices]
            else:
                # データが不足している場合は拡張で補完
                augmented_samples = []

                # 各サンプルから必要な数だけ拡張
                augmentation_per_sample = (target_per_class // len(samples)) + 1

                for sample in samples:
                    # 画像とアノテーションを取得
                    image = sample["image"]
                    bboxes = sample.get("bboxes", [[0, 0, image.shape[1], image.shape[0]]])
                    class_ids = sample.get("class_ids", [self._get_class_id(class_name)])

                    # 一時的に拡張係数を設定
                    original_factor = self.augmentation_factor
                    self.augmentation_factor = augmentation_per_sample

                    # 拡張実行
                    augmented = self.augment_single_image(image, bboxes, class_ids)

                    # 拡張係数を元に戻す
                    self.augmentation_factor = original_factor

                    # サンプル情報を追加
                    for aug_sample in augmented:
                        augmented_samples.append(
                            {
                                **aug_sample,
                                "class_name": class_name,
                                "original_sample_id": sample.get("id", "unknown"),
                            }
                        )

                # 目標数までランダムに選択
                if len(augmented_samples) > target_per_class:
                    indices = np.random.choice(
                        len(augmented_samples), target_per_class, replace=False
                    )
                    balanced_data[class_name] = [augmented_samples[i] for i in indices]
                else:
                    balanced_data[class_name] = augmented_samples

        return balanced_data

    def _get_class_id(self, class_name: str) -> int:
        """クラス名からクラスIDを取得"""
        # 麻雀牌のクラスマッピング
        tile_classes = []

        # 数牌
        for suit in ["m", "p", "s"]:
            for num in range(1, 10):
                tile_classes.append(f"{num}{suit}")

        # 字牌
        tile_classes.extend(["1z", "2z", "3z", "4z", "5z", "6z", "7z"])

        # 赤ドラ
        tile_classes.extend(["0m", "0p", "0s"])

        # 裏面
        tile_classes.append("back")

        try:
            return tile_classes.index(class_name)
        except ValueError:
            return 0  # デフォルト値

    def save_augmented_dataset(
        self, augmented_data: dict[str, list[dict]], output_dir: str, format: str = "yolo"
    ):
        """
        拡張されたデータセットを保存

        Args:
            augmented_data: 拡張されたデータ
            output_dir: 出力ディレクトリ
            format: 出力形式 ('yolo', 'coco', 'custom')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 画像とアノテーションを保存
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # メタデータ
        metadata = {
            "created_at": datetime.now().isoformat(),
            "augmentation_factor": self.augmentation_factor,
            "total_samples": sum(len(samples) for samples in augmented_data.values()),
            "classes": list(augmented_data.keys()),
            "format": format,
        }

        sample_count = 0

        for class_name, samples in augmented_data.items():
            for sample in samples:
                # ファイル名の生成
                filename = f"{class_name}_{sample_count:06d}"

                # 画像を保存
                image_path = images_dir / f"{filename}.jpg"
                cv2.imwrite(str(image_path), sample["image"])

                # アノテーションを保存
                if format == "yolo":
                    self._save_yolo_annotation(
                        labels_dir / f"{filename}.txt",
                        sample["bboxes"],
                        sample["class_ids"],
                        sample["image"].shape,
                    )

                sample_count += 1

        # メタデータを保存
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"拡張データセットを保存しました: {output_path}")
        print(f"総サンプル数: {sample_count}")

    def _save_yolo_annotation(
        self,
        label_path: Path,
        bboxes: list[list[float]],
        class_ids: list[int],
        image_shape: tuple[int, int, int],
    ):
        """YOLO形式でアノテーションを保存"""
        height, width = image_shape[:2]

        with open(label_path, "w") as f:
            for bbox, class_id in zip(bboxes, class_ids, strict=False):
                x1, y1, x2, y2 = bbox

                # YOLO形式に変換 (中心座標と幅高さの正規化値)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                # クラスID x_center y_center width height
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                )
