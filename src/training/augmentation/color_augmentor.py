"""
赤ドラ検出のための色分析拡張モジュール

通常の5の牌から赤ドラのバリエーションを生成し、
色情報を活用した識別を可能にする
"""

from typing import Any

import albumentations as A
import cv2
import numpy as np


class RedDoraAugmentor:
    """赤ドラ検出のための特殊な色拡張"""

    def __init__(self):
        self.red_enhancement_pipeline = self._create_red_enhancement_pipeline()
        self.color_variations = self._create_color_variation_pipeline()

    def _create_red_enhancement_pipeline(self) -> A.Compose:
        """赤色を強調する拡張パイプライン"""
        return A.Compose(
            [
                # 赤色チャンネルの強調
                A.ChannelShuffle(p=0.3),
                A.RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=10, p=0.7),
                # 赤色の彩度を上げる
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=50, val_shift_limit=20, p=0.8
                ),
                # 赤色領域のコントラスト強調
                A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.5),
            ]
        )

    def _create_color_variation_pipeline(self) -> A.Compose:
        """様々な赤色のバリエーションを生成するパイプライン"""
        return A.Compose(
            [
                A.OneOf(
                    [
                        # 明るい赤
                        A.HueSaturationValue(
                            hue_shift_limit=(-5, 5),
                            sat_shift_limit=(20, 50),
                            val_shift_limit=(10, 30),
                            p=1,
                        ),
                        # 暗い赤
                        A.HueSaturationValue(
                            hue_shift_limit=(-5, 5),
                            sat_shift_limit=(10, 30),
                            val_shift_limit=(-30, -10),
                            p=1,
                        ),
                        # オレンジがかった赤
                        A.HueSaturationValue(
                            hue_shift_limit=(5, 15),
                            sat_shift_limit=(20, 40),
                            val_shift_limit=(-10, 10),
                            p=1,
                        ),
                        # 紫がかった赤
                        A.HueSaturationValue(
                            hue_shift_limit=(-15, -5),
                            sat_shift_limit=(20, 40),
                            val_shift_limit=(-10, 10),
                            p=1,
                        ),
                    ],
                    p=1,
                ),
                # 照明条件の変化
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # ノイズの追加
                A.GaussNoise(p=0.3),
            ]
        )

    def create_red_dora_variations(
        self, base_five_tile: np.ndarray, n_variations: int = 50
    ) -> list[dict[str, Any]]:
        """
        通常の5の牌から赤ドラのバリエーションを生成

        Args:
            base_five_tile: 元となる5の牌の画像
            n_variations: 生成するバリエーション数

        Returns:
            赤ドラのバリエーション画像リスト
        """
        variations = []

        for i in range(n_variations):
            # 赤色オーバーレイの作成方法を選択
            method = np.random.choice(["overlay", "color_replace", "gradient", "mixed"])

            if method == "overlay":
                red_tile = self._apply_red_overlay(base_five_tile)
            elif method == "color_replace":
                red_tile = self._replace_with_red(base_five_tile)
            elif method == "gradient":
                red_tile = self._apply_red_gradient(base_five_tile)
            else:  # mixed
                red_tile = self._apply_mixed_red_effect(base_five_tile)

            # 追加の色拡張
            if i % 2 == 0:
                augmented = self.red_enhancement_pipeline(image=red_tile)["image"]
            else:
                augmented = self.color_variations(image=red_tile)["image"]

            variations.append(
                {
                    "image": augmented,
                    "method": method,
                    "variation_id": i,
                    "color_stats": self._calculate_color_statistics(augmented),
                }
            )

        return variations

    def _apply_red_overlay(self, image: np.ndarray) -> np.ndarray:
        """赤色オーバーレイを適用"""
        red_overlay = np.zeros_like(image)

        # 赤チャンネルを強調
        red_intensity = np.random.randint(100, 200)
        red_overlay[:, :, 2] = red_intensity  # BGR形式でRチャンネル

        # アルファブレンディング
        alpha = np.random.uniform(0.3, 0.7)
        result = cv2.addWeighted(image, 1 - alpha, red_overlay, alpha, 0)

        return result

    def _replace_with_red(self, image: np.ndarray) -> np.ndarray:
        """牌の主要色を赤に置換"""
        # HSV変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 主要色（通常は白や緑）のマスクを作成
        # 明度が高い部分を検出
        _, mask = cv2.threshold(hsv[:, :, 2], 180, 255, cv2.THRESH_BINARY)

        # マスク部分を赤色に置換
        result = image.copy()
        red_color = np.array([40, 40, 200], dtype=np.uint8)  # BGR形式で赤

        # マスクをぼかして自然な境界を作成
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        result = (result * (1 - mask_3ch) + red_color * mask_3ch).astype(np.uint8)

        return result

    def _apply_red_gradient(self, image: np.ndarray) -> np.ndarray:
        """赤色グラデーションを適用"""
        height, width = image.shape[:2]

        # グラデーションマスクの作成
        gradient = np.linspace(0, 1, width).reshape(1, -1)
        gradient = np.repeat(gradient, height, axis=0)
        gradient = np.stack([gradient] * 3, axis=-1)

        # ランダムな方向
        if np.random.random() > 0.5:
            gradient = np.fliplr(gradient)
        if np.random.random() > 0.5:
            gradient = np.flipud(gradient)

        # 赤色グラデーション
        red_gradient = np.zeros_like(image, dtype=np.float32)
        red_gradient[:, :, 2] = 200  # 赤チャンネル

        # 適用
        result = image.astype(np.float32) * (1 - gradient * 0.5) + red_gradient * gradient * 0.5

        return result.astype(np.uint8)

    def _apply_mixed_red_effect(self, image: np.ndarray) -> np.ndarray:
        """複数の赤色効果を組み合わせて適用"""
        # ランダムに2つの効果を選択して組み合わせる
        methods = [self._apply_red_overlay, self._replace_with_red, self._apply_red_gradient]

        # 最初の効果
        method1 = np.random.choice(methods)
        result = method1(image)

        # 2つ目の効果を弱く適用
        method2 = np.random.choice([m for m in methods if m != method1])
        effect2 = method2(image)

        # ブレンド
        alpha = np.random.uniform(0.2, 0.4)
        result = cv2.addWeighted(result, 1 - alpha, effect2, alpha, 0)

        return result

    def _calculate_color_statistics(self, image: np.ndarray) -> dict[str, float]:
        """画像の色統計を計算"""
        # BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 各チャンネルの平均と標準偏差
        stats = {
            "r_mean": np.mean(rgb[:, :, 0]),
            "g_mean": np.mean(rgb[:, :, 1]),
            "b_mean": np.mean(rgb[:, :, 2]),
            "r_std": np.std(rgb[:, :, 0]),
            "g_std": np.std(rgb[:, :, 1]),
            "b_std": np.std(rgb[:, :, 2]),
        }

        # 赤色の割合
        red_ratio = stats["r_mean"] / (stats["r_mean"] + stats["g_mean"] + stats["b_mean"] + 1e-6)
        stats["red_ratio"] = red_ratio

        # HSV統計
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        stats["h_mean"] = np.mean(hsv[:, :, 0])
        stats["s_mean"] = np.mean(hsv[:, :, 1])
        stats["v_mean"] = np.mean(hsv[:, :, 2])

        return stats

    def detect_red_dora(self, tile_image: np.ndarray, tile_class: str) -> tuple[bool, float]:
        """
        牌画像が赤ドラかどうかを判定

        Args:
            tile_image: 牌の画像
            tile_class: 牌の種類（例: "5m", "5p", "5s"）

        Returns:
            (is_red_dora, confidence): 赤ドラかどうかと信頼度
        """
        # 5の牌でない場合は赤ドラではない
        if tile_class not in ["5m", "5p", "5s"]:
            return False, 1.0

        # 色統計を計算
        stats = self._calculate_color_statistics(tile_image)

        # 赤色判定の条件
        conditions = [
            stats["red_ratio"] > 0.4,  # 赤色が支配的
            stats["r_mean"] > stats["g_mean"] * 1.5,  # 赤が緑より明らかに強い
            stats["r_mean"] > stats["b_mean"] * 1.5,  # 赤が青より明らかに強い
            stats["s_mean"] > 50,  # 彩度が高い
        ]

        # 条件を満たす数に基づいて判定
        satisfied_conditions = sum(conditions)
        is_red_dora = satisfied_conditions >= 3

        # 信頼度の計算
        confidence = satisfied_conditions / len(conditions)

        # HSV色空間での詳細チェック
        if is_red_dora:
            # 赤色の色相範囲（0-10または170-180）
            h_mean = stats["h_mean"]
            if (0 <= h_mean <= 10) or (170 <= h_mean <= 180):
                confidence = min(confidence * 1.2, 1.0)
            else:
                confidence *= 0.8

        return is_red_dora, confidence

    def create_training_pairs(
        self, five_tiles: list[np.ndarray], n_variations: int = 30
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        通常の5と赤5の訓練ペアを作成

        Args:
            five_tiles: 5の牌の画像リスト
            n_variations: 各画像から生成するバリエーション数

        Returns:
            (images, labels): 画像とラベル（0: 通常の5, 1: 赤5）
        """
        images = []
        labels = []

        for tile in five_tiles:
            # 通常の5（オリジナル + 少し拡張）
            images.append(tile)
            labels.append(0)

            # 通常の5の色バリエーション（赤くない）
            normal_augmented = self._create_normal_variations(tile, n_variations // 2)
            images.extend([v["image"] for v in normal_augmented])
            labels.extend([0] * len(normal_augmented))

            # 赤5のバリエーション
            red_variations = self.create_red_dora_variations(tile, n_variations // 2)
            images.extend([v["image"] for v in red_variations])
            labels.extend([1] * len(red_variations))

        return images, labels

    def _create_normal_variations(
        self, tile: np.ndarray, n_variations: int
    ) -> list[dict[str, Any]]:
        """通常の牌の色バリエーションを作成（赤くない）"""
        pipeline = A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=(-20, 20),
                    sat_shift_limit=(-30, 30),
                    val_shift_limit=(-20, 20),
                    p=0.8,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.OneOf(
                    [
                        A.ChannelShuffle(p=1),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=20, b_shift_limit=20, p=1),
                    ],
                    p=0.5,
                ),
            ]
        )

        variations = []
        for i in range(n_variations):
            augmented = pipeline(image=tile)["image"]

            # 赤くなっていないことを確認
            stats = self._calculate_color_statistics(augmented)
            if stats["red_ratio"] < 0.35:  # 赤が強すぎない
                variations.append({"image": augmented, "variation_id": i, "color_stats": stats})

        return variations
