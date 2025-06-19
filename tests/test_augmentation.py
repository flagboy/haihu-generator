"""
データ拡張機能のテストコード
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.training.augmentation import AdvancedAugmentor, RedDoraAugmentor
from src.training.augmentation.unified_augmentor import UnifiedAugmentor


class TestAdvancedAugmentor:
    """AdvancedAugmentorのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        augmentor = AdvancedAugmentor(augmentation_factor=20)
        assert augmentor.augmentation_factor == 20
        assert len(augmentor.pipelines) > 0

    def test_augment_single_image(self):
        """単一画像の拡張テスト"""
        # テスト画像の作成
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_bbox = [[10, 10, 90, 90]]
        test_class_id = [0]

        augmentor = AdvancedAugmentor(augmentation_factor=10)
        augmented = augmentor.augment_single_image(test_image, test_bbox, test_class_id)

        # 指定した数の画像が生成されることを確認
        assert len(augmented) == 10

        # 各拡張画像が異なることを確認
        images = [sample["image"] for sample in augmented]

        # 少なくとも半分は異なる画像であることを確認
        unique_count = 0
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if not np.array_equal(images[i], images[j]):
                    unique_count += 1

        assert unique_count > len(images) * 0.3  # 30%以上は異なる

    def test_bbox_transformation(self):
        """バウンディングボックスの変換テスト"""
        # より大きなテスト画像
        test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        test_bbox = [[50, 50, 150, 150]]
        test_class_id = [5]

        augmentor = AdvancedAugmentor(augmentation_factor=5)
        augmented = augmentor.augment_single_image(test_image, test_bbox, test_class_id)

        # すべての拡張画像でバウンディングボックスが有効であることを確認
        for sample in augmented:
            assert len(sample["bboxes"]) > 0
            bbox = sample["bboxes"][0]

            # バウンディングボックスが画像内に収まっていることを確認
            assert 0 <= bbox[0] < sample["image"].shape[1]
            assert 0 <= bbox[1] < sample["image"].shape[0]
            assert 0 < bbox[2] <= sample["image"].shape[1]
            assert 0 < bbox[3] <= sample["image"].shape[0]
            assert bbox[2] > bbox[0]
            assert bbox[3] > bbox[1]

    def test_create_balanced_dataset(self):
        """バランスの取れたデータセット作成のテスト"""
        # テストデータの作成
        test_data = {
            "1m": [
                {
                    "image": np.ones((50, 50, 3), dtype=np.uint8) * i,
                    "bboxes": [[5, 5, 45, 45]],
                    "class_ids": [0],
                }
                for i in range(10)
            ],
            "2m": [
                {
                    "image": np.ones((50, 50, 3), dtype=np.uint8) * i,
                    "bboxes": [[5, 5, 45, 45]],
                    "class_ids": [1],
                }
                for i in range(5)
            ],
        }

        augmentor = AdvancedAugmentor(augmentation_factor=10)
        balanced = augmentor.create_balanced_dataset(test_data, target_per_class=20)

        # 各クラスが目標数に達していることを確認
        assert len(balanced["1m"]) == 20
        assert len(balanced["2m"]) == 20


class TestRedDoraAugmentor:
    """RedDoraAugmentorのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        augmentor = RedDoraAugmentor()
        assert augmentor.red_enhancement_pipeline is not None
        assert augmentor.color_variations is not None

    def test_create_red_dora_variations(self):
        """赤ドラバリエーション生成のテスト"""
        # テスト用の5の牌画像（白っぽい画像）
        test_tile = np.ones((50, 50, 3), dtype=np.uint8) * 200

        augmentor = RedDoraAugmentor()
        variations = augmentor.create_red_dora_variations(test_tile, n_variations=10)

        # 指定した数のバリエーションが生成されることを確認
        assert len(variations) == 10

        # 各バリエーションが赤みを帯びていることを確認
        for var in variations:
            stats = var["color_stats"]
            # 赤色の割合が増加していることを確認
            assert stats["red_ratio"] > 0.1

    def test_detect_red_dora(self):
        """赤ドラ検出のテスト"""
        augmentor = RedDoraAugmentor()

        # 赤い画像のテスト
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :, 2] = 200  # BGR形式で赤

        is_red, confidence = augmentor.detect_red_dora(red_image, "5m")
        assert is_red
        assert confidence > 0.7

        # 白い画像のテスト
        white_image = np.ones((50, 50, 3), dtype=np.uint8) * 200

        is_red, confidence = augmentor.detect_red_dora(white_image, "5m")
        assert not is_red

        # 5以外の牌のテスト
        is_red, confidence = augmentor.detect_red_dora(red_image, "3m")
        assert not is_red
        assert confidence == 1.0

    def test_create_training_pairs(self):
        """訓練ペア作成のテスト"""
        # テスト用の5の牌画像
        test_tiles = [
            np.ones((50, 50, 3), dtype=np.uint8) * 200,
            np.ones((50, 50, 3), dtype=np.uint8) * 180,
        ]

        augmentor = RedDoraAugmentor()
        images, labels = augmentor.create_training_pairs(test_tiles, n_variations=10)

        # 生成された画像とラベルの数が一致することを確認
        assert len(images) == len(labels)

        # 通常の5（ラベル0）と赤5（ラベル1）の両方が含まれることを確認
        assert 0 in labels
        assert 1 in labels

        # ラベル1の画像が赤みを帯びていることを確認
        red_indices = [i for i, label in enumerate(labels) if label == 1]
        for idx in red_indices[:5]:  # 最初の5つをチェック
            image = images[idx]
            # HSV色空間で赤色の特徴を確認（赤色は色相0付近または180付近）
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])

            # 赤色の特徴：低い色相値（0-20または160-180）かつ高い彩度
            is_red_hue = (h_mean < 20 or h_mean > 160) and s_mean > 50

            # または、RGB色空間で赤色成分が強い
            b_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            r_mean = np.mean(image[:, :, 2])
            is_red_dominant = r_mean > max(b_mean, g_mean) * 1.1

            # いずれかの条件を満たせばOK
            assert is_red_hue or is_red_dominant


class TestUnifiedAugmentor:
    """UnifiedAugmentorのテスト"""

    def test_initialization(self):
        """初期化のテスト"""
        config = {"augmentation_factor": 15, "enable_red_dora": True, "output_format": "yolo"}
        augmentor = UnifiedAugmentor(config)

        assert augmentor.augmentation_factor == 15
        assert augmentor.enable_red_dora
        assert augmentor.output_format == "yolo"
        assert augmentor.red_dora is not None

    def test_augment_dataset(self):
        """データセット全体の拡張テスト"""
        # 一時ディレクトリの作成
        with tempfile.TemporaryDirectory() as temp_dir:
            # テスト用のデータセットを作成
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"

            # クラスごとのディレクトリを作成
            for class_name in ["1m", "5m", "1z"]:
                class_dir = input_dir / class_name
                class_dir.mkdir(parents=True)

                # 各クラスに2枚の画像を作成
                for i in range(2):
                    image = np.ones((100, 100, 3), dtype=np.uint8) * (100 + i * 50)
                    image_path = class_dir / f"image_{i}.jpg"
                    cv2.imwrite(str(image_path), image)

            # 拡張を実行
            config = {"augmentation_factor": 5, "enable_red_dora": True, "output_format": "yolo"}
            augmentor = UnifiedAugmentor(config)
            report = augmentor.augment_dataset(str(input_dir), str(output_dir))

            # 出力ディレクトリの確認
            assert output_dir.exists()
            assert (output_dir / "train").exists()
            assert (output_dir / "val").exists()
            assert (output_dir / "dataset.yaml").exists()
            assert (output_dir / "augmentation_report.json").exists()
            assert (output_dir / "augmentation_report.md").exists()

            # レポートの確認
            assert report["statistics"]["total_original_images"] == 6  # 3クラス × 2枚
            assert report["statistics"]["total_augmented_images"] > 6

            # 5mクラスで赤ドラが生成されていることを確認
            if "5m" in report["class_distribution"]:
                assert report["statistics"]["red_dora_generated"] > 0

    def test_yolo_format_output(self):
        """YOLO形式の出力テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 簡単なテストデータ
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"

            # アノテーション付きのテストデータを作成
            input_dir.mkdir()
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(input_dir / "test.jpg"), test_image)

            # アノテーションファイル
            annotations = [
                {
                    "class": "1m",
                    "image_path": "test.jpg",
                    "bboxes": [[50, 50, 150, 150]],
                    "class_ids": [0],
                }
            ]

            with open(input_dir / "annotations.json", "w") as f:
                json.dump(annotations, f)

            # 拡張を実行
            config = {"augmentation_factor": 2, "output_format": "yolo"}
            augmentor = UnifiedAugmentor(config)
            augmentor.augment_dataset(str(input_dir), str(output_dir))

            # YOLO形式のラベルファイルを確認
            label_files = list((output_dir / "train" / "labels").glob("*.txt"))
            assert len(label_files) > 0

            # ラベルファイルの内容を確認
            with open(label_files[0]) as f:
                lines = f.readlines()
                assert len(lines) > 0

                # YOLO形式の確認（class_id x_center y_center width height）
                parts = lines[0].strip().split()
                assert len(parts) == 5
                assert all(0 <= float(parts[i]) <= 1 for i in range(1, 5))


# 実行用のヘルパー関数
def run_simple_augmentation_test():
    """簡単な拡張テストを実行"""
    print("データ拡張機能の簡単なテストを実行中...")

    # テスト画像の作成
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    test_image[40:60, 40:60] = [100, 150, 200]  # 中央に色付き領域

    # 拡張実行
    augmentor = AdvancedAugmentor(augmentation_factor=5)
    augmented = augmentor.augment_single_image(test_image, [[30, 30, 70, 70]], [0])

    print(f"✓ {len(augmented)}枚の拡張画像を生成しました")

    # 赤ドラテスト
    red_augmentor = RedDoraAugmentor()
    red_variations = red_augmentor.create_red_dora_variations(test_image, n_variations=3)

    print(f"✓ {len(red_variations)}枚の赤ドラバリエーションを生成しました")

    print("\nテスト完了！")


if __name__ == "__main__":
    run_simple_augmentation_test()
