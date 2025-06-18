"""
統合データ拡張モジュール

すべての拡張機能を統合し、簡単に使用できるインターフェースを提供
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .advanced_augmentor import AdvancedAugmentor
from .color_augmentor import RedDoraAugmentor


class UnifiedAugmentor:
    """すべての拡張機能を統合したクラス"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: 設定辞書
                - augmentation_factor: 拡張倍率（デフォルト: 20）
                - enable_red_dora: 赤ドラ拡張を有効化（デフォルト: True）
                - output_format: 出力形式（'yolo', 'coco', 'custom'）
        """
        self.config = config or {}
        self.augmentation_factor = self.config.get("augmentation_factor", 20)
        self.enable_red_dora = self.config.get("enable_red_dora", True)
        self.output_format = self.config.get("output_format", "yolo")

        # 拡張器の初期化
        self.advanced = AdvancedAugmentor(augmentation_factor=self.augmentation_factor)
        self.red_dora = RedDoraAugmentor() if self.enable_red_dora else None

        # 統計情報
        self.stats = {
            "total_images_processed": 0,
            "total_augmented_generated": 0,
            "red_dora_generated": 0,
            "processing_time": 0,
        }

    def augment_dataset(
        self, input_path: str, output_path: str, train_val_split: float = 0.8
    ) -> dict[str, Any]:
        """
        データセット全体の拡張を実行

        Args:
            input_path: 入力データセットのパス
            output_path: 出力先のパス
            train_val_split: 訓練/検証データの分割比率

        Returns:
            処理結果の統計情報
        """
        start_time = datetime.now()

        # パスの準備
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 訓練/検証ディレクトリの作成
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        # データの読み込み
        print("データセットを読み込み中...")
        dataset = self._load_dataset(input_dir)

        # クラスごとに処理
        all_samples = []
        class_stats = {}

        for class_name, samples in dataset.items():
            print(f"\nクラス '{class_name}' を処理中 ({len(samples)} サンプル)...")

            augmented_samples = []

            # 赤ドラの特別処理
            if self.enable_red_dora and class_name in ["5m", "5p", "5s"]:
                augmented_samples.extend(self._process_potential_red_dora(samples, class_name))
            else:
                # 通常の拡張処理
                for sample in samples:
                    augmented = self._augment_sample(sample)
                    augmented_samples.extend(augmented)

            # クラス統計を記録
            class_stats[class_name] = {
                "original": len(samples),
                "augmented": len(augmented_samples),
            }

            # 全サンプルに追加
            all_samples.extend(augmented_samples)

        # 訓練/検証に分割
        print("\n訓練/検証データに分割中...")
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_val_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        # データを保存
        print("\nデータを保存中...")
        self._save_samples(train_samples, train_dir, "train")
        self._save_samples(val_samples, val_dir, "val")

        # データセット設定ファイルを作成
        if self.output_format == "yolo":
            self._create_yolo_config(output_dir, class_stats.keys())

        # 統計情報を更新
        end_time = datetime.now()
        self.stats["processing_time"] = (end_time - start_time).total_seconds()
        self.stats["total_images_processed"] = sum(len(samples) for samples in dataset.values())
        self.stats["total_augmented_generated"] = len(all_samples)

        # レポートを生成
        report = self._generate_report(class_stats, train_samples, val_samples, output_dir)

        print("\n処理完了！")
        print(f"処理時間: {self.stats['processing_time']:.1f}秒")
        print(f"生成されたサンプル数: {len(all_samples)}")

        return report

    def _load_dataset(self, input_dir: Path) -> dict[str, list[dict]]:
        """データセットを読み込む"""
        dataset = {}

        # アノテーションファイルがある場合
        annotations_file = input_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file) as f:
                annotations = json.load(f)

            for ann in annotations:
                class_name = ann["class"]
                if class_name not in dataset:
                    dataset[class_name] = []

                # 画像を読み込み
                image_path = input_dir / ann["image_path"]
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    dataset[class_name].append(
                        {
                            "image": image,
                            "bboxes": ann.get("bboxes", [[0, 0, image.shape[1], image.shape[0]]]),
                            "class_ids": ann.get("class_ids", [self._get_class_id(class_name)]),
                            "original_path": str(image_path),
                        }
                    )
        else:
            # ディレクトリ構造から推測
            for class_dir in input_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    dataset[class_name] = []

                    for image_path in class_dir.glob("*.jpg"):
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            dataset[class_name].append(
                                {
                                    "image": image,
                                    "bboxes": [[0, 0, image.shape[1], image.shape[0]]],
                                    "class_ids": [self._get_class_id(class_name)],
                                    "original_path": str(image_path),
                                }
                            )

        return dataset

    def _augment_sample(self, sample: dict) -> list[dict]:
        """単一サンプルを拡張"""
        augmented = self.advanced.augment_single_image(
            sample["image"], sample["bboxes"], sample["class_ids"]
        )

        # 元のサンプル情報を追加
        for aug in augmented:
            aug["original_path"] = sample.get("original_path", "unknown")

        return augmented

    def _process_potential_red_dora(self, samples: list[dict], class_name: str) -> list[dict]:
        """5の牌を通常版と赤ドラ版の両方に拡張"""
        all_augmented = []

        for sample in samples:
            # 通常の5として拡張（半分の量）
            normal_factor = self.augmentation_factor // 2
            self.advanced.augmentation_factor = normal_factor
            normal_augmented = self._augment_sample(sample)

            # クラス名を維持
            for aug in normal_augmented:
                aug["class_name"] = class_name
                aug["is_red_dora"] = False

            all_augmented.extend(normal_augmented)

            # 赤ドラとして拡張（半分の量）
            red_dora_class = class_name.replace("5", "0")
            red_variations = self.red_dora.create_red_dora_variations(
                sample["image"], n_variations=normal_factor
            )

            # 赤ドラのバウンディングボックス情報を追加
            for var in red_variations:
                all_augmented.append(
                    {
                        "image": var["image"],
                        "bboxes": sample["bboxes"],
                        "class_ids": [self._get_class_id(red_dora_class)],
                        "class_name": red_dora_class,
                        "is_red_dora": True,
                        "original_path": sample.get("original_path", "unknown"),
                        "augmentation_info": var,
                    }
                )

            self.stats["red_dora_generated"] += len(red_variations)

        # 拡張係数を元に戻す
        self.advanced.augmentation_factor = self.augmentation_factor

        return all_augmented

    def _save_samples(self, samples: list[dict], output_dir: Path, split: str):
        """サンプルを保存"""
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        for idx, sample in enumerate(samples):
            # ファイル名の生成
            class_name = sample.get("class_name", "unknown")
            filename = f"{split}_{class_name}_{idx:06d}"

            # 画像を保存
            image_path = images_dir / f"{filename}.jpg"
            cv2.imwrite(str(image_path), sample["image"])

            # ラベルを保存（YOLO形式）
            if self.output_format == "yolo":
                label_path = labels_dir / f"{filename}.txt"
                self._save_yolo_annotation(
                    label_path, sample["bboxes"], sample["class_ids"], sample["image"].shape
                )

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

                # YOLO形式に変換
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                )

    def _create_yolo_config(self, output_dir: Path, class_names):
        """YOLO用の設定ファイルを作成"""
        # すべてのクラス名を取得
        all_classes = []
        for suit in ["m", "p", "s"]:
            for num in range(1, 10):
                all_classes.append(f"{num}{suit}")
        all_classes.extend(["1z", "2z", "3z", "4z", "5z", "6z", "7z"])
        all_classes.extend(["0m", "0p", "0s"])
        all_classes.append("back")

        config = {
            "path": str(output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "names": dict(enumerate(all_classes)),
            "nc": len(all_classes),
        }

        with open(output_dir / "dataset.yaml", "w") as f:
            import yaml

            yaml.dump(config, f, default_flow_style=False)

    def _get_class_id(self, class_name: str) -> int:
        """クラス名からクラスIDを取得"""
        all_classes = []
        for suit in ["m", "p", "s"]:
            for num in range(1, 10):
                all_classes.append(f"{num}{suit}")
        all_classes.extend(["1z", "2z", "3z", "4z", "5z", "6z", "7z"])
        all_classes.extend(["0m", "0p", "0s"])
        all_classes.append("back")

        try:
            return all_classes.index(class_name)
        except ValueError:
            return 0

    def _generate_report(
        self, class_stats: dict, train_samples: list, val_samples: list, output_dir: Path
    ) -> dict[str, Any]:
        """処理レポートを生成"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "augmentation_factor": self.augmentation_factor,
                "enable_red_dora": self.enable_red_dora,
                "output_format": self.output_format,
            },
            "statistics": {
                "total_original_images": self.stats["total_images_processed"],
                "total_augmented_images": self.stats["total_augmented_generated"],
                "red_dora_generated": self.stats["red_dora_generated"],
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "processing_time_seconds": self.stats["processing_time"],
            },
            "class_distribution": class_stats,
            "output_directory": str(output_dir.absolute()),
        }

        # レポートを保存
        report_path = output_dir / "augmentation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Markdownレポートも生成
        self._generate_markdown_report(report, output_dir)

        return report

    def _generate_markdown_report(self, report: dict, output_dir: Path):
        """Markdownフォーマットのレポートを生成"""
        md_content = f"""# データ拡張レポート

生成日時: {report["timestamp"]}

## 設定

- 拡張倍率: {report["configuration"]["augmentation_factor"]}倍
- 赤ドラ拡張: {"有効" if report["configuration"]["enable_red_dora"] else "無効"}
- 出力形式: {report["configuration"]["output_format"]}

## 統計情報

| 項目 | 値 |
|------|-----|
| 元画像数 | {report["statistics"]["total_original_images"]} |
| 生成画像数 | {report["statistics"]["total_augmented_images"]} |
| 赤ドラ生成数 | {report["statistics"]["red_dora_generated"]} |
| 訓練サンプル数 | {report["statistics"]["train_samples"]} |
| 検証サンプル数 | {report["statistics"]["val_samples"]} |
| 処理時間 | {report["statistics"]["processing_time_seconds"]:.1f}秒 |

## クラス分布

| クラス | 元画像数 | 拡張後 |
|--------|----------|--------|
"""

        for class_name, stats in report["class_distribution"].items():
            md_content += f"| {class_name} | {stats['original']} | {stats['augmented']} |\n"

        md_content += f"\n## 出力先\n\n`{report['output_directory']}`\n"

        md_path = output_dir / "augmentation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
