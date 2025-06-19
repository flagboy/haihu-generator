"""
YOLOv8を使用した麻雀牌検出器

ultralyticsライブラリを使用して、高精度な牌検出を実現
"""

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


class YOLOv8TileDetector:
    """YOLOv8を使用した麻雀牌検出器"""

    def __init__(self, model_path: str | None = None, device: str = "auto"):
        """
        Args:
            model_path: 学習済みモデルのパス（Noneの場合は新規作成）
            device: 使用デバイス（'auto', 'cpu', 'cuda', 'mps'）
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.class_names = self._setup_class_names()
        self.model_path = model_path

    def _setup_device(self, device: str) -> str:
        """使用デバイスの設定"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self, model_path: str | None) -> YOLO:
        """モデルの読み込みまたは作成"""
        if model_path and Path(model_path).exists():
            # 既存のモデルを読み込み
            print(f"既存のモデルを読み込み: {model_path}")
            return YOLO(model_path)
        else:
            # 新規モデルの作成（nano版から開始）
            print("新規YOLOv8nモデルを作成")
            return YOLO("yolov8n.yaml")

    def _setup_class_names(self) -> list[str]:
        """麻雀牌のクラス名を設定"""
        tiles = []

        # 数牌
        for suit in ["m", "p", "s"]:
            for num in range(1, 10):
                tiles.append(f"{num}{suit}")

        # 字牌
        tiles.extend(["1z", "2z", "3z", "4z", "5z", "6z", "7z"])

        # 赤ドラ
        tiles.extend(["0m", "0p", "0s"])

        # 裏面
        tiles.append("back")

        return tiles

    def prepare_training_data(
        self, dataset_path: str, output_path: str, train_val_split: float = 0.8
    ) -> str:
        """
        YOLOv8形式のデータセット準備

        Args:
            dataset_path: 入力データセットのパス
            output_path: 出力先のパス
            train_val_split: 訓練/検証の分割比率

        Returns:
            データセット設定ファイルのパス
        """
        # ディレクトリ構造の作成
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # YOLOv8のディレクトリ構造
        for split in ["train", "val"]:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # データセットの読み込みと変換
        self._convert_dataset(dataset_path, output_dir, train_val_split)

        # データセット設定ファイルの作成
        yaml_path = output_dir / "dataset.yaml"
        self._create_dataset_yaml(output_dir, yaml_path)

        print(f"YOLOv8データセットを作成しました: {output_dir}")
        return str(yaml_path)

    def _convert_dataset(self, input_path: str, output_dir: Path, train_val_split: float):
        """データセットをYOLO形式に変換"""
        input_dir = Path(input_path)

        # アノテーションファイルがある場合
        annotations_file = input_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file) as f:
                annotations = json.load(f)

            # ランダムに分割
            np.random.shuffle(annotations)
            split_idx = int(len(annotations) * train_val_split)
            train_anns = annotations[:split_idx]
            val_anns = annotations[split_idx:]

            # 各分割を処理
            self._process_annotations(train_anns, input_dir, output_dir, "train")
            self._process_annotations(val_anns, input_dir, output_dir, "val")
        else:
            # ディレクトリベースの処理
            self._process_directory_structure(input_dir, output_dir, train_val_split)

    def _process_annotations(
        self, annotations: list[dict], input_dir: Path, output_dir: Path, split: str
    ):
        """アノテーションの処理"""
        for idx, ann in enumerate(annotations):
            # 画像のコピー
            src_image = input_dir / ann.get("image_path", "")
            if not src_image.exists():
                continue

            # 新しいファイル名
            new_name = f"{split}_{idx:06d}.jpg"
            dst_image = output_dir / "images" / split / new_name

            # 画像をコピー
            shutil.copy(src_image, dst_image)

            # ラベルファイルの作成
            label_path = output_dir / "labels" / split / f"{split}_{idx:06d}.txt"
            self._create_yolo_label(ann, label_path, src_image)

    def _process_directory_structure(
        self, input_dir: Path, output_dir: Path, train_val_split: float
    ):
        """ディレクトリ構造からの処理"""
        all_images = []

        # 画像を収集
        for image_path in input_dir.rglob("*.jpg"):
            all_images.append(image_path)

        # ランダムに分割
        np.random.shuffle(all_images)
        split_idx = int(len(all_images) * train_val_split)

        # 訓練データ
        for idx, image_path in enumerate(all_images[:split_idx]):
            new_name = f"train_{idx:06d}.jpg"
            dst_image = output_dir / "images" / "train" / new_name
            shutil.copy(image_path, dst_image)

            # 対応するラベルファイルがあれば処理
            label_src = image_path.with_suffix(".txt")
            if label_src.exists():
                label_dst = output_dir / "labels" / "train" / f"train_{idx:06d}.txt"
                shutil.copy(label_src, label_dst)

        # 検証データ
        for idx, image_path in enumerate(all_images[split_idx:]):
            new_name = f"val_{idx:06d}.jpg"
            dst_image = output_dir / "images" / "val" / new_name
            shutil.copy(image_path, dst_image)

            # 対応するラベルファイルがあれば処理
            label_src = image_path.with_suffix(".txt")
            if label_src.exists():
                label_dst = output_dir / "labels" / "val" / f"val_{idx:06d}.txt"
                shutil.copy(label_src, label_dst)

    def _create_yolo_label(self, annotation: dict, label_path: Path, image_path: Path):
        """YOLO形式のラベルファイルを作成"""
        # 画像サイズを取得
        image = cv2.imread(str(image_path))
        if image is None:
            return

        height, width = image.shape[:2]

        with open(label_path, "w") as f:
            # バウンディングボックスがある場合
            if "bboxes" in annotation:
                for bbox, class_name in zip(
                    annotation["bboxes"], annotation.get("class_names", []), strict=False
                ):
                    if class_name in self.class_names:
                        class_id = self.class_names.index(class_name)

                        # YOLO形式に変換 (x_center, y_center, width, height)
                        x1, y1, x2, y2 = bbox
                        x_center = (x1 + x2) / 2 / width
                        y_center = (y1 + y2) / 2 / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height

                        # 正規化された値が有効な範囲内か確認
                        if all(0 <= v <= 1 for v in [x_center, y_center, bbox_width, bbox_height]):
                            f.write(
                                f"{class_id} {x_center:.6f} {y_center:.6f} "
                                f"{bbox_width:.6f} {bbox_height:.6f}\n"
                            )

            # 単一のクラスラベルの場合（画像全体）
            elif "class" in annotation and annotation["class"] in self.class_names:
                class_id = self.class_names.index(annotation["class"])
                # 画像全体をバウンディングボックスとする
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    def _create_dataset_yaml(self, data_dir: Path, yaml_path: Path):
        """データセット設定ファイルを作成"""
        config = {
            "path": str(data_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(self.class_names)),
            "nc": len(self.class_names),
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "models/yolov8",
        name: str = "mahjong_tiles",
        **kwargs,
    ) -> dict[str, Any]:
        """
        YOLOv8モデルの訓練

        Args:
            data_yaml: データセット設定ファイルのパス
            epochs: エポック数
            batch_size: バッチサイズ
            imgsz: 入力画像サイズ
            project: プロジェクトディレクトリ
            name: 実行名
            **kwargs: その他のパラメータ

        Returns:
            訓練結果
        """
        print("YOLOv8訓練を開始します...")
        print(f"データ: {data_yaml}")
        print(f"エポック: {epochs}")
        print(f"バッチサイズ: {batch_size}")
        print(f"画像サイズ: {imgsz}")
        print(f"デバイス: {self.device}")

        # デフォルトパラメータ
        default_params = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": self.device,
            "project": project,
            "name": name,
            "exist_ok": True,
            # 最適化設定
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            # データ拡張（YOLOv8内蔵）
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            # その他の設定
            "close_mosaic": 10,
            "amp": True,  # 自動混合精度
            "patience": 50,  # 早期停止
            "save": True,
            "save_period": 10,
            "val": True,
            "plots": True,
            "verbose": True,
        }

        # カスタムパラメータで上書き
        default_params.update(kwargs)

        # 訓練実行
        results = self.model.train(**default_params)

        # 最良モデルのパスを保存
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            self.model_path = str(best_model_path)
            print(f"最良モデルを保存: {self.model_path}")

        return results

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> list[dict[str, Any]]:
        """
        画像から麻雀牌を検出

        Args:
            image: 入力画像（BGR）
            conf_threshold: 信頼度閾値
            iou_threshold: NMS閾値
            max_det: 最大検出数

        Returns:
            検出結果のリスト
        """
        results = self.model(
            image, conf=conf_threshold, iou=iou_threshold, max_det=max_det, verbose=False
        )

        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # バウンディングボックス
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                    # クラスと信頼度
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])

                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": self.class_names[cls]
                        if cls < len(self.class_names)
                        else "unknown",
                    }
                    detections.append(detection)

        return detections

    def predict_batch(self, images: list[np.ndarray], **kwargs) -> list[list[dict]]:
        """
        複数画像のバッチ予測

        Args:
            images: 画像のリスト
            **kwargs: predict()メソッドのパラメータ

        Returns:
            各画像の検出結果
        """
        batch_results = []

        for image in images:
            detections = self.predict(image, **kwargs)
            batch_results.append(detections)

        return batch_results

    def evaluate(self, data_yaml: str | None = None) -> dict[str, float]:
        """
        モデルの評価

        Args:
            data_yaml: 評価用データセット（Noneの場合は訓練時のデータを使用）

        Returns:
            評価メトリクス
        """
        if self.model_path and Path(self.model_path).exists():
            # 最良モデルを読み込み
            self.model = YOLO(self.model_path)

        # 検証データでの評価
        metrics = self.model.val(data=data_yaml)

        return {
            "mAP": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "mAP75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "fitness": float(metrics.fitness),
        }

    def export_model(self, format: str = "onnx", **kwargs) -> str:
        """
        モデルをエクスポート

        Args:
            format: エクスポート形式（'onnx', 'torchscript', 'coreml', など）
            **kwargs: エクスポートパラメータ

        Returns:
            エクスポートされたモデルのパス
        """
        if self.model_path and Path(self.model_path).exists():
            # 最良モデルを読み込み
            self.model = YOLO(self.model_path)

        # エクスポート実行
        path = self.model.export(format=format, **kwargs)

        return str(path)

    def get_model_info(self) -> dict[str, Any]:
        """モデル情報を取得"""
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }

        if hasattr(self.model, "names"):
            info["trained_classes"] = self.model.names

        if hasattr(self.model, "args"):
            info["training_args"] = vars(self.model.args)

        return info

    def visualize_predictions(
        self,
        image: np.ndarray,
        detections: list[dict],
        output_path: str | None = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        予測結果を可視化

        Args:
            image: 入力画像
            detections: 検出結果
            output_path: 保存先パス
            show: 画像を表示するか

        Returns:
            描画された画像
        """
        result_image = image.copy()

        # カラーマップ（シンプルな実装）
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3))

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            # バウンディングボックスの描画
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = colors[class_id % len(colors)].astype(int).tolist()

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # ラベルの描画
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # ラベル背景
            cv2.rectangle(
                result_image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1
            )

            # ラベルテキスト
            cv2.putText(
                result_image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # 保存
        if output_path:
            cv2.imwrite(output_path, result_image)

        # 表示
        if show:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Detections: {len(detections)}")
            plt.show()

        return result_image


# 便利な関数
def create_yolov8_dataset_from_labelme(
    labelme_dir: str, output_dir: str, class_mapping: dict[str, str] | None = None
):
    """
    LabelMe形式からYOLOv8データセットを作成

    Args:
        labelme_dir: LabelMeアノテーションのディレクトリ
        output_dir: 出力ディレクトリ
        class_mapping: クラス名のマッピング
    """
    # 実装は省略（必要に応じて追加）
    pass
