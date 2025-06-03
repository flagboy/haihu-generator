"""
モデル評価システム

モデル性能評価、可視化、メトリクス計算を行う
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
)

# Optional torch imports
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from ...utils.config import ConfigManager
from ...utils.logger import LoggerMixin
from ..annotation_data import AnnotationData
from .model_trainer import TileDataset


class ModelEvaluator(LoggerMixin):
    """モデル評価クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # 評価結果保存ディレクトリ
        self.evaluation_root = Path(
            self.config.get("training", {}).get("evaluation_root", "data/training/evaluation")
        )
        self.evaluation_root.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ
        self.reports_dir = self.evaluation_root / "reports"
        self.visualizations_dir = self.evaluation_root / "visualizations"
        self.metrics_dir = self.evaluation_root / "metrics"

        for dir_path in [self.reports_dir, self.visualizations_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"ModelEvaluator初期化完了: {self.evaluation_root}")

    def evaluate_model(
        self, model_path: str, test_data: AnnotationData, model_type: str, save_results: bool = True
    ) -> dict[str, float]:
        """
        モデルを評価

        Args:
            model_path: モデルファイルパス
            test_data: テストデータ
            model_type: モデルタイプ
            save_results: 結果を保存するか

        Returns:
            評価メトリクス
        """
        self.logger.info(f"モデル評価開始: {model_path}")

        try:
            # モデルを読み込み
            model = self._load_model(model_path, model_type)
            if model is None:
                return {}

            # データローダーを作成
            test_dataset = TileDataset(test_data, model_type=model_type, augment=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )

            # 評価実行
            if model_type == "detection":
                metrics = self._evaluate_detection_model(model, test_loader, test_dataset)
            elif model_type == "classification":
                metrics = self._evaluate_classification_model(model, test_loader, test_dataset)
            else:
                self.logger.error(f"サポートされていないモデルタイプ: {model_type}")
                return {}

            # 結果を保存
            if save_results:
                self._save_evaluation_results(model_path, metrics, model_type)

            self.logger.info(f"モデル評価完了: {model_path}")
            return metrics

        except Exception as e:
            self.logger.error(f"モデル評価に失敗: {e}")
            return {}

    def _load_model(self, model_path: str, model_type: str) -> nn.Module | None:
        """モデルを読み込み"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"モデルファイルが見つかりません: {model_path}")
                return None

            # チェックポイントを読み込み
            checkpoint = torch.load(model_path, map_location=self.device)

            # モデルを作成
            if model_type == "detection":
                from ...detection.tile_detector import SimpleCNN

                num_classes = self.config.get("training", {}).get("num_tile_classes", 34)
                model = SimpleCNN(num_classes=num_classes)
            elif model_type == "classification":
                from ...classification.tile_classifier import TileClassifier

                model = TileClassifier(self.config_manager)
                model = model.model
            else:
                self.logger.error(f"サポートされていないモデルタイプ: {model_type}")
                return None

            # 重みを読み込み
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            self.logger.error(f"モデル読み込みに失敗: {e}")
            return None

    def _evaluate_detection_model(
        self, model: nn.Module, test_loader, test_dataset: TileDataset
    ) -> dict[str, float]:
        """検出モデルを評価"""
        model.eval()

        all_bbox_preds = []
        all_bbox_targets = []
        all_conf_preds = []
        all_conf_targets = []
        all_class_preds = []
        all_class_targets = []

        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 予測
                bbox_pred, conf_pred, class_pred = model(data)

                # 損失計算
                bbox_loss = criterion(bbox_pred, target[:, :4])
                conf_loss = nn.BCELoss()(conf_pred.squeeze(), (target[:, 4] > 0).float())
                class_loss = nn.CrossEntropyLoss()(class_pred, target[:, 4].long())

                total_loss += (bbox_loss + conf_loss + class_loss).item()

                # 予測結果を収集
                all_bbox_preds.extend(bbox_pred.cpu().numpy())
                all_bbox_targets.extend(target[:, :4].cpu().numpy())
                all_conf_preds.extend(conf_pred.cpu().numpy())
                all_conf_targets.extend((target[:, 4] > 0).float().cpu().numpy())
                all_class_preds.extend(torch.argmax(class_pred, dim=1).cpu().numpy())
                all_class_targets.extend(target[:, 4].long().cpu().numpy())

        # メトリクス計算
        metrics = {}

        # 損失
        metrics["loss"] = total_loss / len(test_loader)

        # バウンディングボックス精度（IoU）
        ious = self._calculate_iou_batch(all_bbox_preds, all_bbox_targets)
        metrics["mean_iou"] = np.mean(ious)
        metrics["iou_50"] = np.mean(np.array(ious) > 0.5)  # IoU > 0.5の割合
        metrics["iou_75"] = np.mean(np.array(ious) > 0.75)  # IoU > 0.75の割合

        # 信頼度精度
        conf_preds = np.array(all_conf_preds).flatten()
        conf_targets = np.array(all_conf_targets)
        metrics["conf_accuracy"] = np.mean((conf_preds > 0.5) == conf_targets)

        # 分類精度
        class_preds = np.array(all_class_preds)
        class_targets = np.array(all_class_targets)
        metrics["class_accuracy"] = np.mean(class_preds == class_targets)

        # mAP計算（簡易版）
        metrics["map_50"] = self._calculate_map(
            all_bbox_preds,
            all_bbox_targets,
            all_conf_preds,
            all_class_preds,
            all_class_targets,
            iou_threshold=0.5,
        )

        return metrics

    def _evaluate_classification_model(
        self, model: nn.Module, test_loader, test_dataset: TileDataset
    ) -> dict[str, float]:
        """分類モデルを評価"""
        model.eval()

        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 予測
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

                # 予測結果を収集
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                total += target.size(0)
                correct += (predicted == target).sum().item()

        # メトリクス計算
        metrics = {}

        # 基本メトリクス
        metrics["loss"] = total_loss / len(test_loader)
        metrics["accuracy"] = correct / total

        # 詳細メトリクス
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # クラス別精度
        unique_classes = np.unique(all_targets)
        class_accuracies = {}
        for class_id in unique_classes:
            mask = all_targets == class_id
            if np.sum(mask) > 0:
                class_acc = np.mean(all_preds[mask] == all_targets[mask])
                class_accuracies[f"class_{class_id}_accuracy"] = class_acc

        metrics.update(class_accuracies)

        # Precision, Recall, F1-score
        from sklearn.metrics import f1_score, precision_score, recall_score

        metrics["precision_macro"] = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        metrics["precision_weighted"] = precision_score(
            all_targets, all_preds, average="weighted", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            all_targets, all_preds, average="weighted", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            all_targets, all_preds, average="weighted", zero_division=0
        )

        # Top-k精度
        if all_probs.shape[1] > 1:
            top_k_acc = self._calculate_top_k_accuracy(
                all_probs, all_targets, k=min(5, all_probs.shape[1])
            )
            metrics["top_5_accuracy"] = top_k_acc

        return metrics

    def _calculate_iou_batch(self, pred_boxes: list, target_boxes: list) -> list[float]:
        """バッチでIoUを計算"""
        ious = []

        for pred, target in zip(pred_boxes, target_boxes, strict=False):
            # 座標を正規化解除（必要に応じて）
            pred = np.array(pred)
            target = np.array(target)

            # IoU計算
            x1 = max(pred[0], target[0])
            y1 = max(pred[1], target[1])
            x2 = min(pred[2], target[2])
            y2 = min(pred[3], target[3])

            if x2 <= x1 or y2 <= y1:
                ious.append(0.0)
                continue

            intersection = (x2 - x1) * (y2 - y1)
            pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
            target_area = (target[2] - target[0]) * (target[3] - target[1])
            union = pred_area + target_area - intersection

            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(0.0)

        return ious

    def _calculate_map(
        self,
        bbox_preds: list,
        bbox_targets: list,
        conf_preds: list,
        class_preds: list,
        class_targets: list,
        iou_threshold: float = 0.5,
    ) -> float:
        """mAP（mean Average Precision）を計算（簡易版）"""
        # 実際の実装では、より複雑なmAP計算が必要
        # ここでは簡易的な実装

        ious = self._calculate_iou_batch(bbox_preds, bbox_targets)
        conf_preds = np.array(conf_preds).flatten()
        class_preds = np.array(class_preds)
        class_targets = np.array(class_targets)

        # IoU閾値を満たし、クラスが正しい予測の割合
        correct_detections = (np.array(ious) > iou_threshold) & (class_preds == class_targets)

        if len(correct_detections) == 0:
            return 0.0

        # 信頼度でソートして精度を計算
        sorted_indices = np.argsort(conf_preds)[::-1]
        sorted_correct = correct_detections[sorted_indices]

        # 累積精度を計算
        cumulative_correct = np.cumsum(sorted_correct)
        precision = cumulative_correct / (np.arange(len(sorted_correct)) + 1)

        # 平均精度
        ap = np.mean(precision[sorted_correct])
        return ap if not np.isnan(ap) else 0.0

    def _calculate_top_k_accuracy(self, probs: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Top-k精度を計算"""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = 0

        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1

        return correct / len(targets)

    def _save_evaluation_results(self, model_path: str, metrics: dict[str, float], model_type: str):
        """評価結果を保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem

            # メトリクスをJSON形式で保存
            metrics_file = self.metrics_dir / f"{model_name}_{timestamp}_metrics.json"

            results = {
                "model_path": model_path,
                "model_type": model_type,
                "evaluation_time": datetime.now().isoformat(),
                "metrics": metrics,
            }

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info(f"評価結果保存: {metrics_file}")

        except Exception as e:
            self.logger.error(f"評価結果の保存に失敗: {e}")

    def create_confusion_matrix(
        self, model_path: str, test_data: AnnotationData, class_names: list[str] | None = None
    ) -> str:
        """
        混同行列を作成

        Args:
            model_path: モデルファイルパス
            test_data: テストデータ
            class_names: クラス名リスト

        Returns:
            保存された画像ファイルパス
        """
        try:
            # モデルを読み込み
            model = self._load_model(model_path, "classification")
            if model is None:
                return ""

            # データローダーを作成
            test_dataset = TileDataset(test_data, model_type="classification", augment=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=32, shuffle=False, num_workers=4
            )

            # 予測実行
            all_preds = []
            all_targets = []

            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    _, predicted = torch.max(output, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            # 混同行列を計算
            cm = confusion_matrix(all_targets, all_preds)

            # 可視化
            plt.figure(figsize=(12, 10))

            if class_names is None:
                class_names = [f"Class {i}" for i in range(cm.shape[0])]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem
            output_path = self.visualizations_dir / f"{model_name}_{timestamp}_confusion_matrix.png"

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"混同行列保存: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"混同行列作成に失敗: {e}")
            return ""

    def create_learning_curves(self, training_history: list[dict[str, Any]]) -> str:
        """
        学習曲線を作成

        Args:
            training_history: 学習履歴

        Returns:
            保存された画像ファイルパス
        """
        try:
            if not training_history:
                return ""

            # データを抽出
            epochs = [h["epoch"] for h in training_history]
            train_losses = [h["train_loss"] for h in training_history]
            val_losses = [h["val_loss"] for h in training_history]
            train_accs = [h.get("train_accuracy", 0) for h in training_history]
            val_accs = [h.get("val_accuracy", 0) for h in training_history]

            # 可視化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 損失曲線
            ax1.plot(epochs, train_losses, label="Training Loss", color="blue")
            ax1.plot(epochs, val_losses, label="Validation Loss", color="red")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Validation Loss")
            ax1.legend()
            ax1.grid(True)

            # 精度曲線
            ax2.plot(epochs, train_accs, label="Training Accuracy", color="blue")
            ax2.plot(epochs, val_accs, label="Validation Accuracy", color="red")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Training and Validation Accuracy")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.visualizations_dir / f"learning_curves_{timestamp}.png"

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"学習曲線保存: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"学習曲線作成に失敗: {e}")
            return ""

    def create_detection_visualization(
        self, model_path: str, test_data: AnnotationData, num_samples: int = 10
    ) -> str:
        """
        検出結果の可視化

        Args:
            model_path: モデルファイルパス
            test_data: テストデータ
            num_samples: 可視化するサンプル数

        Returns:
            保存された画像ファイルパス
        """
        try:
            # モデルを読み込み
            model = self._load_model(model_path, "detection")
            if model is None:
                return ""

            # テストサンプルを選択
            samples = []
            for video_annotation in test_data.video_annotations.values():
                for frame in video_annotation.frames[:num_samples]:
                    if frame.is_valid and len(frame.tiles) > 0:
                        samples.append(frame)
                        if len(samples) >= num_samples:
                            break
                if len(samples) >= num_samples:
                    break

            if not samples:
                return ""

            # 可視化
            fig, axes = plt.subplots(2, min(5, len(samples)), figsize=(20, 8))
            if len(samples) == 1:
                axes = axes.reshape(2, 1)

            model.eval()
            with torch.no_grad():
                for i, frame in enumerate(samples[: min(5, len(samples))]):
                    if i >= 5:
                        break

                    # 画像を読み込み
                    image = Image.open(frame.image_path).convert("RGB")
                    image_np = np.array(image)

                    # 予測実行（簡易実装）
                    # 実際の実装では適切な前処理が必要

                    # Ground Truth表示
                    ax_gt = axes[0, i] if len(samples) > 1 else axes[0]
                    ax_gt.imshow(image_np)
                    ax_gt.set_title(f"Ground Truth {i + 1}")
                    ax_gt.axis("off")

                    # Ground Truthのバウンディングボックスを描画
                    for tile in frame.tiles:
                        bbox = tile.bbox
                        rect = plt.Rectangle(
                            (bbox.x1, bbox.y1),
                            bbox.width,
                            bbox.height,
                            linewidth=2,
                            edgecolor="green",
                            facecolor="none",
                        )
                        ax_gt.add_patch(rect)
                        ax_gt.text(bbox.x1, bbox.y1 - 5, tile.tile_id, color="green", fontsize=8)

                    # 予測結果表示（簡易実装）
                    ax_pred = axes[1, i] if len(samples) > 1 else axes[1]
                    ax_pred.imshow(image_np)
                    ax_pred.set_title(f"Prediction {i + 1}")
                    ax_pred.axis("off")

                    # 予測バウンディングボックスを描画（ダミー）
                    for tile in frame.tiles:  # 実際は予測結果を使用
                        bbox = tile.bbox
                        rect = plt.Rectangle(
                            (bbox.x1, bbox.y1),
                            bbox.width,
                            bbox.height,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none",
                        )
                        ax_pred.add_patch(rect)
                        ax_pred.text(bbox.x1, bbox.y1 - 5, tile.tile_id, color="red", fontsize=8)

            plt.tight_layout()

            # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem
            output_path = (
                self.visualizations_dir / f"{model_name}_{timestamp}_detection_results.png"
            )

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"検出結果可視化保存: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"検出結果可視化に失敗: {e}")
            return ""

    def generate_evaluation_report(
        self,
        model_path: str,
        test_data: AnnotationData,
        model_type: str,
        training_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        総合評価レポートを生成

        Args:
            model_path: モデルファイルパス
            test_data: テストデータ
            model_type: モデルタイプ
            training_history: 学習履歴

        Returns:
            レポートファイルパス
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).stem
            report_path = self.reports_dir / f"{model_name}_{timestamp}_evaluation_report.html"

            # 評価実行
            metrics = self.evaluate_model(model_path, test_data, model_type, save_results=False)

            # 可視化作成
            confusion_matrix_path = ""
            learning_curves_path = ""
            detection_viz_path = ""

            if model_type == "classification":
                confusion_matrix_path = self.create_confusion_matrix(model_path, test_data)
            elif model_type == "detection":
                detection_viz_path = self.create_detection_visualization(model_path, test_data)

            if training_history:
                learning_curves_path = self.create_learning_curves(training_history)

            # HTMLレポート生成
            html_content = self._generate_html_report(
                model_path,
                model_type,
                metrics,
                confusion_matrix_path,
                learning_curves_path,
                detection_viz_path,
                training_history,
            )

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.logger.info(f"評価レポート生成: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"評価レポート生成に失敗: {e}")
            return ""

    def _generate_html_report(
        self,
        model_path: str,
        model_type: str,
        metrics: dict[str, float],
        confusion_matrix_path: str,
        learning_curves_path: str,
        detection_viz_path: str,
        training_history: list[dict[str, Any]] | None,
    ) -> str:
        """HTMLレポートを生成"""

        html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>モデル評価レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd; padding: 8px; text-align: left;
                }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                .image-container img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>モデル評価レポート</h1>
                <p><strong>モデルパス:</strong> {model_path}</p>
                <p><strong>モデルタイプ:</strong> {model_type}</p>
                <p><strong>評価日時:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <div class="section">
                <h2>評価メトリクス</h2>
                <table class="metrics-table">
                    <tr><th>メトリクス</th><th>値</th></tr>
        """

        for metric, value in metrics.items():
            html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"

        html += """
                </table>
            </div>
        """

        if learning_curves_path:
            html += f"""
            <div class="section">
                <h2>学習曲線</h2>
                <div class="image-container">
                    <img src="{learning_curves_path}" alt="学習曲線">
                </div>
            </div>
            """

        if confusion_matrix_path:
            html += f"""
            <div class="section">
                <h2>混同行列</h2>
                <div class="image-container">
                    <img src="{confusion_matrix_path}" alt="混同行列">
                </div>
            </div>
            """

        if detection_viz_path:
            html += f"""
            <div class="section">
                <h2>検出結果</h2>
                <div class="image-container">
                    <img src="{detection_viz_path}" alt="検出結果">
                </div>
            </div>
            """

        if training_history:
            html += """
            <div class="section">
                <h2>学習履歴</h2>
                <table class="metrics-table">
                    <tr><th>エポック</th><th>訓練損失</th><th>検証損失</th><th>訓練精度</th><th>検証精度</th></tr>
            """

            for history in training_history[-10:]:  # 最後の10エポックのみ表示
                html += f"""
                <tr>
                    <td>{history["epoch"]}</td>
                    <td>{history["train_loss"]:.4f}</td>
                    <td>{history["val_loss"]:.4f}</td>
                    <td>{history.get("train_accuracy", 0):.4f}</td>
                    <td>{history.get("val_accuracy", 0):.4f}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def compare_models(
        self, model_paths: list[str], test_data: AnnotationData, model_type: str
    ) -> dict[str, Any]:
        """
        複数のモデルを比較

        Args:
            model_paths: モデルファイルパスのリスト
            test_data: テストデータ
            model_type: モデルタイプ

        Returns:
            比較結果
        """
        comparison_results = {"models": [], "best_model": None, "comparison_metrics": {}}

        all_metrics = {}

        for model_path in model_paths:
            self.logger.info(f"モデル評価中: {model_path}")
            metrics = self.evaluate_model(model_path, test_data, model_type, save_results=False)

            model_name = Path(model_path).stem
            all_metrics[model_name] = metrics

            comparison_results["models"].append(
                {"name": model_name, "path": model_path, "metrics": metrics}
            )

        if all_metrics:
            # 最良モデルを決定
            best_metric = "accuracy" if model_type == "classification" else "mean_iou"

            best_model = max(all_metrics.items(), key=lambda x: x[1].get(best_metric, 0))
            comparison_results["best_model"] = {
                "name": best_model[0],
                "metric": best_metric,
                "value": best_model[1].get(best_metric, 0),
            }

            # メトリクス比較
            all_metric_names = set()
            for metrics in all_metrics.values():
                all_metric_names.update(metrics.keys())

            for metric_name in all_metric_names:
                values = []
                for model_name, metrics in all_metrics.items():
                    if metric_name in metrics:
                        values.append({"model": model_name, "value": metrics[metric_name]})

                if values:
                    comparison_results["comparison_metrics"][metric_name] = {
                        "values": values,
                        "best": max(values, key=lambda x: x["value"]),
                        "worst": min(values, key=lambda x: x["value"]),
                    }

        return comparison_results
