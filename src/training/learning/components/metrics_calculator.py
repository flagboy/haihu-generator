"""
メトリクス計算器

訓練および検証中のメトリクス計算責務を分離し、
様々なメトリクスを一貫した方法で計算する
"""

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from ....utils.logger import LoggerMixin


class MetricsCalculator(LoggerMixin):
    """メトリクス計算クラス"""

    def __init__(self, num_classes: int, model_type: str = "classification"):
        """
        初期化

        Args:
            num_classes: クラス数
            model_type: モデルタイプ（classification/detection）
        """
        self.num_classes = num_classes
        self.model_type = model_type
        self.reset()

    def reset(self):
        """メトリクスの累積値をリセット"""
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct_predictions = 0

        # 分類用の詳細メトリクス
        if self.model_type == "classification":
            self.all_predictions = []
            self.all_targets = []

        # 検出用の詳細メトリクス
        elif self.model_type == "detection":
            self.total_iou = 0.0
            self.detection_count = 0

    def update(self, loss: float, outputs: Any, targets: Any, batch_size: int) -> dict[str, float]:
        """
        バッチごとのメトリクスを更新

        Args:
            loss: 損失値
            outputs: モデル出力
            targets: 正解ラベル
            batch_size: バッチサイズ

        Returns:
            現在のバッチのメトリクス
        """
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        batch_metrics = {"loss": loss}

        if self.model_type == "classification":
            batch_metrics.update(self._update_classification_metrics(outputs, targets))
        elif self.model_type == "detection":
            batch_metrics.update(self._update_detection_metrics(outputs, targets))

        return batch_metrics

    def _update_classification_metrics(self, outputs: Any, targets: Any) -> dict[str, float]:
        """分類メトリクスの更新"""
        if not TORCH_AVAILABLE:
            return {}

        # 予測クラスを取得
        _, predicted = torch.max(outputs.data, 1)

        # 正解数を更新
        correct = (predicted == targets).sum().item()
        self.correct_predictions += correct

        # 詳細メトリクス用にデータを保存
        if hasattr(predicted, "cpu"):
            self.all_predictions.extend(predicted.cpu().numpy())
            self.all_targets.extend(targets.cpu().numpy())

        # バッチの精度を計算
        batch_accuracy = correct / targets.size(0)

        return {"accuracy": batch_accuracy}

    def _update_detection_metrics(self, outputs: Any, targets: Any) -> dict[str, float]:
        """検出メトリクスの更新"""
        if not TORCH_AVAILABLE:
            return {}

        # 簡易的なIoU計算（実際のYOLOではより複雑）
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            bbox_pred, conf_pred, _ = outputs

            # バウンディングボックスのIoUを計算
            iou = self._calculate_iou(bbox_pred, targets[:, :4])
            self.total_iou += iou.sum().item()
            self.detection_count += len(targets)

            return {"mean_iou": iou.mean().item()}

        return {}

    def _calculate_iou(self, boxes1: Any, boxes2: Any) -> Any:
        """IoU（Intersection over Union）を計算"""
        if not TORCH_AVAILABLE:
            return 0.0

        # 簡易的なIoU計算
        # 実際の実装ではより正確な計算が必要
        intersection = torch.min(boxes1[:, 2:], boxes2[:, 2:]) - torch.max(
            boxes1[:, :2], boxes2[:, :2]
        )
        intersection = torch.clamp(intersection, min=0)
        intersection_area = intersection[:, 0] * intersection[:, 1]

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        union_area = area1 + area2 - intersection_area
        iou = intersection_area / (union_area + 1e-6)

        return iou

    def compute_epoch_metrics(self) -> dict[str, float]:
        """
        エポック全体のメトリクスを計算

        Returns:
            エポックのメトリクス
        """
        if self.total_samples == 0:
            return {"loss": 0.0}

        metrics = {
            "loss": self.total_loss / self.total_samples,
        }

        if self.model_type == "classification":
            metrics["accuracy"] = self.correct_predictions / self.total_samples

            # 詳細メトリクスを計算
            if len(self.all_predictions) > 0:
                detailed_metrics = self._compute_classification_details()
                metrics.update(detailed_metrics)

        elif self.model_type == "detection":
            if self.detection_count > 0:
                metrics["mean_iou"] = self.total_iou / self.detection_count

        return metrics

    def _compute_classification_details(self) -> dict[str, float]:
        """分類の詳細メトリクスを計算"""
        try:
            # NumPy配列に変換
            predictions = np.array(self.all_predictions)
            targets = np.array(self.all_targets)

            # Precision, Recall, F1スコアを計算
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions, average="weighted", zero_division=0
            )

            # クラスごとの精度
            class_accuracies = {}
            for class_id in range(self.num_classes):
                mask = targets == class_id
                if mask.sum() > 0:
                    class_correct = (predictions[mask] == class_id).sum()
                    class_accuracies[f"class_{class_id}_accuracy"] = float(
                        class_correct / mask.sum()
                    )

            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                **class_accuracies,
            }

        except Exception as e:
            self.logger.warning(f"詳細メトリクスの計算に失敗: {e}")
            return {}

    def get_confusion_matrix(self) -> np.ndarray | None:
        """
        混同行列を取得（分類タスクのみ）

        Returns:
            混同行列
        """
        if self.model_type != "classification" or len(self.all_predictions) == 0:
            return None

        try:
            return confusion_matrix(
                self.all_targets, self.all_predictions, labels=list(range(self.num_classes))
            )
        except Exception as e:
            self.logger.error(f"混同行列の計算に失敗: {e}")
            return None

    def format_metrics(self, metrics: dict[str, float], prefix: str = "") -> str:
        """
        メトリクスを整形して文字列に変換

        Args:
            metrics: メトリクス辞書
            prefix: プレフィックス（例: "train", "val"）

        Returns:
            整形された文字列
        """
        formatted_parts = []

        # 主要メトリクスを先に表示
        primary_metrics = ["loss", "accuracy", "mean_iou", "precision", "recall", "f1_score"]

        for key in primary_metrics:
            if key in metrics:
                metric_name = f"{prefix}_{key}" if prefix else key
                formatted_parts.append(f"{metric_name}={metrics[key]:.4f}")

        # その他のメトリクス
        for key, value in metrics.items():
            if key not in primary_metrics and not key.startswith("class_"):
                metric_name = f"{prefix}_{key}" if prefix else key
                formatted_parts.append(f"{metric_name}={value:.4f}")

        return ", ".join(formatted_parts)
