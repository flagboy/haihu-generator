"""
メトリクス計算専用クラス
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from ....utils.logger import LoggerMixin


class MetricsCalculator(LoggerMixin):
    """メトリクス計算専用クラス"""

    def __init__(self, class_names: list[str] | None = None):
        """
        初期化

        Args:
            class_names: クラス名のリスト
        """
        self.class_names = class_names
        self.logger.info("MetricsCalculator初期化完了")

    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        精度を計算

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            精度
        """
        return float(accuracy_score(y_true, y_pred))

    def calculate_precision_recall_f1(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """
        Precision, Recall, F1スコアを計算

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            (precision, recall, f1_score)の辞書タプル
        """
        # クラスごとのメトリクスを計算
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # クラス名がない場合は番号を使用
        if self.class_names is None:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            self.class_names = [str(label) for label in unique_labels]

        # 辞書形式に変換
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}

        for i, class_name in enumerate(self.class_names):
            if i < len(precision):
                precision_dict[class_name] = float(precision[i])
                recall_dict[class_name] = float(recall[i])
                f1_dict[class_name] = float(f1[i])

        # マクロ平均も追加
        precision_dict["macro_avg"] = float(np.mean(precision))
        recall_dict["macro_avg"] = float(np.mean(recall))
        f1_dict["macro_avg"] = float(np.mean(f1))

        # 重み付き平均も追加
        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        precision_dict["weighted_avg"] = float(precision_weighted)
        recall_dict["weighted_avg"] = float(recall_weighted)
        f1_dict["weighted_avg"] = float(f1_weighted)

        return precision_dict, recall_dict, f1_dict

    def calculate_per_class_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        クラスごとの精度を計算

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            クラスごとの精度
        """
        per_class_accuracy = {}

        # クラス名がない場合は番号を使用
        if self.class_names is None:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            self.class_names = [str(label) for label in unique_labels]

        for i, class_name in enumerate(self.class_names):
            # 該当クラスのインデックスを取得
            class_mask = y_true == i
            if np.any(class_mask):
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                per_class_accuracy[class_name] = float(class_accuracy)
            else:
                per_class_accuracy[class_name] = 0.0

        return per_class_accuracy

    def calculate_class_distribution(self, y_true: np.ndarray) -> dict[str, int]:
        """
        クラス分布を計算

        Args:
            y_true: 正解ラベル

        Returns:
            クラスごとのサンプル数
        """
        unique_labels, counts = np.unique(y_true, return_counts=True)

        distribution = {}
        for label, count in zip(unique_labels, counts, strict=False):
            if self.class_names and label < len(self.class_names):
                class_name = self.class_names[label]
            else:
                class_name = str(label)
            distribution[class_name] = int(count)

        return distribution

    def calculate_confidence_statistics(
        self, confidence_scores: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        信頼度スコアの統計を計算

        Args:
            confidence_scores: 予測の信頼度スコア
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            信頼度統計
        """
        correct_mask = y_true == y_pred

        stats = {
            "mean_confidence": float(np.mean(confidence_scores)),
            "std_confidence": float(np.std(confidence_scores)),
            "min_confidence": float(np.min(confidence_scores)),
            "max_confidence": float(np.max(confidence_scores)),
            "mean_confidence_correct": float(np.mean(confidence_scores[correct_mask]))
            if np.any(correct_mask)
            else 0.0,
            "mean_confidence_incorrect": float(np.mean(confidence_scores[~correct_mask]))
            if np.any(~correct_mask)
            else 0.0,
        }

        return stats

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        分類レポートを生成

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            分類レポート文字列
        """
        return classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)

    def calculate_iou_batch(
        self, pred_boxes: list[list[float]], target_boxes: list[list[float]]
    ) -> list[float]:
        """
        バッチ単位でIoUを計算

        Args:
            pred_boxes: 予測ボックスのリスト [[x1, y1, x2, y2], ...]
            target_boxes: 正解ボックスのリスト [[x1, y1, x2, y2], ...]

        Returns:
            各ペアのIoUスコアのリスト
        """
        ious = []
        for pred_box, target_box in zip(pred_boxes, target_boxes, strict=False):
            iou = self._calculate_iou(pred_box, target_box)
            ious.append(iou)
        return ious

    def _calculate_iou(self, box1: list[float], box2: list[float]) -> float:
        """
        2つのボックス間のIoUを計算

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoUスコア
        """
        # 交差領域の座標
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # 交差領域の面積
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # 各ボックスの面積
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 和集合の面積
        union = area1 + area2 - intersection

        # IoU
        return intersection / union if union > 0 else 0.0

    def calculate_top_k_accuracy(self, probs: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
        """
        Top-k精度を計算

        Args:
            probs: 予測確率 (n_samples, n_classes)
            targets: 正解ラベル (n_samples,)
            k: 上位k個を考慮

        Returns:
            Top-k精度
        """
        # 上位k個のインデックスを取得
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]

        # 正解がTop-kに含まれるかチェック
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_preds[i]:
                correct += 1

        return correct / len(targets)
