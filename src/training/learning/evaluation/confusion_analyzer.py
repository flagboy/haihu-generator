"""
混同行列分析専用クラス
"""

import numpy as np
from sklearn.metrics import confusion_matrix

from ....utils.logger import LoggerMixin


class ConfusionAnalyzer(LoggerMixin):
    """混同行列分析専用クラス"""

    def __init__(self, class_names: list[str] | None = None):
        """
        初期化

        Args:
            class_names: クラス名のリスト
        """
        self.class_names = class_names
        self.logger.info("ConfusionAnalyzer初期化完了")

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        混同行列を計算

        Args:
            y_true: 正解ラベル
            y_pred: 予測ラベル

        Returns:
            混同行列
        """
        return confusion_matrix(y_true, y_pred)

    def analyze_confusion_patterns(self, cm: np.ndarray) -> dict[str, list[tuple[str, str, int]]]:
        """
        混同パターンを分析

        Args:
            cm: 混同行列

        Returns:
            混同パターンの辞書
        """
        patterns = {
            "most_confused": [],  # 最も混同されやすいペア
            "perfect_predictions": [],  # 完璧に予測されたクラス
            "worst_predictions": [],  # 最も予測精度が低いクラス
        }

        n_classes = cm.shape[0]

        # クラス名がない場合は番号を使用
        if self.class_names is None:
            self.class_names = [str(i) for i in range(n_classes)]

        # 最も混同されやすいペアを抽出（対角線以外）
        confusion_pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append(
                        (self.class_names[i], self.class_names[j], int(cm[i, j]))
                    )

        # 混同数でソート
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        patterns["most_confused"] = confusion_pairs[:10]  # 上位10ペア

        # クラスごとの精度を計算
        class_accuracy = []
        for i in range(n_classes):
            total = np.sum(cm[i, :])
            if total > 0:
                accuracy = cm[i, i] / total
                class_accuracy.append((self.class_names[i], accuracy))

        # 精度でソート
        class_accuracy.sort(key=lambda x: x[1], reverse=True)

        # 完璧に予測されたクラス（精度100%）
        patterns["perfect_predictions"] = [
            (name, 1.0) for name, acc in class_accuracy if acc == 1.0
        ]

        # 最も予測精度が低いクラス（下位5つ）
        patterns["worst_predictions"] = [
            (name, acc) for name, acc in class_accuracy[-5:] if acc < 1.0
        ]

        return patterns

    def calculate_class_wise_metrics(self, cm: np.ndarray) -> dict[str, dict[str, float]]:
        """
        クラスごとのメトリクスを計算

        Args:
            cm: 混同行列

        Returns:
            クラスごとのメトリクス
        """
        n_classes = cm.shape[0]
        metrics = {}

        # クラス名がない場合は番号を使用
        if self.class_names is None:
            self.class_names = [str(i) for i in range(n_classes)]

        for i in range(n_classes):
            class_name = self.class_names[i]

            # True Positive, False Positive, False Negative, True Negative
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn

            # メトリクス計算
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics[class_name] = {
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "specificity": float(specificity),
                "support": int(tp + fn),
            }

        return metrics

    def get_misclassification_details(
        self, cm: np.ndarray, threshold: int = 5
    ) -> list[dict[str, any]]:
        """
        誤分類の詳細を取得

        Args:
            cm: 混同行列
            threshold: 表示する最小誤分類数

        Returns:
            誤分類の詳細リスト
        """
        details = []
        n_classes = cm.shape[0]

        # クラス名がない場合は番号を使用
        if self.class_names is None:
            self.class_names = [str(i) for i in range(n_classes)]

        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] >= threshold:
                    details.append(
                        {
                            "true_class": self.class_names[i],
                            "predicted_class": self.class_names[j],
                            "count": int(cm[i, j]),
                            "percentage": float(cm[i, j] / np.sum(cm[i, :]) * 100)
                            if np.sum(cm[i, :]) > 0
                            else 0.0,
                        }
                    )

        # 誤分類数でソート
        details.sort(key=lambda x: x["count"], reverse=True)

        return details
