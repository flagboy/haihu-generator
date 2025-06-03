"""
リファクタリングされたモデル評価システム
分割されたクラスを統合して使用
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None

from ...utils.config import ConfigManager
from ..annotation_data import AnnotationData

# 分割されたクラスをインポート
from .evaluation import (
    BaseEvaluator,
    ConfusionAnalyzer,
    EvaluationResult,
    MetricsCalculator,
    ModelComparator,
    ReportGenerator,
    VisualizationGenerator,
)
from .model_trainer import TileDataset


class ModelEvaluator(BaseEvaluator):
    """リファクタリングされたモデル評価クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        super().__init__(config_manager or ConfigManager())

        # 分割されたコンポーネントを初期化
        self.metrics_calculator = MetricsCalculator()
        self.confusion_analyzer = ConfusionAnalyzer()
        self.visualization_generator = VisualizationGenerator(
            self.evaluation_dir / "visualizations"
        )
        self.report_generator = ReportGenerator(self.evaluation_dir / "reports")
        self.model_comparator = ModelComparator()

        # PyTorch設定
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU.")

        self.logger.info(f"ModelEvaluator初期化完了: {self.evaluation_dir}")

    def evaluate_model(
        self, model_path: str, test_data: AnnotationData, model_type: str, save_results: bool = True
    ) -> dict[str, Any]:
        """
        モデルを評価

        Args:
            model_path: モデルファイルパス
            test_data: テストデータ
            model_type: モデルタイプ ("detection" or "classification")
            save_results: 結果を保存するか

        Returns:
            評価結果辞書
        """
        self.logger.info(f"モデル評価開始: {model_path}")

        if not TORCH_AVAILABLE:
            self.logger.error("PyTorchが利用できません")
            return {}

        try:
            # モデルを読み込み
            model = self._load_model(model_path, model_type)
            if model is None:
                return {}

            # データセットを作成
            dataset = TileDataset(test_data, model_type=model_type, augment=False)

            # 評価を実行
            evaluation_result = self.evaluate(model, dataset)

            # 結果を保存
            if save_results:
                self._save_evaluation_results(model_path, evaluation_result, model_type)

            # 辞書形式で返す（後方互換性のため）
            return {
                "accuracy": evaluation_result.accuracy,
                "precision": evaluation_result.precision.get("weighted_avg", 0),
                "recall": evaluation_result.recall.get("weighted_avg", 0),
                "f1_score": evaluation_result.f1_score.get("weighted_avg", 0),
                "evaluation_time": evaluation_result.evaluation_time,
            }

        except Exception as e:
            self.logger.error(f"モデル評価に失敗: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return {}

    def evaluate(self, model: Any, dataset: Any) -> EvaluationResult:
        """
        モデルを評価（BaseEvaluatorの抽象メソッドを実装）

        Args:
            model: 評価対象モデル
            dataset: 評価用データセット

        Returns:
            評価結果
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorchが利用できません")

        start_time = time.time()

        # データローダーを作成
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("training", {}).get("batch_size", 32),
            shuffle=False,
            num_workers=4,
        )

        # 予測を収集
        all_predictions = []
        all_labels = []
        all_confidences = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 予測
                outputs = model(images)

                # ソフトマックスを適用
                if outputs.dim() > 1:
                    probs = torch.softmax(outputs, dim=1)
                    confidences, predictions = torch.max(probs, dim=1)
                else:
                    predictions = outputs
                    confidences = torch.ones_like(outputs)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())

        # NumPy配列に変換
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        confidence_scores = np.array(all_confidences)

        # クラス名を取得
        class_names = dataset.get_class_names() if hasattr(dataset, "get_class_names") else None
        if class_names:
            self.metrics_calculator.class_names = class_names
            self.confusion_analyzer.class_names = class_names

        # メトリクスを計算
        accuracy = self.metrics_calculator.calculate_accuracy(y_true, y_pred)
        precision, recall, f1_score = self.metrics_calculator.calculate_precision_recall_f1(
            y_true, y_pred
        )
        per_class_accuracy = self.metrics_calculator.calculate_per_class_accuracy(y_true, y_pred)
        class_distribution = self.metrics_calculator.calculate_class_distribution(y_true)
        confidence_stats = self.metrics_calculator.calculate_confidence_statistics(
            confidence_scores, y_true, y_pred
        )

        # 混同行列を計算
        confusion_matrix = self.confusion_analyzer.calculate_confusion_matrix(y_true, y_pred)

        evaluation_time = time.time() - start_time

        # 評価結果を作成
        return EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_matrix,
            per_class_accuracy=per_class_accuracy,
            total_samples=len(y_true),
            correct_predictions=int(np.sum(y_true == y_pred)),
            class_distribution=class_distribution,
            confidence_scores=confidence_stats,
            evaluation_time=evaluation_time,
            additional_metrics={},
        )

    def _load_model(self, model_path: str, model_type: str) -> Any | None:
        """モデルを読み込み"""
        if not TORCH_AVAILABLE:
            return None

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

                classifier = TileClassifier(self.config_manager)
                model = classifier.model
            else:
                self.logger.error(f"サポートされていないモデルタイプ: {model_type}")
                return None

            # 重みを読み込み
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            self.logger.error(f"モデルの読み込みに失敗: {e}")
            return None

    def _save_evaluation_results(
        self, model_path: str, evaluation_result: EvaluationResult, model_type: str
    ) -> None:
        """評価結果を保存"""
        model_name = Path(model_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 結果を保存
        result_path = self.evaluation_dir / f"{model_name}_{timestamp}_result.json"
        self.save_evaluation_result(evaluation_result, result_path)

        # 可視化を生成
        if self.visualization_enabled:
            # 混同行列
            cm_path = (
                self.evaluation_dir
                / "visualizations"
                / f"{model_name}_{timestamp}_confusion_matrix.png"
            )
            self.visualization_generator.plot_confusion_matrix(
                evaluation_result.confusion_matrix,
                class_names=self.metrics_calculator.class_names,
                title=f"Confusion Matrix - {model_name}",
                save_path=cm_path,
                normalize=True,
            )

            # クラスごとの精度
            accuracy_path = (
                self.evaluation_dir / "visualizations" / f"{model_name}_{timestamp}_accuracy.png"
            )
            self.visualization_generator.plot_per_class_metrics(
                evaluation_result.per_class_accuracy,
                metric_name="Accuracy",
                title=f"Per-Class Accuracy - {model_name}",
                save_path=accuracy_path,
            )

            # 評価サマリー
            summary_path = (
                self.evaluation_dir / "visualizations" / f"{model_name}_{timestamp}_summary.png"
            )
            self.visualization_generator.create_evaluation_summary_plot(
                evaluation_result, save_path=summary_path
            )

        # レポートを生成
        model_info = {
            "model_path": model_path,
            "model_type": model_type,
            "model_name": model_name,
        }

        dataset_info = {
            "total_samples": evaluation_result.total_samples,
            "num_classes": len(evaluation_result.class_distribution),
        }

        # テキストレポート
        report_path = self.evaluation_dir / "reports" / f"{model_name}_{timestamp}_report.txt"
        self.report_generator.generate_evaluation_report(
            evaluation_result, model_info, dataset_info, save_path=report_path
        )

        # JSONレポート
        json_report_path = self.evaluation_dir / "reports" / f"{model_name}_{timestamp}_report.json"
        self.report_generator.generate_json_report(
            evaluation_result, model_info, dataset_info, save_path=json_report_path
        )

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
        results = {}

        for model_path in model_paths:
            self.logger.info(f"評価中: {model_path}")

            # モデルを評価
            model = self._load_model(model_path, model_type)
            if model is None:
                continue

            dataset = TileDataset(test_data, model_type=model_type, augment=False)
            evaluation_result = self.evaluate(model, dataset)

            model_name = Path(model_path).stem
            results[model_name] = evaluation_result

        # 比較結果を生成
        comparison = self.model_comparator.compare_models(results)

        # 比較レポートを生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_report_path = self.evaluation_dir / "reports" / f"comparison_{timestamp}.txt"
        self.report_generator.generate_comparison_report(results, save_path=comparison_report_path)

        # 比較可視化
        if self.visualization_enabled:
            metrics_dict = {
                "Accuracy": {name: r.accuracy for name, r in results.items()},
                "Precision": {
                    name: r.precision.get("weighted_avg", 0) for name, r in results.items()
                },
                "Recall": {name: r.recall.get("weighted_avg", 0) for name, r in results.items()},
                "F1-Score": {
                    name: r.f1_score.get("weighted_avg", 0) for name, r in results.items()
                },
            }

            comparison_plot_path = (
                self.evaluation_dir / "visualizations" / f"comparison_{timestamp}.png"
            )
            self.visualization_generator.plot_metrics_comparison(
                metrics_dict, title="Model Comparison", save_path=comparison_plot_path
            )

        return comparison


# 必要なインポートを追加
