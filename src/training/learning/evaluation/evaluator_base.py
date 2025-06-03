"""
評価システムの基底クラスと共通機能
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....utils.config import ConfigManager
from ....utils.logger import LoggerMixin


@dataclass
class EvaluationResult:
    """評価結果データクラス"""

    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1_score: dict[str, float]
    confusion_matrix: Any  # numpy array
    per_class_accuracy: dict[str, float]
    total_samples: int
    correct_predictions: int
    class_distribution: dict[str, int]
    confidence_scores: dict[str, float]
    evaluation_time: float
    additional_metrics: dict[str, Any]


class BaseEvaluator(ABC, LoggerMixin):
    """評価器の基底クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.training_config = self.config.get("training", {})
        self.evaluation_config = self.training_config.get("evaluation", {})

        # 評価用ディレクトリ
        self.evaluation_dir = Path(
            self.training_config.get("evaluation_dir", "data/training/evaluation")
        )
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 可視化設定
        self.visualization_enabled = self.evaluation_config.get("visualization_enabled", True)
        self.save_confusion_matrix = self.evaluation_config.get("save_confusion_matrix", True)
        self.save_per_class_metrics = self.evaluation_config.get("save_per_class_metrics", True)

        self.logger.info("BaseEvaluator初期化完了")

    @abstractmethod
    def evaluate(self, model: Any, dataset: Any) -> EvaluationResult:
        """
        モデルを評価

        Args:
            model: 評価対象モデル
            dataset: 評価用データセット

        Returns:
            評価結果
        """
        pass

    def save_evaluation_result(self, result: EvaluationResult, output_path: Path) -> None:
        """
        評価結果を保存

        Args:
            result: 評価結果
            output_path: 出力パス
        """
        # 基本メトリクスを辞書形式に変換
        result_dict = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "total_samples": result.total_samples,
            "correct_predictions": result.correct_predictions,
            "class_distribution": result.class_distribution,
            "confidence_scores": result.confidence_scores,
            "evaluation_time": result.evaluation_time,
            "additional_metrics": result.additional_metrics,
            "per_class_accuracy": result.per_class_accuracy,
        }

        # JSONとして保存
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)

        self.logger.info(f"評価結果を保存: {output_path}")

    def load_evaluation_result(self, input_path: Path) -> EvaluationResult:
        """
        評価結果を読み込み

        Args:
            input_path: 入力パス

        Returns:
            評価結果
        """
        import json

        with open(input_path, encoding="utf-8") as f:
            result_dict = json.load(f)

        # EvaluationResultに変換（confusion_matrixは除く）
        return EvaluationResult(
            accuracy=result_dict["accuracy"],
            precision=result_dict["precision"],
            recall=result_dict["recall"],
            f1_score=result_dict["f1_score"],
            confusion_matrix=None,  # 別途読み込み必要
            per_class_accuracy=result_dict["per_class_accuracy"],
            total_samples=result_dict["total_samples"],
            correct_predictions=result_dict["correct_predictions"],
            class_distribution=result_dict["class_distribution"],
            confidence_scores=result_dict["confidence_scores"],
            evaluation_time=result_dict["evaluation_time"],
            additional_metrics=result_dict.get("additional_metrics", {}),
        )
