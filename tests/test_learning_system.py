"""
学習システムのテスト

フェーズ2: 学習システムの各コンポーネントをテストする
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.annotation_data import (
    AnnotationData,
    BoundingBox,
    FrameAnnotation,
    TileAnnotation,
    VideoAnnotation,
)
from src.training.learning.learning_scheduler import (
    HyperParameter,
    LearningScheduler,
)
from src.training.learning.model_evaluator import ModelEvaluator
from src.training.learning.model_trainer import ModelTrainer, TileDataset
from src.training.learning.training_manager import TrainingConfig, TrainingManager
from src.utils.config import ConfigManager


class TestLearningSystem(unittest.TestCase):
    """学習システムテストクラス"""

    def setUp(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()

        # テスト用設定を上書き
        test_config = {
            "training": {
                "training_root": str(Path(self.temp_dir) / "training"),
                "dataset_root": str(Path(self.temp_dir) / "dataset"),
                "database_path": str(Path(self.temp_dir) / "test.db"),
                "num_tile_classes": 16,
            }
        }
        self.config_manager.config.update(test_config)

        # サンプルデータを作成
        self.sample_data = self._create_sample_annotation_data()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_annotation_data(self) -> AnnotationData:
        """サンプルアノテーションデータを作成"""
        annotation_data = AnnotationData()

        video_annotation = VideoAnnotation(
            video_id="test_video_001",
            video_path="test_video.mp4",
            video_name="テスト動画",
            duration=60.0,
            fps=30.0,
            width=1920,
            height=1080,
            frames=[],
        )

        # 5フレームのサンプルデータ
        tile_types = ["1m", "2m", "3m", "1p", "2p"]
        for i in range(5):
            frame = FrameAnnotation(
                frame_id=f"frame_{i}",
                image_path=f"frame_{i}.jpg",
                image_width=1920,
                image_height=1080,
                timestamp=i * 2.0,
                tiles=[],
                is_valid=True,
            )

            # 各フレームに3個の牌
            for j in range(3):
                tile = TileAnnotation(
                    tile_id=tile_types[j],
                    bbox=BoundingBox(x1=100 + j * 50, y1=200, x2=140 + j * 50, y2=260),
                    confidence=0.9,
                    area_type="hand",
                )
                frame.tiles.append(tile)

            video_annotation.frames.append(frame)

        annotation_data.video_annotations[video_annotation.video_id] = video_annotation
        return annotation_data


class TestTrainingManager(TestLearningSystem):
    """TrainingManagerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        training_manager = TrainingManager(self.config_manager)

        self.assertIsNotNone(training_manager.dataset_manager)
        self.assertIsNotNone(training_manager.model_manager)
        self.assertIsNotNone(training_manager.model_trainer)
        self.assertIsNotNone(training_manager.learning_scheduler)
        self.assertIsNotNone(training_manager.model_evaluator)

        # ディレクトリが作成されているか確認
        self.assertTrue(training_manager.sessions_dir.exists())
        self.assertTrue(training_manager.experiments_dir.exists())
        self.assertTrue(training_manager.checkpoints_dir.exists())

    def test_training_config_creation(self):
        """学習設定作成テスト"""
        config = TrainingConfig(
            model_type="classification",
            model_name="test_model",
            dataset_version_id="test_version",
            epochs=10,
            batch_size=16,
            learning_rate=0.001,
        )

        self.assertEqual(config.model_type, "classification")
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 0.001)

    def test_session_management(self):
        """セッション管理テスト"""
        training_manager = TrainingManager(self.config_manager)

        # 初期状態では空
        sessions = training_manager.list_sessions()
        self.assertEqual(len(sessions), 0)

        # セッション情報の保存・読み込みテスト
        training_manager._save_sessions()
        training_manager._load_sessions()


class TestModelTrainer(TestLearningSystem):
    """ModelTrainerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        trainer = ModelTrainer(self.config_manager)

        self.assertIsNotNone(trainer.scheduler)
        self.assertEqual(str(trainer.device), "cuda" if torch.cuda.is_available() else "cpu")

    def test_tile_dataset_creation(self):
        """TileDatasetの作成テスト"""
        # 分類用データセット
        dataset = TileDataset(self.sample_data, model_type="classification")

        self.assertGreater(len(dataset), 0)
        self.assertGreater(dataset.num_classes, 0)

        # データセットからサンプルを取得
        if len(dataset) > 0:
            image, label = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(label, torch.Tensor)

    def test_detection_dataset_creation(self):
        """検出用データセットの作成テスト"""
        dataset = TileDataset(self.sample_data, model_type="detection")

        self.assertGreater(len(dataset), 0)

        if len(dataset) > 0:
            image, target = dataset[0]
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)
            self.assertEqual(target.shape[0], 5)  # bbox(4) + class(1)


class TestLearningScheduler(TestLearningSystem):
    """LearningSchedulerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        scheduler = LearningScheduler(self.config_manager)

        self.assertGreater(len(scheduler.hyperparameters), 0)
        self.assertTrue(scheduler.scheduler_root.exists())

    def test_hyperparameter_definition(self):
        """ハイパーパラメータ定義テスト"""
        param = HyperParameter(
            name="learning_rate", param_type="float", min_value=1e-5, max_value=1e-1, log_scale=True
        )

        self.assertEqual(param.name, "learning_rate")
        self.assertEqual(param.param_type, "float")
        self.assertEqual(param.min_value, 1e-5)
        self.assertEqual(param.max_value, 1e-1)
        self.assertTrue(param.log_scale)

    def test_random_parameter_sampling(self):
        """ランダムパラメータサンプリングテスト"""
        scheduler = LearningScheduler(self.config_manager)

        params = scheduler._sample_random_parameters()

        self.assertIsInstance(params, dict)
        self.assertGreater(len(params), 0)

        # 各パラメータが適切な範囲内にあるかチェック
        for param_name, value in params.items():
            if param_name in scheduler.hyperparameters:
                param_def = scheduler.hyperparameters[param_name]

                if param_def.param_type == "float" or param_def.param_type == "int":
                    self.assertGreaterEqual(value, param_def.min_value)
                    self.assertLessEqual(value, param_def.max_value)
                elif param_def.param_type == "choice":
                    self.assertIn(value, param_def.choices)

    def test_optimization_trial_management(self):
        """最適化試行管理テスト"""
        scheduler = LearningScheduler(self.config_manager)

        # 試行を作成
        trial_ids = scheduler._random_search({}, 3, "accuracy")

        self.assertEqual(len(trial_ids), 3)
        self.assertEqual(len(scheduler.optimization_trials), 3)

        # 試行結果を更新
        trial_id = trial_ids[0]
        scheduler.update_trial_result(trial_id, 0.85, {"accuracy": 0.85, "loss": 0.5})

        trial = scheduler.optimization_trials[trial_id]
        self.assertEqual(trial.score, 0.85)
        self.assertEqual(trial.status, "completed")
        self.assertIsNotNone(trial.end_time)

    def test_best_parameters_retrieval(self):
        """最良パラメータ取得テスト"""
        scheduler = LearningScheduler(self.config_manager)

        # 複数の試行を作成して結果を設定
        trial_ids = scheduler._random_search({}, 3, "accuracy")

        scores = [0.8, 0.9, 0.7]
        for i, trial_id in enumerate(trial_ids):
            scheduler.update_trial_result(trial_id, scores[i])

        # 最良パラメータを取得
        best_params = scheduler.get_best_parameters(n_best=2)

        self.assertEqual(len(best_params), 2)
        self.assertEqual(best_params[0]["score"], 0.9)  # 最高スコア
        self.assertEqual(best_params[1]["score"], 0.8)  # 2番目


class TestModelEvaluator(TestLearningSystem):
    """ModelEvaluatorのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        evaluator = ModelEvaluator(self.config_manager)

        self.assertTrue(evaluator.evaluation_root.exists())
        self.assertTrue(evaluator.reports_dir.exists())
        self.assertTrue(evaluator.visualizations_dir.exists())
        self.assertTrue(evaluator.metrics_dir.exists())

    def test_learning_curves_creation(self):
        """学習曲線作成テスト"""
        evaluator = ModelEvaluator(self.config_manager)

        # サンプル学習履歴
        history = [
            {
                "epoch": 0,
                "train_loss": 2.0,
                "val_loss": 1.8,
                "train_accuracy": 0.3,
                "val_accuracy": 0.35,
            },
            {
                "epoch": 1,
                "train_loss": 1.5,
                "val_loss": 1.4,
                "train_accuracy": 0.5,
                "val_accuracy": 0.52,
            },
            {
                "epoch": 2,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "train_accuracy": 0.7,
                "val_accuracy": 0.68,
            },
        ]

        try:
            curves_path = evaluator.create_learning_curves(history)
            # matplotlibが利用可能な場合のみテスト
            if curves_path:
                self.assertTrue(Path(curves_path).exists())
        except ImportError:
            # matplotlibが利用できない場合はスキップ
            pass

    def test_iou_calculation(self):
        """IoU計算テスト"""
        evaluator = ModelEvaluator(self.config_manager)

        # テストケース: 完全一致
        pred_boxes = [[0, 0, 10, 10]]
        target_boxes = [[0, 0, 10, 10]]
        ious = evaluator._calculate_iou_batch(pred_boxes, target_boxes)
        self.assertAlmostEqual(ious[0], 1.0, places=5)

        # テストケース: 半分重複
        pred_boxes = [[0, 0, 10, 10]]
        target_boxes = [[5, 0, 15, 10]]
        ious = evaluator._calculate_iou_batch(pred_boxes, target_boxes)
        self.assertAlmostEqual(ious[0], 1 / 3, places=5)  # IoU = 50/(100+100-50) = 1/3

        # テストケース: 重複なし
        pred_boxes = [[0, 0, 5, 5]]
        target_boxes = [[10, 10, 15, 15]]
        ious = evaluator._calculate_iou_batch(pred_boxes, target_boxes)
        self.assertEqual(ious[0], 0.0)

    def test_top_k_accuracy_calculation(self):
        """Top-k精度計算テスト"""
        evaluator = ModelEvaluator(self.config_manager)

        import numpy as np

        # テストケース: 3クラス、2サンプル
        probs = np.array(
            [
                [0.1, 0.7, 0.2],  # 正解クラス1
                [0.3, 0.2, 0.5],  # 正解クラス2
            ]
        )
        targets = np.array([1, 2])

        # Top-1精度
        top1_acc = evaluator._calculate_top_k_accuracy(probs, targets, k=1)
        self.assertEqual(top1_acc, 1.0)  # 両方とも正解

        # Top-2精度
        top2_acc = evaluator._calculate_top_k_accuracy(probs, targets, k=2)
        self.assertEqual(top2_acc, 1.0)  # 両方ともTop-2に含まれる


class TestIntegration(TestLearningSystem):
    """統合テスト"""

    def test_full_workflow_simulation(self):
        """全体ワークフローのシミュレーション"""
        # 1. データセット管理
        from src.training.dataset_manager import DatasetManager

        dataset_manager = DatasetManager(self.config_manager)

        # アノテーションデータを保存
        success = dataset_manager.save_annotation_data(self.sample_data)
        self.assertTrue(success)

        # バージョンを作成
        version_id = dataset_manager.create_dataset_version(
            self.sample_data, "v1.0.0", "テスト用データセット"
        )
        self.assertIsNotNone(version_id)

        # 2. 学習設定
        TrainingConfig(
            model_type="classification",
            model_name="test_model",
            dataset_version_id=version_id,
            epochs=2,  # テスト用に短縮
            batch_size=2,
            learning_rate=0.01,
        )

        # 3. 学習管理
        training_manager = TrainingManager(self.config_manager)

        # セッション管理のテスト
        sessions = training_manager.list_sessions()
        len(sessions)

        # 4. 評価システム
        evaluator = ModelEvaluator(self.config_manager)

        # サンプル履歴で学習曲線作成をテスト
        sample_history = [
            {
                "epoch": 0,
                "train_loss": 1.0,
                "val_loss": 1.1,
                "train_accuracy": 0.5,
                "val_accuracy": 0.48,
            },
            {
                "epoch": 1,
                "train_loss": 0.8,
                "val_loss": 0.9,
                "train_accuracy": 0.7,
                "val_accuracy": 0.68,
            },
        ]

        try:
            curves_path = evaluator.create_learning_curves(sample_history)
            # 成功した場合のみチェック
            if curves_path:
                self.assertTrue(Path(curves_path).exists())
        except Exception:
            # 依存関係の問題でスキップ
            pass

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 存在しないモデルパスでの評価
        evaluator = ModelEvaluator(self.config_manager)
        metrics = evaluator.evaluate_model(
            "nonexistent_model.pt", self.sample_data, "classification"
        )
        self.assertEqual(len(metrics), 0)  # エラー時は空の辞書

        # 不正な設定での学習スケジューラー
        scheduler = LearningScheduler(self.config_manager)

        # 存在しない試行IDでの結果更新
        scheduler.update_trial_result("nonexistent_trial", 0.5)
        # エラーログが出力されるが、例外は発生しない


if __name__ == "__main__":
    # テスト実行
    unittest.main(verbosity=2)
