"""
リファクタリングされたModelTrainerのテスト

モックを使用してPyTorchの依存関係を適切に分離
"""

from unittest.mock import Mock, patch

import pytest


class TestRefactoredModelTrainer:
    """RefactoredModelTrainerのテストクラス"""

    @pytest.fixture
    def mock_torch(self):
        """PyTorchモックのフィクスチャ"""
        with patch("src.training.learning.refactored_model_trainer.torch") as mock:
            mock.device.return_value = Mock()
            mock.cuda.is_available.return_value = False
            mock.backends.mps.is_available.return_value = False
            yield mock

    @pytest.fixture
    def mock_components(self):
        """コンポーネントモックのフィクスチャ"""
        with (
            patch("src.training.learning.refactored_model_trainer.DataLoaderFactory") as dl_mock,
            patch("src.training.learning.refactored_model_trainer.CheckpointManager") as cp_mock,
            patch("src.training.learning.refactored_model_trainer.MetricsCalculator") as mc_mock,
            patch("src.training.learning.refactored_model_trainer.LearningScheduler") as ls_mock,
        ):
            yield {
                "data_loader_factory": dl_mock,
                "checkpoint_manager": cp_mock,
                "metrics_calculator": mc_mock,
                "learning_scheduler": ls_mock,
            }

    @pytest.fixture
    def config_manager(self):
        """設定管理のモック"""
        config_manager = Mock()
        config_manager.get_config.return_value = {
            "ai": {
                "training": {
                    "device": "cpu",
                    "batch_size": 32,
                    "epochs": 10,
                    "learning_rate": 0.001,
                }
            },
            "training": {"training_root": "data/training"},
        }
        return config_manager

    @pytest.fixture
    def trainer(self, config_manager, mock_torch, mock_components):
        """トレーナーインスタンス"""
        with patch(
            "src.training.learning.refactored_model_trainer.get_available_device"
        ) as mock_device:
            mock_device.return_value = Mock()

            # TORCH_AVAILABLEをTrueに設定
            with patch("src.training.learning.refactored_model_trainer.TORCH_AVAILABLE", True):
                from src.training.learning.refactored_model_trainer import RefactoredModelTrainer

                trainer = RefactoredModelTrainer(config_manager)
                return trainer

    def test_initialization(self, trainer):
        """初期化のテスト"""
        assert trainer is not None
        assert hasattr(trainer, "data_loader_factory")
        assert hasattr(trainer, "checkpoint_manager")
        assert trainer.training_sessions == {}
        assert trainer.stop_flags == {}

    def test_components_separation(self, trainer):
        """コンポーネントの分離を確認"""
        # 各コンポーネントが独立していることを確認
        assert hasattr(trainer, "data_loader_factory")
        assert hasattr(trainer, "checkpoint_manager")
        assert hasattr(trainer, "scheduler")

        # ModelTrainerクラスが以下のメソッドを持たないことを確認
        # （これらは分離されたコンポーネントの責務）
        assert not hasattr(trainer, "_calculate_entropy")
        assert not hasattr(trainer, "_calculate_top_k_confidence")
        assert not hasattr(trainer, "_backup_best_model")

    def test_create_optimizer(self, trainer):
        """オプティマイザー作成のテスト"""
        with patch("src.training.learning.refactored_model_trainer.optim") as optim_mock:
            model = Mock()
            model.parameters.return_value = []

            # Adamオプティマイザー
            config = Mock(optimizer_type="adam", learning_rate=0.001)
            trainer._create_optimizer(model, config)
            optim_mock.Adam.assert_called_once()

            # SGDオプティマイザー
            optim_mock.reset_mock()
            config.optimizer_type = "sgd"
            trainer._create_optimizer(model, config)
            optim_mock.SGD.assert_called_once()

    def test_create_criterion(self, trainer):
        """損失関数作成のテスト"""
        with patch("src.training.learning.refactored_model_trainer.nn") as nn_mock:
            # 分類タスク
            config = Mock(model_type="classification")
            trainer._create_criterion(config)
            nn_mock.CrossEntropyLoss.assert_called_once()

            # 検出タスク
            nn_mock.reset_mock()
            config.model_type = "detection"
            trainer._create_criterion(config)
            nn_mock.MSELoss.assert_called_once()

    def test_session_management(self, trainer):
        """セッション管理のテスト"""
        session_id = "test_session"

        # セッション初期化
        trainer._initialize_session(session_id)
        assert session_id in trainer.training_sessions
        assert trainer.training_sessions[session_id]["status"] == "running"
        assert not trainer.stop_flags[session_id]

        # 訓練停止
        trainer.stop_training(session_id)
        assert trainer.stop_flags[session_id]
        assert trainer.training_sessions[session_id]["status"] == "stopping"

        # 進捗取得
        progress = trainer.get_training_progress(session_id)
        assert progress is not None
        assert progress["status"] == "stopping"

    @patch("src.training.learning.refactored_model_trainer.CallbackManager")
    @patch("src.training.learning.refactored_model_trainer.ProgressBarCallback")
    @patch("src.training.learning.refactored_model_trainer.ModelCheckpointCallback")
    @patch("src.training.learning.refactored_model_trainer.TensorBoardCallback")
    @patch("src.training.learning.refactored_model_trainer.TrainingHistoryCallback")
    def test_callbacks_integration(
        self, mock_history, mock_tb, mock_checkpoint, mock_progress, mock_manager, trainer
    ):
        """コールバック統合のテスト"""
        model = Mock()
        optimizer = Mock()
        config = Mock(model_type="classification", epochs=10, use_tensorboard=True)

        # CallbackManagerのモックインスタンスを設定
        manager_instance = Mock()
        manager_instance.callbacks = []
        mock_manager.return_value = manager_instance

        callbacks = trainer._setup_callbacks(model, optimizer, config, "test_session", 100, 10)

        assert callbacks == manager_instance
        mock_manager.assert_called_once()

        # 各コールバックが作成されたことを確認
        mock_progress.assert_called_once()
        mock_checkpoint.assert_called_once()
        mock_tb.assert_called_once()
        mock_history.assert_called_once()

    def test_update_training_history(self, trainer):
        """訓練履歴更新のテスト"""
        session_id = "test_session"
        trainer._initialize_session(session_id)

        metrics = {"val_loss": 0.5, "val_accuracy": 0.9, "train_loss": 0.6, "train_accuracy": 0.85}

        trainer._update_training_history(session_id, 0, metrics)

        history = trainer.training_sessions[session_id]["training_history"]
        assert len(history) == 1
        assert history[0]["epoch"] == 0
        assert history[0]["val_loss"] == 0.5
        assert history[0]["val_accuracy"] == 0.9

        # ベストスコアの更新を確認
        assert trainer.training_sessions[session_id]["best_loss"] == 0.5
        assert trainer.training_sessions[session_id]["best_accuracy"] == 0.9

    def test_prepare_final_results(self, trainer):
        """最終結果準備のテスト"""
        session_id = "test_session"
        trainer._initialize_session(session_id)

        # 履歴を追加
        trainer.training_sessions[session_id]["training_history"] = [
            {"epoch": 0, "loss": 0.5},
            {"epoch": 1, "loss": 0.3},
        ]
        trainer.training_sessions[session_id]["best_loss"] = 0.3
        trainer.training_sessions[session_id]["best_accuracy"] = 0.95

        # CheckpointManagerのfind_best_checkpointをモック
        trainer.checkpoint_manager.find_best_checkpoint.return_value = "/path/to/best/model.pt"

        config = Mock()
        results = trainer._prepare_final_results(session_id, config)

        assert results["best_model_path"] == "/path/to/best/model.pt"
        assert results["final_metrics"]["best_val_loss"] == 0.3
        assert results["final_metrics"]["best_val_accuracy"] == 0.95
        assert results["final_metrics"]["total_epochs"] == 2
        assert "training_time" in results["final_metrics"]
        assert len(results["training_history"]) == 2


class TestMetricsCalculator:
    """MetricsCalculatorの独立したテスト"""

    @patch("src.training.learning.components.metrics_calculator.TORCH_AVAILABLE", True)
    @patch("src.training.learning.components.metrics_calculator.torch")
    def test_metrics_calculator_update(self, mock_torch):
        """メトリクス計算器の更新テスト"""
        from src.training.learning.components import MetricsCalculator

        calculator = MetricsCalculator(num_classes=10, model_type="classification")

        # モックの設定
        mock_outputs = Mock()
        mock_outputs.data = Mock()

        mock_targets = Mock()
        mock_targets.size.return_value = 32

        # torch.maxのモック
        mock_predicted = Mock()
        mock_torch.max.return_value = (Mock(), mock_predicted)

        # 正解数の計算をモック
        mock_comparison = Mock()
        mock_comparison.sum.return_value.item.return_value = 25
        mock_predicted.__eq__ = Mock(return_value=mock_comparison)

        # CPUとnumpyのモック
        mock_cpu_result = Mock()
        mock_cpu_result.numpy.return_value = [1, 2, 3, 4, 5]  # 予測結果の例
        mock_predicted.cpu.return_value = mock_cpu_result
        mock_targets.cpu.return_value = mock_cpu_result

        # バッチ精度の計算のためにtargets.size(0)をモック
        mock_targets.size.return_value = 32

        loss = 0.5
        batch_size = 32

        batch_metrics = calculator.update(loss, mock_outputs, mock_targets, batch_size)

        assert "loss" in batch_metrics
        assert batch_metrics["loss"] == loss
        assert "accuracy" in batch_metrics

    def test_metrics_calculator_without_torch(self):
        """PyTorchなしでのメトリクス計算器テスト"""
        with patch("src.training.learning.components.metrics_calculator.TORCH_AVAILABLE", False):
            from src.training.learning.components import MetricsCalculator

            calculator = MetricsCalculator(num_classes=10, model_type="classification")

            loss = 0.5
            outputs = None
            targets = None
            batch_size = 32

            batch_metrics = calculator.update(loss, outputs, targets, batch_size)

            assert "loss" in batch_metrics
            assert batch_metrics["loss"] == loss

            epoch_metrics = calculator.compute_epoch_metrics()
            assert epoch_metrics["loss"] == loss
