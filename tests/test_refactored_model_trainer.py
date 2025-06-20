"""
リファクタリングされたModelTrainerのテスト

注意: このテストは他のテストとは独立して実行する必要があります。
PyTorchのモックを使用しているため、実際のPyTorchを使用する他のテストに影響を与える可能性があります。
"""

from unittest.mock import MagicMock, Mock, patch  # noqa: E402

import pytest

# テストをスキップする条件を設定
pytest.skip(
    "このテストは他のPyTorchテストと干渉するため、現在スキップしています", allow_module_level=True
)

# PyTorchのモックを設定
torch_mock = MagicMock()
nn_mock = MagicMock()
optim_mock = MagicMock()

# モジュールレベルでモック
import sys  # noqa: E402

sys.modules["torch"] = torch_mock
sys.modules["torch.nn"] = nn_mock
sys.modules["torch.optim"] = optim_mock
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.cuda.amp"] = MagicMock()

from src.training.learning.components import (  # noqa: E402
    CallbackManager,
    CheckpointManager,
    DataLoaderFactory,
    MetricsCalculator,
)
from src.training.learning.refactored_model_trainer import RefactoredModelTrainer  # noqa: E402
from src.utils.config import ConfigManager  # noqa: E402


class TestRefactoredModelTrainer:
    """RefactoredModelTrainerのテストクラス"""

    @pytest.fixture
    def config_manager(self):
        """設定管理のモック"""
        config_manager = Mock(spec=ConfigManager)
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
    def trainer(self, config_manager):
        """トレーナーインスタンス"""
        with patch(
            "src.training.learning.refactored_model_trainer.get_available_device"
        ) as mock_device:
            mock_device.return_value = torch_mock.device("cpu")
            trainer = RefactoredModelTrainer(config_manager)
            return trainer

    def test_initialization(self, trainer):
        """初期化のテスト"""
        assert trainer is not None
        assert isinstance(trainer.data_loader_factory, DataLoaderFactory)
        assert isinstance(trainer.checkpoint_manager, CheckpointManager)
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

    @patch("src.training.learning.refactored_model_trainer.TORCH_AVAILABLE", True)
    def test_callbacks_integration(self, trainer):
        """コールバック統合のテスト"""
        model = Mock()
        optimizer = Mock()
        config = Mock(model_type="classification", epochs=10, use_tensorboard=True)

        callbacks = trainer._setup_callbacks(model, optimizer, config, "test_session", 100, 10)

        assert isinstance(callbacks, CallbackManager)
        assert len(callbacks.callbacks) > 0

        # コールバックの種類を確認
        callback_types = [type(cb).__name__ for cb in callbacks.callbacks]
        assert "ProgressBarCallback" in callback_types
        assert "ModelCheckpointCallback" in callback_types
        assert "TensorBoardCallback" in callback_types
        assert "TrainingHistoryCallback" in callback_types

    def test_metrics_calculator_integration(self):
        """メトリクス計算器の統合テスト"""
        calculator = MetricsCalculator(num_classes=10, model_type="classification")

        # メトリクスの更新
        loss = 0.5
        outputs = torch_mock.randn(32, 10)
        targets = torch_mock.randint(0, 10, (32,))
        batch_size = 32

        # max()の戻り値を設定
        max_values = Mock()
        max_values.data = torch_mock.tensor([5] * 32)
        torch_mock.max.return_value = (Mock(), max_values)

        batch_metrics = calculator.update(loss, outputs, targets, batch_size)

        assert "loss" in batch_metrics
        assert batch_metrics["loss"] == loss

        # エポックメトリクスの計算
        epoch_metrics = calculator.compute_epoch_metrics()
        assert "loss" in epoch_metrics
        assert epoch_metrics["loss"] == loss
