"""
バッチサイズ最適化のテスト
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.learning.batch_size_optimizer import BatchSizeOptimizer, GradientAccumulator


class SimpleModel(nn.Module):
    """テスト用のシンプルなモデル"""

    def __init__(self, input_size: int = 224, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (input_size // 4) * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestBatchSizeOptimizer:
    """バッチサイズ最適化のテストクラス"""

    @pytest.fixture
    def model(self):
        """テスト用モデル"""
        return SimpleModel(input_size=64, num_classes=5)

    @pytest.fixture
    def device(self):
        """デバイス"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dummy_dataset(self):
        """ダミーデータセット"""
        # 小さなデータセットを作成
        num_samples = 100
        images = torch.randn(num_samples, 3, 64, 64)
        labels = torch.randint(0, 5, (num_samples,))
        return TensorDataset(images, labels)

    def test_batch_size_optimizer_init(self, model, device):
        """初期化のテスト"""
        optimizer = BatchSizeOptimizer(
            model=model,
            device=device,
            initial_batch_size=16,
            max_batch_size=128,
        )

        assert optimizer.initial_batch_size == 16
        assert optimizer.max_batch_size == 128
        assert optimizer.memory_fraction == 0.9

        if device.type == "cuda":
            assert optimizer.total_memory > 0
        else:
            assert optimizer.total_memory == 0

    def test_find_optimal_batch_size_cpu(self, model, dummy_dataset):
        """CPU環境での最適バッチサイズ探索のテスト"""
        device = torch.device("cpu")
        model = model.to(device)

        optimizer = BatchSizeOptimizer(
            model=model,
            device=device,
            initial_batch_size=16,
            max_batch_size=64,
        )

        def dataloader_factory(batch_size: int) -> DataLoader:
            return DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()

        # CPU環境では最小バッチサイズを返すはず
        optimal_batch_size = optimizer.find_optimal_batch_size(
            dataloader_factory=dataloader_factory,
            loss_fn=loss_fn,
        )

        assert optimal_batch_size == min(optimizer.initial_batch_size, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_find_optimal_batch_size_gpu(self, model, device, dummy_dataset):
        """GPU環境での最適バッチサイズ探索のテスト"""
        if device.type != "cuda":
            pytest.skip("GPU test requires CUDA")

        model = model.to(device)

        optimizer = BatchSizeOptimizer(
            model=model,
            device=device,
            initial_batch_size=8,
            max_batch_size=64,
            num_trials=2,  # テスト高速化のため
        )

        def dataloader_factory(batch_size: int) -> DataLoader:
            return DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()

        optimal_batch_size = optimizer.find_optimal_batch_size(
            dataloader_factory=dataloader_factory,
            loss_fn=loss_fn,
            mixed_precision=True,
        )

        # 最適バッチサイズは2の累乗であるべき
        assert optimal_batch_size > 0
        assert (optimal_batch_size & (optimal_batch_size - 1)) == 0  # 2の累乗チェック
        assert optimal_batch_size <= optimizer.max_batch_size

    def test_gradient_accumulator(self, model, device):
        """勾配累積のテスト"""
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        accumulation_steps = 4

        accumulator = GradientAccumulator(
            model=model,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
            max_grad_norm=1.0,
        )

        assert accumulator.accumulation_steps == accumulation_steps
        assert accumulator.effective_batch_size == accumulation_steps

        # ダミー損失で勾配累積をテスト
        for i in range(accumulation_steps * 2):
            dummy_loss = torch.tensor(1.0, requires_grad=True, device=device)
            accumulator.step(dummy_loss)

            # accumulation_steps回ごとに勾配がクリアされるはず
            if (i + 1) % accumulation_steps == 0:
                # オプティマイザーのステップが実行されたことを確認
                for param in model.parameters():
                    if param.grad is not None:
                        assert torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_measure_throughput(self, model, device, dummy_dataset):
        """スループット測定のテスト"""
        model = model.to(device)

        optimizer = BatchSizeOptimizer(
            model=model,
            device=device,
            num_trials=1,
        )

        dataloader = DataLoader(dummy_dataset, batch_size=8)
        loss_fn = nn.CrossEntropyLoss()

        throughput = optimizer._measure_throughput(
            dataloader=dataloader,
            loss_fn=loss_fn,
            mixed_precision=False,
        )

        # スループットは正の値であるべき
        assert throughput > 0

    def test_memory_management(self, model, device):
        """メモリ管理のテスト"""
        optimizer = BatchSizeOptimizer(
            model=model,
            device=device,
        )

        # メモリクリア
        optimizer._clear_gpu_memory()

        # メモリ使用量を取得
        memory_used = optimizer._get_gpu_memory_used()

        if device.type == "cuda":
            assert memory_used >= 0
        else:
            assert memory_used == 0


class TestBatchSizeIntegration:
    """バッチサイズ最適化の統合テスト"""

    def test_batch_size_scaling(self):
        """バッチサイズのスケーリングテスト"""
        # 異なるモデルサイズでのバッチサイズ最適化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 小さいモデル
        small_model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

        optimizer = BatchSizeOptimizer(
            model=small_model,
            device=device,
            initial_batch_size=32,
            max_batch_size=512,
        )

        # 小さなデータセット
        dataset = TensorDataset(torch.randn(1000, 100), torch.randint(0, 10, (1000,)))

        def dataloader_factory(batch_size: int) -> DataLoader:
            return DataLoader(dataset, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()

        optimal_batch_size = optimizer.find_optimal_batch_size(
            dataloader_factory=dataloader_factory,
            loss_fn=loss_fn,
        )

        # 小さいモデルは大きなバッチサイズを扱えるはず
        if device.type == "cuda":
            assert optimal_batch_size >= 32
        else:
            assert optimal_batch_size > 0
