#!/usr/bin/env python3
"""
MPS（Metal Performance Shaders）対応のテストスクリプト
Apple SiliconのMacでGPU加速が利用できるか確認する
"""

import sys
from pathlib import Path

import torch

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.device_utils import get_available_device, get_device_info


def test_device_detection():
    """デバイス検出機能のテスト"""
    print("=== デバイス検出テスト ===\n")

    # デバイス情報を取得
    device_info = get_device_info()

    print("利用可能なデバイス:")
    print(f"- CUDA: {device_info['cuda']['available']}")
    if device_info["cuda"]["available"]:
        print(f"  デバイス数: {device_info['cuda']['device_count']}")
        for dev in device_info["cuda"]["devices"]:
            print(f"  - {dev['name']} (メモリ: {dev['total_memory'] / 1024**3:.1f} GB)")

    print(f"- MPS: {device_info['mps']['available']}")
    if device_info["mps"]["available"]:
        print(f"  ビルド済み: {device_info['mps']['built']}")
        print(f"  プラットフォーム: {device_info['mps'].get('platform', 'Unknown')}")

    print(f"- CPU: {device_info['cpu']['available']}")
    print(f"  スレッド数: {device_info['cpu']['threads']}")

    print()


def test_device_selection():
    """デバイス選択機能のテスト"""
    print("=== デバイス選択テスト ===\n")

    # 自動選択
    device = get_available_device(preferred_device="auto")
    print(f"自動選択されたデバイス: {device}")

    # 各デバイスを明示的に指定
    for device_type in ["cuda", "mps", "cpu"]:
        device = get_available_device(preferred_device=device_type, fallback_to_cpu=True)
        print(f"{device_type}を指定 → 実際のデバイス: {device}")

    print()


def test_tensor_operations():
    """テンソル演算のテスト"""
    print("=== テンソル演算テスト ===\n")

    device = get_available_device(preferred_device="auto")
    print(f"使用デバイス: {device}\n")

    # 簡単なテンソル演算
    try:
        # テンソルを作成
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)

        # 演算を実行
        z = torch.matmul(x, y)
        result = z.mean().item()

        print("✓ テンソル演算成功")
        print(f"  結果の平均値: {result:.6f}")

        # メモリ使用量を確認（CUDAの場合のみ）
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"  GPU メモリ使用量: {allocated:.1f} MB / {reserved:.1f} MB")

    except Exception as e:
        print(f"✗ テンソル演算失敗: {e}")

    print()


def test_model_training():
    """簡単なモデル訓練のテスト"""
    print("=== モデル訓練テスト ===\n")

    device = get_available_device(preferred_device="auto")
    print(f"使用デバイス: {device}\n")

    try:
        # 簡単なモデルを定義
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1)
        ).to(device)

        # ダミーデータ
        X = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)

        # 損失関数とオプティマイザ
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 数ステップ訓練
        losses = []
        for _step in range(10):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print("✓ モデル訓練成功")
        print(f"  初期損失: {losses[0]:.6f}")
        print(f"  最終損失: {losses[-1]:.6f}")
        print(f"  改善率: {(1 - losses[-1] / losses[0]) * 100:.1f}%")

    except Exception as e:
        print(f"✗ モデル訓練失敗: {e}")

    print()


def test_mixed_precision():
    """混合精度訓練のテスト"""
    print("=== 混合精度訓練テスト ===\n")

    device = get_available_device(preferred_device="auto")
    print(f"使用デバイス: {device}\n")

    if device.type not in ["cuda", "mps"]:
        print("混合精度訓練はGPU（CUDA/MPS）でのみ利用可能です")
        return

    try:
        model = torch.nn.Linear(10, 1).to(device)
        X = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 混合精度の文脈で実行
        autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device_type):
            output = model(X)
            loss = torch.nn.functional.mse_loss(output, y)

        loss.backward()
        optimizer.step()

        print("✓ 混合精度訓練成功")
        print(f"  損失: {loss.item():.6f}")

    except Exception as e:
        print(f"✗ 混合精度訓練失敗: {e}")

    print()


def main():
    """メインテスト実行"""
    print("MPS対応テストを開始します...\n")

    # PyTorchバージョン確認
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"Python バージョン: {sys.version.split()[0]}\n")

    # 各テストを実行
    test_device_detection()
    test_device_selection()
    test_tensor_operations()
    test_model_training()
    test_mixed_precision()

    print("\nテスト完了！")


if __name__ == "__main__":
    main()
