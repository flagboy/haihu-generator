#!/usr/bin/env python3
"""
SceneDatasetのパフォーマンス問題をデバッグするためのスクリプト
"""

import sys
import time
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader  # noqa: E402

from src.training.game_scene.learning.scene_dataset import SceneDataset  # noqa: E402


def test_dataset_performance():
    """データセットのパフォーマンスをテスト"""
    print("🚀 SceneDataset パフォーマンステスト開始")
    print("=" * 60)

    # データセットを初期化
    print("📊 データセット初期化中...")
    start_time = time.time()

    try:
        dataset = SceneDataset(
            db_path="web_interface/data/training/game_scene_labels.db",
            cache_dir="web_interface/data/training/game_scene_cache",
            split="train",
        )
        init_time = time.time() - start_time
        print(f"✅ データセット初期化完了: {init_time:.3f}s")
        print(f"📈 データセットサイズ: {len(dataset)}サンプル")

    except Exception as e:
        print(f"❌ データセット初期化エラー: {e}")
        return

    # 最初の5サンプルを個別にテスト
    print("\n" + "=" * 60)
    print("🔍 個別サンプルテスト（最初の5サンプル）")
    print("=" * 60)

    total_time = 0
    success_count = 0

    for i in range(min(5, len(dataset))):
        print(f"\n📝 サンプル {i} テスト開始...")
        sample_start = time.time()

        try:
            image, label = dataset[i]
            sample_time = time.time() - sample_start
            total_time += sample_time
            success_count += 1

            print(f"✅ サンプル {i} 成功: {sample_time:.3f}s")
            print(f"   - 画像shape: {image.shape}")
            print(f"   - ラベル: {label}")

        except Exception as e:
            sample_time = time.time() - sample_start
            total_time += sample_time
            print(f"❌ サンプル {i} エラー: {sample_time:.3f}s, error={e}")

    if success_count > 0:
        avg_time = total_time / success_count
        print("\n📊 個別テスト結果:")
        print(f"   - 成功: {success_count}/5")
        print(f"   - 平均時間: {avg_time:.3f}s")
        print(f"   - 総時間: {total_time:.3f}s")

    # DataLoaderテスト（少量）
    print("\n" + "=" * 60)
    print("🔄 DataLoader テスト（バッチサイズ=4, 2バッチ）")
    print("=" * 60)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # シングルプロセスでテスト
        )

        batch_count = 0
        dataloader_start = time.time()

        for batch_idx, (images, labels) in enumerate(dataloader):
            batch_time = time.time() - dataloader_start
            batch_count += 1

            print(f"✅ バッチ {batch_idx}: {batch_time:.3f}s")
            print(f"   - 画像shape: {images.shape}")
            print(f"   - ラベル: {labels}")

            if batch_count >= 2:  # 2バッチのみテスト
                break

            dataloader_start = time.time()

        print(f"\n📊 DataLoader テスト完了: {batch_count}バッチ処理")

    except Exception as e:
        print(f"❌ DataLoader テストエラー: {e}")

    print("\n" + "=" * 60)
    print("🏁 テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset_performance()
