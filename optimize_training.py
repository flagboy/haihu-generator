"""
学習速度最適化スクリプト
お金をかけずに学習を高速化する設定
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.game_scene.learning.scene_dataset import SceneDataset  # noqa: E402
from src.training.game_scene.learning.scene_trainer import SceneTrainer  # noqa: E402


def optimize_training():
    """最適化された学習設定で実行"""

    # 1. 小さなバッチサイズとエポック数
    optimized_config = {
        "epochs": 5,  # 50 -> 5
        "batch_size": 8,  # 32 -> 8
        "learning_rate": 0.001,
        "early_stopping_patience": 3,  # 10 -> 3
        "checkpoint_interval": 1,  # 5 -> 1
    }

    # 2. データローダーの最適化設定
    loader_config = {
        "num_workers": 0,  # CPUでは0が最適な場合が多い
        "pin_memory": False,  # CPUではFalse
    }

    # 3. 画像サイズの縮小（scene_dataset.pyで設定）
    # transforms.Resize((112, 112)) に変更することを推奨

    print("最適化された設定:")
    print(f"- エポック数: {optimized_config['epochs']}")
    print(f"- バッチサイズ: {optimized_config['batch_size']}")
    print(f"- ワーカー数: {loader_config['num_workers']}")
    print(f"- 早期終了: {optimized_config['early_stopping_patience']}エポック")

    # データセットの準備
    train_dataset = SceneDataset(split="train")
    val_dataset = SceneDataset(split="val")

    print(f"\n学習データ: {len(train_dataset)}サンプル")
    print(f"検証データ: {len(val_dataset)}サンプル")

    # トレーナーの初期化
    trainer = SceneTrainer(
        output_dir="web_interface/models/game_scene",
        device="cpu",  # GPUがない場合
        num_workers=loader_config["num_workers"],
    )

    # 学習の実行
    print("\n最適化された設定で学習を開始します...")
    results = trainer.train(
        train_dataset=train_dataset, val_dataset=val_dataset, **optimized_config
    )

    print("\n学習が完了しました！")
    print(f"最終精度: {results.get('final_val_acc', 0):.2%}")


def create_frame_cache():
    """全フレームを事前にキャッシュ"""
    print("フレームキャッシュを作成中...")

    for split in ["train", "val", "test"]:
        dataset = SceneDataset(split=split)
        print(f"\n{split}データセット: {len(dataset)}フレーム")

        for i in range(len(dataset)):
            try:
                _ = dataset[i]
                if i % 100 == 0:
                    print(f"  キャッシュ済み: {i}/{len(dataset)}")
            except Exception as e:
                print(f"  エラー (index {i}): {e}")
                continue

    print("\nキャッシュ作成が完了しました！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="学習最適化スクリプト")
    parser.add_argument("--cache-only", action="store_true", help="キャッシュ作成のみ実行")
    parser.add_argument("--train", action="store_true", help="最適化された設定で学習を実行")

    args = parser.parse_args()

    if args.cache_only:
        create_frame_cache()
    elif args.train:
        optimize_training()
    else:
        print("使用方法:")
        print("  python optimize_training.py --cache-only  # キャッシュ作成")
        print("  python optimize_training.py --train       # 最適化学習")
