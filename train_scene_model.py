#!/usr/bin/env python3
"""
対局画面分類モデルの学習スクリプト

使用方法:
    python train_scene_model.py
    python train_scene_model.py --epochs 100 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from src.training.game_scene.learning.scene_dataset import SceneDataset
from src.training.game_scene.learning.scene_trainer import SceneTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="対局画面分類モデルの学習")
    parser.add_argument(
        "--db-path",
        type=str,
        default="web_interface/data/training/game_scene_labels.db",
        help="ラベルデータベースのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/game_scene",
        help="モデル保存ディレクトリ",
    )
    parser.add_argument("--epochs", type=int, default=50, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学習率")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用デバイス (cuda/cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="データローダーのワーカー数",
    )

    args = parser.parse_args()

    # データベースの存在確認
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"データベースが見つかりません: {db_path}")
        sys.exit(1)

    logger.info("=== 対局画面分類モデルの学習を開始 ===")
    logger.info(f"データベース: {args.db_path}")
    logger.info(f"出力ディレクトリ: {args.output_dir}")
    logger.info(f"エポック数: {args.epochs}")
    logger.info(f"バッチサイズ: {args.batch_size}")
    logger.info(f"学習率: {args.learning_rate}")

    # データセットの作成
    logger.info("\nデータセットを準備中...")
    try:
        train_dataset = SceneDataset(db_path=str(db_path), split="train")
        val_dataset = SceneDataset(db_path=str(db_path), split="val")
        test_dataset = SceneDataset(db_path=str(db_path), split="test")

        logger.info(f"学習データ: {len(train_dataset)}サンプル")
        logger.info(f"検証データ: {len(val_dataset)}サンプル")
        logger.info(f"テストデータ: {len(test_dataset)}サンプル")

        # データセットの統計情報
        train_stats = train_dataset.get_statistics()
        logger.info(f"対局画面: {train_stats['game_scenes']}枚")
        logger.info(f"非対局画面: {train_stats['non_game_scenes']}枚")
        logger.info(f"動画数: {train_stats['videos']}本")

    except Exception as e:
        logger.error(f"データセット作成エラー: {e}")
        sys.exit(1)

    # データセットが空でないことを確認
    if len(train_dataset) == 0:
        logger.error("学習データが空です。ラベル付けを行ってください。")
        sys.exit(1)

    # トレーナーの初期化
    logger.info("\nトレーナーを初期化中...")
    trainer = SceneTrainer(
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
    )

    # 学習の実行
    logger.info("\n学習を開始します...")
    try:
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        # 結果の表示
        logger.info("\n=== 学習完了 ===")
        logger.info(f"学習エポック数: {results['epochs_trained']}")
        logger.info(f"最良検証精度: {results['best_val_acc']:.2%}")
        logger.info(f"最終学習精度: {results['final_train_acc']:.2%}")
        logger.info(f"最良モデル: {results['paths']['best_model']}")
        logger.info(f"最終モデル: {results['paths']['final_model']}")

        # テストデータでの評価
        if len(test_dataset) > 0:
            logger.info("\nテストデータで評価中...")
            test_results = trainer.evaluate(
                test_dataset=test_dataset,
                model_path=results["paths"]["best_model"],
                batch_size=args.batch_size,
            )
            logger.info(f"テスト精度: {test_results['accuracy']:.2%}")
            logger.info(f"テストF1スコア: {test_results['f1_score']:.2%}")

    except KeyboardInterrupt:
        logger.info("\n学習が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"学習中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
