#!/usr/bin/env python3
"""
対局画面分類モデルの学習スクリプト
"""

import argparse
from pathlib import Path

from src.training.game_scene.learning.scene_dataset import SceneDataset
from src.training.game_scene.learning.scene_trainer import SceneTrainer
from src.utils.logger import get_logger, setup_logger

# ロガー設定
setup_logger()
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
        help="モデル出力ディレクトリ",
    )
    parser.add_argument("--epochs", type=int, default=20, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学習率")
    parser.add_argument("--validation-split", type=float, default=0.2, help="検証データの割合")
    parser.add_argument("--device", type=str, default="cuda", help="使用デバイス (cuda/cpu)")

    args = parser.parse_args()

    # データベースの存在確認
    if not Path(args.db_path).exists():
        logger.error(f"データベースが見つかりません: {args.db_path}")
        return

    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("対局画面分類モデルの学習を開始します")
    logger.info("=" * 50)

    # データセットの作成
    logger.info("データセットを準備中...")
    dataset = SceneDataset(args.db_path)

    # データセットの統計情報を表示
    stats = dataset.get_statistics()
    logger.info(f"総サンプル数: {stats['total_samples']}")
    logger.info(f"対局画面: {stats['game_scenes']}")
    logger.info(f"非対局画面: {stats['non_game_scenes']}")
    logger.info(f"動画数: {stats['num_videos']}")

    if stats["total_samples"] < 100:
        logger.warning("サンプル数が少ないため、学習結果が不安定になる可能性があります")

    # トレーナーの作成
    trainer = SceneTrainer(
        output_dir=str(output_dir),
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    # 学習の実行
    logger.info("学習を開始します...")
    results = trainer.train(
        dataset,
        validation_split=args.validation_split,
        save_best=True,
        early_stopping_patience=5,
    )

    # 結果の表示
    logger.info("=" * 50)
    logger.info("学習が完了しました")
    logger.info("=" * 50)
    logger.info(f"最良検証精度: {results['best_val_accuracy']:.2%}")
    logger.info(f"最終学習精度: {results['final_train_accuracy']:.2%}")
    logger.info(f"学習エポック数: {results['epochs_trained']}")
    logger.info(f"モデル保存先: {results['model_path']}")

    # 混同行列の表示（オプション）
    if "confusion_matrix" in results:
        logger.info("\n混同行列:")
        logger.info("  予測\\実際  対局  非対局")
        logger.info(
            f"  対局     {results['confusion_matrix'][1][1]:5d} {results['confusion_matrix'][1][0]:5d}"
        )
        logger.info(
            f"  非対局   {results['confusion_matrix'][0][1]:5d} {results['confusion_matrix'][0][0]:5d}"
        )


if __name__ == "__main__":
    main()
