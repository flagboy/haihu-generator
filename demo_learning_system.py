"""
学習システムデモンストレーション

フェーズ2: 学習システムの機能をデモンストレーションする
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

try:
    from src.training.annotation_data import (
        AnnotationData,
        BoundingBox,
        FrameAnnotation,
        TileAnnotation,
        VideoAnnotation,
    )
    from src.training.dataset_manager import DatasetManager
    from src.training.learning.model_evaluator import ModelEvaluator
    from src.training.learning.training_manager import TrainingConfig, TrainingManager
    from src.utils.config import ConfigManager

    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("一部の依存関係が不足していますが、デモを続行します...")
    IMPORTS_AVAILABLE = False


def create_sample_dataset():
    """サンプルデータセットを作成"""
    print("サンプルデータセットを作成中...")

    if not IMPORTS_AVAILABLE:
        print("  依存関係が不足しているため、サンプルデータセット作成をスキップします")
        return None

    # サンプルアノテーションデータを作成
    annotation_data = AnnotationData()

    # サンプル動画アノテーション
    video_annotation = VideoAnnotation(
        video_id="sample_video_001",
        video_path="data/sample/video_001.mp4",
        video_name="サンプル動画1",
        duration=120.0,
        fps=30.0,
        width=1920,
        height=1080,
        frames=[],
    )

    # サンプルフレームアノテーション
    for i in range(10):  # 10フレームのサンプル
        frame_annotation = FrameAnnotation(
            frame_id=f"frame_{i:03d}",
            image_path=f"data/sample/frames/frame_{i:03d}.jpg",
            image_width=1920,
            image_height=1080,
            timestamp=i * 2.0,
            tiles=[],
            quality_score=0.9,
            is_valid=True,
            scene_type="game",
            game_phase="playing",
        )

        # サンプル牌アノテーション
        tile_types = [
            "1m",
            "2m",
            "3m",
            "1p",
            "2p",
            "3p",
            "1s",
            "2s",
            "3s",
            "東",
            "南",
            "西",
            "北",
            "白",
            "發",
            "中",
        ]
        for j in range(5):  # フレームあたり5個の牌
            tile_annotation = TileAnnotation(
                tile_id=tile_types[j % len(tile_types)],
                bbox=BoundingBox(x1=100 + j * 50, y1=200, x2=140 + j * 50, y2=260),
                confidence=0.95,
                area_type="hand",
                is_face_up=True,
                is_occluded=False,
                occlusion_ratio=0.0,
                annotator="demo_system",
            )
            frame_annotation.tiles.append(tile_annotation)

        video_annotation.frames.append(frame_annotation)

    annotation_data.video_annotations[video_annotation.video_id] = video_annotation

    return annotation_data


def demo_dataset_management():
    """データセット管理のデモ"""
    print("\n=== データセット管理デモ ===")

    if not IMPORTS_AVAILABLE:
        print("依存関係が不足しているため、実際のデモはスキップします")
        print("1. アノテーションデータを保存... (スキップ)")
        print("2. データセットバージョンを作成... (スキップ)")
        print("3. データセット統計情報: (スキップ)")
        return "dummy_version_id"

    config_manager = ConfigManager()
    dataset_manager = DatasetManager(config_manager)

    # サンプルデータセットを作成
    annotation_data = create_sample_dataset()

    # データセットを保存
    print("1. アノテーションデータを保存...")
    success = dataset_manager.save_annotation_data(annotation_data)
    print(f"   保存結果: {'成功' if success else '失敗'}")

    # データセットバージョンを作成
    print("2. データセットバージョンを作成...")
    version_id = dataset_manager.create_dataset_version(
        annotation_data=annotation_data,
        version="v1.0.0",
        description="学習システムデモ用サンプルデータセット",
    )
    print(f"   バージョンID: {version_id}")

    # 統計情報を表示
    print("3. データセット統計情報:")
    stats = dataset_manager.get_dataset_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    return version_id


def demo_training_configuration():
    """学習設定のデモ"""
    print("\n=== 学習設定デモ ===")

    if not IMPORTS_AVAILABLE:
        print("依存関係が不足しているため、設定例のみ表示します")
        print("1. 検出モデル設定:")
        print("   モデルタイプ: detection")
        print("   エポック数: 50")
        print("   バッチサイズ: 16")
        print("   学習率: 0.001")
        print("2. 分類モデル設定:")
        print("   モデルタイプ: classification")
        print("   エポック数: 100")
        print("   バッチサイズ: 32")
        print("   学習率: 0.0001")
        return None, None

    # 検出モデル用設定
    detection_config = TrainingConfig(
        model_type="detection",
        model_name="tile_detector_v1",
        dataset_version_id="dummy_version_id",  # 実際のバージョンIDに置き換え
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        validation_split=0.2,
        test_split=0.1,
        early_stopping_patience=10,
        use_data_augmentation=True,
        transfer_learning=False,
        gpu_enabled=True,
    )

    # 分類モデル用設定
    classification_config = TrainingConfig(
        model_type="classification",
        model_name="tile_classifier_v1",
        dataset_version_id="dummy_version_id",
        epochs=100,
        batch_size=32,
        learning_rate=0.0001,
        validation_split=0.2,
        test_split=0.1,
        early_stopping_patience=15,
        use_data_augmentation=True,
        transfer_learning=False,
        gpu_enabled=True,
    )

    print("1. 検出モデル設定:")
    print(f"   モデルタイプ: {detection_config.model_type}")
    print(f"   エポック数: {detection_config.epochs}")
    print(f"   バッチサイズ: {detection_config.batch_size}")
    print(f"   学習率: {detection_config.learning_rate}")

    print("2. 分類モデル設定:")
    print(f"   モデルタイプ: {classification_config.model_type}")
    print(f"   エポック数: {classification_config.epochs}")
    print(f"   バッチサイズ: {classification_config.batch_size}")
    print(f"   学習率: {classification_config.learning_rate}")

    return detection_config, classification_config


def demo_training_manager():
    """学習管理のデモ"""
    print("\n=== 学習管理デモ ===")

    if not IMPORTS_AVAILABLE:
        print("依存関係が不足しているため、概念的なデモのみ表示します")
        print("1. 学習セッション管理:")
        print("   - 検出モデル学習をスケジュール... (概念的)")
        print("     （デモモードのため実際の学習はスキップ）")
        print("2. セッション一覧:")
        print("   セッションがありません")
        return

    config_manager = ConfigManager()
    training_manager = TrainingManager(config_manager)

    # データセットバージョンを作成
    version_id = demo_dataset_management()

    # 学習設定を作成
    detection_config, classification_config = demo_training_configuration()
    detection_config.dataset_version_id = version_id
    classification_config.dataset_version_id = version_id

    print("1. 学習セッション管理:")

    # 学習をスケジュール（実際には実行しない）
    print("   - 検出モデル学習をスケジュール...")
    try:
        # session_id = training_manager.start_training(detection_config)
        # print(f"     セッションID: {session_id}")
        print("     （デモモードのため実際の学習はスキップ）")
    except Exception as e:
        print(f"     エラー: {e}")

    # セッション一覧を表示
    print("2. セッション一覧:")
    sessions = training_manager.list_sessions()
    if sessions:
        for session in sessions[:3]:  # 最新3件のみ表示
            print(f"   - {session['session_id']}: {session['status']}")
    else:
        print("   セッションがありません")


def demo_hyperparameter_optimization():
    """ハイパーパラメータ最適化のデモ"""
    print("\n=== ハイパーパラメータ最適化デモ ===")

    if not IMPORTS_AVAILABLE:
        print("依存関係が不足しているため、概念的なデモのみ表示します")
        print("1. デフォルトハイパーパラメータ:")
        print("   learning_rate: float (範囲: 1e-05 - 0.1)")
        print("   batch_size: choice (選択肢: [8, 16, 32, 64, 128])")
        print("   epochs: int (範囲: 10 - 200)")
        print("   optimizer_type: choice (選択肢: ['adam', 'sgd', 'adamw'])")
        print("\n2. ランダムパラメータサンプリング:")
        print("   サンプル 1: learning_rate=0.001, batch_size=32, epochs=100")
        print("   サンプル 2: learning_rate=0.0001, batch_size=64, epochs=150")
        print("   サンプル 3: learning_rate=0.01, batch_size=16, epochs=80")
        print("\n3. 最適化設定例:")
        print("   手法: random")
        print("   試行数: 10")
        print("   目標メトリクス: val_accuracy")
        return

    config_manager = ConfigManager()
    training_manager = TrainingManager(config_manager)
    scheduler = training_manager.learning_scheduler

    print("1. デフォルトハイパーパラメータ:")
    for name, param in scheduler.hyperparameters.items():
        print(f"   {name}: {param.param_type}")
        if param.param_type == "float":
            print(f"     範囲: {param.min_value} - {param.max_value}")
        elif param.param_type == "choice":
            print(f"     選択肢: {param.choices}")

    print("\n2. ランダムパラメータサンプリング:")
    for i in range(3):
        params = scheduler._sample_random_parameters()
        print(f"   サンプル {i + 1}:")
        for key, value in params.items():
            print(f"     {key}: {value}")

    print("\n3. 最適化設定例:")
    optimization_config = {"method": "random", "n_trials": 10, "target_metric": "val_accuracy"}
    print(f"   手法: {optimization_config['method']}")
    print(f"   試行数: {optimization_config['n_trials']}")
    print(f"   目標メトリクス: {optimization_config['target_metric']}")


def demo_model_evaluation():
    """モデル評価のデモ"""
    print("\n=== モデル評価デモ ===")

    if not IMPORTS_AVAILABLE:
        print("依存関係が不足しているため、概念的なデモのみ表示します")
        print("1. 評価機能:")
        print("   - 精度メトリクス計算")
        print("   - 混同行列生成")
        print("   - 学習曲線可視化")
        print("   - 検出結果可視化")
        print("   - 総合評価レポート生成")
        print("\n2. サンプル学習履歴:")
        print("   エポック 0: 訓練損失=2.5, 検証損失=2.3, 訓練精度=0.3, 検証精度=0.35")
        print("   エポック 10: 訓練損失=1.8, 検証損失=1.9, 訓練精度=0.6, 検証精度=0.58")
        print("   エポック 20: 訓練損失=1.2, 検証損失=1.4, 訓練精度=0.8, 検証精度=0.75")
        print("   学習曲線作成をスキップ（依存関係不足）")
        print("\n3. 評価メトリクス例:")
        print("   accuracy: 0.850")
        print("   precision_macro: 0.830")
        print("   recall_macro: 0.820")
        print("   f1_macro: 0.820")
        print("   loss: 1.000")
        return

    config_manager = ConfigManager()
    evaluator = ModelEvaluator(config_manager)

    print("1. 評価機能:")
    print("   - 精度メトリクス計算")
    print("   - 混同行列生成")
    print("   - 学習曲線可視化")
    print("   - 検出結果可視化")
    print("   - 総合評価レポート生成")

    print("\n2. サンプル学習履歴:")
    sample_history = [
        {
            "epoch": 0,
            "train_loss": 2.5,
            "val_loss": 2.3,
            "train_accuracy": 0.3,
            "val_accuracy": 0.35,
        },
        {
            "epoch": 10,
            "train_loss": 1.8,
            "val_loss": 1.9,
            "train_accuracy": 0.6,
            "val_accuracy": 0.58,
        },
        {
            "epoch": 20,
            "train_loss": 1.2,
            "val_loss": 1.4,
            "train_accuracy": 0.8,
            "val_accuracy": 0.75,
        },
        {
            "epoch": 30,
            "train_loss": 0.8,
            "val_loss": 1.1,
            "train_accuracy": 0.9,
            "val_accuracy": 0.82,
        },
        {
            "epoch": 40,
            "train_loss": 0.5,
            "val_loss": 1.0,
            "train_accuracy": 0.95,
            "val_accuracy": 0.85,
        },
    ]

    # 学習曲線を作成（実際のモデルがないため、履歴のみ）
    try:
        curves_path = evaluator.create_learning_curves(sample_history)
        if curves_path:
            print(f"   学習曲線保存: {curves_path}")
        else:
            print("   学習曲線作成をスキップ（依存関係不足）")
    except Exception as e:
        print(f"   学習曲線作成エラー: {e}")

    print("\n3. 評価メトリクス例:")
    sample_metrics = {
        "accuracy": 0.85,
        "precision_macro": 0.83,
        "recall_macro": 0.82,
        "f1_macro": 0.82,
        "loss": 1.0,
    }

    for metric, value in sample_metrics.items():
        print(f"   {metric}: {value:.3f}")


def demo_continuous_learning():
    """継続学習のデモ"""
    print("\n=== 継続学習デモ ===")

    print("1. 継続学習の流れ:")
    print("   - ベースモデルの選択")
    print("   - 新しい教師データの追加")
    print("   - 転移学習による継続訓練")
    print("   - 性能比較と最適モデル選択")

    print("\n2. モデルバージョン管理:")
    versions = [
        {"version": "v1.0.0", "accuracy": 0.75, "date": "2024-01-01"},
        {"version": "v1.1.0", "accuracy": 0.82, "date": "2024-01-15"},
        {"version": "v1.2.0", "accuracy": 0.85, "date": "2024-02-01"},
        {"version": "v2.0.0", "accuracy": 0.88, "date": "2024-02-15"},
    ]

    for version in versions:
        print(f"   {version['version']}: 精度={version['accuracy']:.3f}, 日付={version['date']}")

    print("\n3. 性能向上の追跡:")
    print("   - 初期モデル (v1.0.0): 75.0%")
    print("   - 最新モデル (v2.0.0): 88.0%")
    print("   - 改善率: +13.0ポイント")


def demo_automated_features():
    """自動化機能のデモ"""
    print("\n=== 自動化機能デモ ===")

    print("1. 自動データ分割:")
    print("   - 訓練データ: 70%")
    print("   - 検証データ: 20%")
    print("   - テストデータ: 10%")
    print("   - 層化サンプリング対応")

    print("\n2. 自動ハイパーパラメータ調整:")
    print("   - ランダムサーチ")
    print("   - グリッドサーチ")
    print("   - ベイジアン最適化（予定）")

    print("\n3. 早期停止機能:")
    print("   - 検証損失の監視")
    print("   - 改善なしの連続エポック数で判定")
    print("   - 最適モデルの自動保存")

    print("\n4. 学習スケジューリング:")
    print("   - 時間指定実行")
    print("   - 依存関係管理")
    print("   - 優先度制御")
    print("   - リトライ機能")


def main():
    """メイン関数"""
    print("=" * 60)
    print("麻雀牌譜作成システム - フェーズ2: 学習システムデモ")
    print("=" * 60)

    if not IMPORTS_AVAILABLE:
        print("\n注意: 一部の依存関係が不足しているため、概念的なデモのみ実行します。")
        print("完全なデモを実行するには、以下のコマンドで依存関係をインストールしてください:")
        print("uv add torch torchvision scikit-learn matplotlib seaborn numpy pandas pillow")
        print("uv add opencv-python  # (オプション)")

    try:
        # 各機能のデモを実行
        demo_dataset_management()
        demo_training_configuration()
        demo_training_manager()
        demo_hyperparameter_optimization()
        demo_model_evaluation()
        demo_continuous_learning()
        demo_automated_features()

        print("\n" + "=" * 60)
        print("学習システムデモ完了")
        print("=" * 60)

        print("\n実装された主要機能:")
        print("✓ TrainingManager - 学習プロセス全体の管理")
        print("✓ ModelTrainer - 検出・分類モデルの訓練")
        print("✓ LearningScheduler - 学習スケジュールと最適化")
        print("✓ ModelEvaluator - モデル性能評価と可視化")
        print("✓ 継続学習機能")
        print("✓ 評価・可視化システム")
        print("✓ 自動化機能")

        print("\n次のステップ:")
        print("1. 実際の教師データでの学習実行")
        print("2. GPU環境での性能最適化")
        print("3. より高度な最適化アルゴリズムの実装")
        print("4. 分散学習対応")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
