"""
継続学習システムの使用例

継続学習の基本的な使い方を示すサンプルコード
"""

import time

from src.training.dataset_manager import DatasetManager
from src.training.learning import (
    ContinuousLearningConfig,
    ContinuousLearningController,
    TrainingManager,
)
from src.training.learning.components import CheckpointManager, DataHistoryManager
from src.utils.config import ConfigManager


def main():
    """継続学習の例"""

    print("=== 継続学習システムの例 ===\n")

    # 設定とマネージャーの初期化
    config_manager = ConfigManager()
    training_manager = TrainingManager(config_manager)
    dataset_manager = DatasetManager(config_manager)
    checkpoint_manager = CheckpointManager()
    data_history_manager = DataHistoryManager()

    # 継続学習コントローラーを初期化
    cl_controller = ContinuousLearningController(
        training_manager=training_manager,
        dataset_manager=dataset_manager,
        checkpoint_manager=checkpoint_manager,
    )

    # 1. Fine-tuning戦略での継続学習
    print("1. Fine-tuning戦略での継続学習")
    print("-" * 40)

    fine_tuning_config = ContinuousLearningConfig(
        base_model_path=None,  # 新規学習から開始
        incremental_data_threshold=50,
        strategy="fine_tuning",
        fine_tuning_lr_factor=0.1,
        min_performance_threshold=0.8,
        auto_train_enabled=True,
    )

    # セッションを開始
    session_id = cl_controller.start_continuous_learning(
        model_type="classification",
        config=fine_tuning_config,
        initial_dataset_version=None,  # 初期データなしで開始
    )

    print(f"セッションID: {session_id}")
    print(f"戦略: {fine_tuning_config.strategy}")
    print(f"データ閾値: {fine_tuning_config.incremental_data_threshold}件\n")

    # 2. リハーサル戦略での継続学習
    print("2. リハーサル戦略での継続学習")
    print("-" * 40)

    rehearsal_config = ContinuousLearningConfig(
        strategy="rehearsal",
        rehearsal_size=500,
        rehearsal_ratio=0.3,
        incremental_data_threshold=100,
    )

    rehearsal_session_id = cl_controller.start_continuous_learning(
        model_type="detection", config=rehearsal_config
    )

    print(f"セッションID: {rehearsal_session_id}")
    print(f"リハーサルバッファサイズ: {rehearsal_config.rehearsal_size}")
    print(f"リハーサル比率: {rehearsal_config.rehearsal_ratio}\n")

    # 3. 知識蒸留を使用した継続学習
    print("3. 知識蒸留を使用した継続学習")
    print("-" * 40)

    distillation_config = ContinuousLearningConfig(
        strategy="fine_tuning",
        use_knowledge_distillation=True,
        distillation_temperature=5.0,
        distillation_alpha=0.7,
        temperature_schedule="cosine",
        alpha_schedule="linear",
    )

    distillation_session_id = cl_controller.start_continuous_learning(
        model_type="classification", config=distillation_config
    )

    print(f"セッションID: {distillation_session_id}")
    print(f"蒸留温度: {distillation_config.distillation_temperature}")
    print(f"蒸留アルファ: {distillation_config.distillation_alpha}\n")

    # 4. データ履歴の管理例
    print("4. データ履歴の管理")
    print("-" * 40)

    # サンプルデータを追加
    from datetime import datetime

    from src.training.learning.components.data_history_manager import DataSample

    sample_data = [
        DataSample(
            sample_id=f"sample_{i}",
            video_id=f"video_{i // 10}",
            frame_id=f"frame_{i}",
            tile_count=34,
            quality_score=0.8 + (i % 20) * 0.01,
            timestamp=datetime.now(),
            labels=[f"tile_{j}" for j in range(5)],
            features={"brightness": 0.5, "contrast": 0.8},
        )
        for i in range(100)
    ]

    data_history_manager.add_samples(sample_data[:50])
    print("50件のサンプルデータを追加しました")

    # リハーサル用サンプルを選択
    selected_samples = data_history_manager.select_rehearsal_samples(
        available_samples=[s.sample_id for s in sample_data[:50]],
        num_samples=10,
        selection_strategy="importance_sampling",
    )

    print(f"リハーサル用に{len(selected_samples)}件のサンプルを選択しました")

    # 重要度スコアを計算
    importance_scores = data_history_manager.get_sample_importance_scores(
        sample_ids=selected_samples[:5], method="performance"
    )

    print("\n重要度スコア（上位5件）:")
    for sample_id, score in list(importance_scores.items())[:5]:
        print(f"  {sample_id}: {score:.3f}")

    # 5. セッション情報の確認
    print("\n5. セッション情報")
    print("-" * 40)

    sessions = cl_controller.list_sessions()
    print(f"アクティブなセッション数: {len(sessions)}")

    for session in sessions[:3]:
        print(f"\nセッションID: {session['session_id']}")
        print(f"  戦略: {session['config']['strategy']}")
        print(f"  状態: {session['status']}")
        print(f"  更新回数: {session['incremental_updates']}")
        print(f"  総サンプル数: {session['total_samples_seen']}")

    # 6. 性能トレンドの確認
    print("\n6. 性能トレンド")
    print("-" * 40)

    # ダミーの性能データを記録
    data_history_manager.record_usage(
        session_id=session_id,
        dataset_version="v1.0",
        sample_ids=[s.sample_id for s in sample_data[:20]],
        performance_metrics={"accuracy": 0.85, "loss": 0.23},
        strategy="fine_tuning",
    )

    time.sleep(0.1)  # タイムスタンプを変えるため

    data_history_manager.record_usage(
        session_id=session_id,
        dataset_version="v1.1",
        sample_ids=[s.sample_id for s in sample_data[20:40]],
        performance_metrics={"accuracy": 0.88, "loss": 0.19},
        strategy="fine_tuning",
    )

    # トレンドを取得
    trend = data_history_manager.get_performance_trend(session_id=session_id, last_n_entries=5)

    print("性能トレンド:")
    for entry in trend:
        metrics = entry["metrics"]
        print(
            f"  {entry['timestamp']}: "
            f"accuracy={metrics.get('accuracy', 0):.3f}, "
            f"loss={metrics.get('loss', 0):.3f}"
        )

    print("\n=== 継続学習システムの例 完了 ===")


if __name__ == "__main__":
    main()
