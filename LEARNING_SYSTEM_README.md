# 学習システム (フェーズ2) - 使用ガイド

## 概要

フェーズ2では、フェーズ1で作成された教師データ作成システムの基盤を活用して、包括的な学習システムを実装しました。このシステムは、検出・分類モデルの訓練、継続学習、性能評価、自動化機能を提供します。

## 主要コンポーネント

### 1. TrainingManager
学習プロセス全体を管理するメインクラス

**主な機能:**
- 学習セッションの管理
- データセットの準備と分割
- モデルの準備と転移学習
- 継続学習の実行
- モデル比較とバージョン管理

### 2. ModelTrainer
実際のモデル訓練を実行するクラス

**主な機能:**
- PyTorchベースのモデル訓練
- データ拡張（Data Augmentation）
- GPU対応とメモリ最適化
- チェックポイント管理
- 早期停止機能

### 3. LearningScheduler
学習スケジュールとハイパーパラメータ最適化を管理

**主な機能:**
- ハイパーパラメータ自動調整
- 学習タスクのスケジューリング
- ランダムサーチ・グリッドサーチ
- 適応的学習スケジュール

### 4. ModelEvaluator
モデル性能評価と可視化を実行

**主な機能:**
- 精度メトリクス計算（mAP、Precision、Recall等）
- 学習曲線の可視化
- 混同行列の生成
- 検出結果の可視化
- 総合評価レポート生成

## インストールと設定

### 環境管理

このプロジェクトは`uv`を使用して依存関係を管理しています。

```bash
# uvがインストールされていない場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate     # Windows
```

### 必要な依存関係

```bash
# uvを使用した追加パッケージのインストール
uv add torch torchvision
uv add scikit-learn
uv add matplotlib seaborn
uv add numpy pandas
uv add Pillow opencv-python
uv add pyyaml

# または、requirements.txtから一括インストール
uv pip install -r requirements.txt
```

### 設定ファイル

`config.yaml`に学習関連の設定を追加：

```yaml
training:
  training_root: "data/training"
  dataset_root: "data/training/dataset"
  database_path: "data/training/dataset.db"
  evaluation_root: "data/training/evaluation"
  scheduler_root: "data/training/scheduler"
  num_tile_classes: 34

  # デフォルト学習設定
  default_epochs: 100
  default_batch_size: 32
  default_learning_rate: 0.001
  default_validation_split: 0.2
  default_test_split: 0.1
```

## 基本的な使用方法

### 1. 学習システムの初期化

```python
from src.utils.config import ConfigManager
from src.training.learning.training_manager import TrainingManager, TrainingConfig

# 設定管理とトレーニングマネージャーを初期化
config_manager = ConfigManager()
training_manager = TrainingManager(config_manager)
```

### 2. 学習設定の作成

```python
# 分類モデルの学習設定
classification_config = TrainingConfig(
    model_type="classification",
    model_name="tile_classifier_v1",
    dataset_version_id="your_dataset_version_id",
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.2,
    test_split=0.1,
    early_stopping_patience=10,
    use_data_augmentation=True,
    transfer_learning=False,
    gpu_enabled=True
)

# 検出モデルの学習設定
detection_config = TrainingConfig(
    model_type="detection",
    model_name="tile_detector_v1",
    dataset_version_id="your_dataset_version_id",
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    use_data_augmentation=True
)
```

### 3. 学習の実行

```python
# 学習を開始
session_id = training_manager.start_training(classification_config)
print(f"学習セッション開始: {session_id}")

# 学習状況の確認
status = training_manager.get_session_status(session_id)
print(f"現在の状況: {status['status']}")
print(f"現在のエポック: {status.get('current_progress', {}).get('current_epoch', 0)}")
```

### 4. 継続学習

```python
# 既存モデルから継続学習
base_session_id = "previous_session_id"
new_config = TrainingConfig(
    model_type="classification",
    model_name="tile_classifier_v2",
    dataset_version_id="new_dataset_version_id",
    epochs=50,
    learning_rate=0.0001  # より小さい学習率
)

# 継続学習を実行
new_session_id = training_manager.continue_training(base_session_id, new_config)
```

### 5. モデル評価

```python
from src.training.learning.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(config_manager)

# モデルを評価
model_path = "path/to/your/model.pt"
test_data = your_test_annotation_data
metrics = evaluator.evaluate_model(model_path, test_data, "classification")

print("評価結果:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# 総合評価レポートを生成
report_path = evaluator.generate_evaluation_report(
    model_path, test_data, "classification", training_history
)
print(f"評価レポート: {report_path}")
```

### 6. ハイパーパラメータ最適化

```python
from src.training.learning.learning_scheduler import LearningScheduler

scheduler = LearningScheduler(config_manager)

# 最適化設定
optimization_config = {
    "method": "random",
    "n_trials": 20,
    "target_metric": "val_accuracy"
}

# ハイパーパラメータ最適化を開始
base_config = {
    "model_type": "classification",
    "dataset_version_id": "your_dataset_version_id"
}

trial_ids = scheduler.start_hyperparameter_optimization(
    base_config, optimization_config
)

# 最良パラメータを取得
best_params = scheduler.get_best_parameters(n_best=3)
print("最良パラメータ:")
for i, params in enumerate(best_params):
    print(f"  {i+1}位: スコア={params['score']:.4f}")
    print(f"       パラメータ: {params['parameters']}")
```

## 高度な機能

### 1. 学習スケジューリング

```python
from datetime import datetime, timedelta

# 将来の時刻に学習をスケジュール
scheduled_time = datetime.now() + timedelta(hours=2)
task_id = scheduler.schedule_training(
    config=classification_config.__dict__,
    scheduled_time=scheduled_time,
    priority=1
)

# 次に実行すべきタスクを取得
next_task = scheduler.get_next_task()
if next_task:
    print(f"次のタスク: {next_task.task_id}")
```

### 2. 適応的学習スケジュール

```python
# 段階的学習スケジュールを作成
base_config = {
    "model_type": "classification",
    "dataset_version_id": "your_dataset_version_id"
}

task_ids = scheduler.create_adaptive_schedule(
    base_config, performance_threshold=0.8
)
print(f"適応的スケジュール作成: {len(task_ids)}ステージ")
```

### 3. モデル比較

```python
# 複数のセッションを比較
session_ids = ["session_1", "session_2", "session_3"]
comparison = training_manager.compare_models(session_ids)

print("モデル比較結果:")
print(f"最良セッション: {comparison['best_session']}")

for metric, data in comparison['metrics_comparison'].items():
    best = data['best']
    print(f"{metric}: 最良={best['value']:.4f} (セッション: {best['session_id']})")
```

### 4. 可視化機能

```python
# 学習曲線の作成
training_history = [
    {"epoch": 0, "train_loss": 2.0, "val_loss": 1.8, "train_accuracy": 0.3, "val_accuracy": 0.35},
    {"epoch": 10, "train_loss": 1.2, "val_loss": 1.4, "train_accuracy": 0.7, "val_accuracy": 0.68},
    # ... more history
]

curves_path = evaluator.create_learning_curves(training_history)
print(f"学習曲線: {curves_path}")

# 混同行列の作成
confusion_matrix_path = evaluator.create_confusion_matrix(
    model_path, test_data, class_names=["1m", "2m", "3m", ...]
)
print(f"混同行列: {confusion_matrix_path}")

# 検出結果の可視化
detection_viz_path = evaluator.create_detection_visualization(
    model_path, test_data, num_samples=10
)
print(f"検出結果可視化: {detection_viz_path}")
```

## データ形式

### 学習データ形式

学習システムは、フェーズ1で作成された`AnnotationData`形式を使用します：

```python
from src.training.annotation_data import AnnotationData, VideoAnnotation, FrameAnnotation, TileAnnotation, BoundingBox

# アノテーションデータの構造
annotation_data = AnnotationData()
annotation_data.video_annotations = {
    "video_id": VideoAnnotation(
        video_id="video_id",
        video_path="path/to/video.mp4",
        frames=[
            FrameAnnotation(
                frame_id="frame_001",
                image_path="path/to/frame.jpg",
                tiles=[
                    TileAnnotation(
                        tile_id="1m",
                        bbox=BoundingBox(x1=100, y1=200, x2=140, y2=260),
                        confidence=0.95,
                        area_type="hand"
                    )
                ]
            )
        ]
    )
}
```

### YOLO形式エクスポート

```python
from src.training.dataset_manager import DatasetManager

dataset_manager = DatasetManager(config_manager)

# YOLO形式でエクスポート
success = dataset_manager.export_dataset(
    version_id="your_version_id",
    export_format="yolo",
    output_dir="data/yolo_dataset"
)
```

## パフォーマンス最適化

### GPU使用

```python
# GPU設定の確認
import torch
print(f"CUDA利用可能: {torch.cuda.is_available()}")
print(f"GPU数: {torch.cuda.device_count()}")

# 学習設定でGPUを有効化
config = TrainingConfig(
    # ... other settings
    gpu_enabled=True
)
```

### メモリ最適化

```python
# バッチサイズの調整
config = TrainingConfig(
    batch_size=16,  # GPUメモリに応じて調整
    num_workers=4   # データローダーのワーカー数
)
```

### 早期停止

```python
config = TrainingConfig(
    early_stopping_patience=10,  # 10エポック改善なしで停止
    save_best_only=True         # 最良モデルのみ保存
)
```

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - バッチサイズを小さくする
   - `num_workers`を減らす
   - 画像サイズを小さくする

2. **学習が収束しない**
   - 学習率を調整する
   - データ拡張を確認する
   - モデルアーキテクチャを見直す

3. **GPU使用時のエラー**
   - CUDA環境を確認する
   - PyTorchのバージョンを確認する
   - GPU メモリを確認する

### ログの確認

```python
# ログレベルの設定
import logging
logging.basicConfig(level=logging.INFO)

# 学習ログの確認
training_manager = TrainingManager(config_manager)
# ログは自動的に出力されます
```

## デモの実行

```bash
# 学習システムのデモを実行
python demo_learning_system.py

# テストの実行
python -m pytest tests/test_learning_system.py -v
```

## 今後の拡張予定

1. **分散学習対応**
   - 複数GPU対応
   - 複数ノード対応

2. **高度な最適化手法**
   - ベイジアン最適化の完全実装
   - 進化的アルゴリズム

3. **モデルアーキテクチャの拡張**
   - Transformer ベースモデル
   - より高度なYOLOバリアント

4. **リアルタイム学習**
   - オンライン学習
   - 増分学習

## サポート

問題や質問がある場合は、以下を確認してください：

1. このドキュメント
2. `demo_learning_system.py`のサンプルコード
3. `tests/test_learning_system.py`のテストケース
4. ログファイル（`logs/`ディレクトリ）

---

**注意**: このシステムは研究・開発目的で作成されています。本番環境での使用前に十分なテストを行ってください。
