# 教師データ作成システム

麻雀牌検出・分類モデルの教師データ作成を支援するシステムです。

## 概要

このシステムは、麻雀対局動画から効率的に教師データを作成するための基盤を提供します。フェーズ1では以下の機能を実装しています：

### 実装済み機能（フェーズ1）

1. **データ管理システム**
   - `DatasetManager`: 教師データの保存、読み込み、バージョン管理
   - `AnnotationData`: ラベリングデータの構造定義
   - SQLiteデータベースによるデータ管理

2. **フレーム抽出システム**
   - `FrameExtractor`: 動画からの効率的なフレーム抽出
   - `FrameQualityAnalyzer`: フレーム品質評価
   - 既存`VideoProcessor`の拡張

3. **半自動ラベリング基盤**
   - `SemiAutoLabeler`: 既存モデルを使った初期予測
   - 予測結果の可視化機能
   - ユーザー修正インターフェースの基盤

## ディレクトリ構造

```
src/training/
├── __init__.py                 # パッケージ初期化
├── annotation_data.py          # アノテーションデータ構造
├── dataset_manager.py          # データセット管理
├── frame_extractor.py          # フレーム抽出
└── semi_auto_labeler.py        # 半自動ラベリング

data/training/                  # 教師データ保存ディレクトリ
├── dataset.db                  # SQLiteデータベース
├── images/                     # 抽出画像
├── annotations/                # アノテーションファイル
├── versions/                   # データセットバージョン
└── exports/                    # エクスポート済みデータ
```

## 使用方法

### 1. 基本的な使用例

```python
from src.training import DatasetManager, FrameExtractor, SemiAutoLabeler
from src.utils.config import ConfigManager

# 設定管理を初期化
config_manager = ConfigManager()

# データセット管理を初期化
dataset_manager = DatasetManager(config_manager)

# フレーム抽出を初期化
frame_extractor = FrameExtractor(config_manager)

# 半自動ラベリングを初期化
semi_auto_labeler = SemiAutoLabeler(config_manager)
```

### 2. 動画からフレーム抽出

```python
# 動画からフレームを抽出
video_path = "path/to/mahjong_video.mp4"
frame_annotations = frame_extractor.extract_training_frames(video_path)

print(f"抽出されたフレーム数: {len(frame_annotations)}")
```

### 3. アノテーションデータの管理

```python
from src.training.annotation_data import AnnotationData

# アノテーションデータを作成
annotation_data = AnnotationData()

# 動画アノテーションを作成
video_info = {
    "duration": 120.0,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
}
video_id = annotation_data.create_video_annotation(video_path, video_info)

# フレームアノテーションを追加
for frame_annotation in frame_annotations:
    annotation_data.add_frame_annotation(video_id, frame_annotation)

# JSONファイルに保存
annotation_data.save_to_json("annotations.json")
```

### 4. データベースへの保存

```python
# データベースに保存
dataset_manager.save_annotation_data(annotation_data)

# データセットバージョンを作成
version_id = dataset_manager.create_dataset_version(
    annotation_data,
    "v1.0.0",
    "初回バージョン"
)

# YOLO形式でエクスポート
dataset_manager.export_dataset(version_id, "yolo", "output/yolo_dataset")
```

### 5. 半自動ラベリング

```python
# フレームの自動予測
prediction_results = []
for frame_annotation in frame_annotations:
    try:
        prediction_result = semi_auto_labeler.predict_frame_annotations(frame_annotation)
        prediction_results.append(prediction_result)
    except Exception as e:
        print(f"予測エラー: {e}")

# 予測結果を保存
semi_auto_labeler.save_predictions(prediction_results, "predictions_v1")

# 可視化を生成
semi_auto_labeler.generate_visualizations(prediction_results, "visualizations_v1")
```

## 設定

`config.yaml`の`training`セクションで設定を調整できます：

```yaml
training:
  # データベース設定
  database_path: "data/training/dataset.db"
  dataset_root: "data/training"

  # フレーム抽出設定
  frame_extraction:
    min_quality_score: 0.6
    max_frames_per_video: 1000
    frame_interval_seconds: 2.0
    diversity_threshold: 0.3
    output_dir: "data/training/extracted_frames"

  # 半自動ラベリング設定
  semi_auto_labeling:
    confidence_threshold: 0.5
    auto_area_classification: true
    enable_occlusion_detection: true
    labeling_output_dir: "data/training/labeling"
```

## データ構造

### BoundingBox

```python
@dataclass
class BoundingBox:
    x1: int  # 左上X座標
    y1: int  # 左上Y座標
    x2: int  # 右下X座標
    y2: int  # 右下Y座標
```

### TileAnnotation

```python
@dataclass
class TileAnnotation:
    tile_id: str           # 牌の種類 ("1m", "2p", "3s", "東" など)
    bbox: BoundingBox      # バウンディングボックス
    confidence: float      # 信頼度 (0.0-1.0)
    area_type: str         # エリアタイプ ("hand", "discard", "call")
    is_face_up: bool       # 表向きかどうか
    is_occluded: bool      # 遮蔽されているかどうか
    occlusion_ratio: float # 遮蔽率 (0.0-1.0)
    annotator: str         # アノテーター
    notes: str             # 備考
```

### FrameAnnotation

```python
@dataclass
class FrameAnnotation:
    frame_id: str              # フレームID
    image_path: str            # 画像パス
    image_width: int           # 画像幅
    image_height: int          # 画像高さ
    timestamp: float           # 動画内の時刻（秒）
    tiles: List[TileAnnotation] # 牌アノテーション
    quality_score: float       # 品質スコア (0.0-1.0)
    is_valid: bool             # 有効なフレームかどうか
    scene_type: str            # シーンタイプ
    game_phase: str            # ゲームフェーズ
    annotated_at: datetime     # アノテーション日時
    annotator: str             # アノテーター
    notes: str                 # 備考
```

## デモンストレーション

システムの動作確認には以下のコマンドを実行してください：

```bash
python demo_training_system.py
```

このデモでは以下の処理を実行します：

1. サンプル動画の作成
2. アノテーションデータの作成・保存
3. データセット管理機能の確認
4. フレーム抽出機能の確認
5. 半自動ラベリング機能の確認

## テスト

単体テストを実行するには：

```bash
pytest tests/test_training_system.py -v
```

## エクスポート形式

### YOLO形式

```
dataset/
├── images/
│   ├── frame_001.jpg
│   └── frame_002.jpg
├── labels/
│   ├── frame_001.txt
│   └── frame_002.txt
└── classes.txt
```

ラベルファイル形式：
```
class_id center_x center_y width height
0 0.5 0.3 0.1 0.15
1 0.7 0.4 0.12 0.18
```

### 統計情報

システムは以下の統計情報を提供します：

- 総動画数・フレーム数・牌数
- 牌種類別分布
- エリア別分布（手牌・捨て牌・鳴き牌）
- 品質スコア分布
- アノテーション進捗

## 今後の拡張予定（フェーズ2以降）

1. **Webインターフェース**
   - ブラウザベースのアノテーションツール
   - リアルタイム予測・修正機能

2. **高度なAI機能**
   - より精度の高い検出・分類モデル
   - アクティブラーニング
   - 自動品質評価

3. **協調作業機能**
   - 複数人でのアノテーション作業
   - 品質管理・レビュー機能

## トラブルシューティング

### よくある問題

1. **モデルファイルが見つからない**
   ```
   WARNING: Model file not found: models/tile_detector.pt
   ```
   → フェーズ1では学習済みモデルは不要です。警告は無視してください。

2. **データベースエラー**
   ```
   ERROR: Failed to save annotation data
   ```
   → `data/training`ディレクトリの書き込み権限を確認してください。

3. **メモリ不足**
   ```
   ERROR: Out of memory during frame extraction
   ```
   → `config.yaml`の`max_frames_per_video`を小さくしてください。

### ログ確認

詳細なログは以下で確認できます：
```bash
tail -f logs/mahjong_system.log
```

## 貢献

バグ報告や機能要望は、プロジェクトのIssueトラッカーまでお願いします。

## ライセンス

このプロジェクトは研究用途での使用を想定しています。
