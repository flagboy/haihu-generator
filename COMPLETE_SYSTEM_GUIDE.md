# 麻雀牌譜作成システム - 完全ガイド

## 目次

1. [システム概要](#システム概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [インストール・セットアップ](#インストール・セットアップ)
4. [使用方法](#使用方法)
5. [API仕様](#api仕様)
6. [設定ガイド](#設定ガイド)
7. [トラブルシューティング](#トラブルシューティング)
8. [パフォーマンス最適化](#パフォーマンス最適化)
9. [品質保証](#品質保証)
10. [開発・拡張](#開発・拡張)

---

## システム概要

### 概要

麻雀牌譜作成システムは、麻雀の対局動画（MP4等）を入力として受け取り、動画から麻雀牌の動きや配置を認識して、天鳳JSON形式で牌譜を自動生成するAIシステムです。

### 主要機能

#### フェーズ1: 基盤システム
- **動画処理**: フレーム抽出、前処理、シーン検出
- **AI/ML処理**: 牌検出、牌分類、ゲーム状態追跡
- **教師データ作成**: アノテーション管理、データセット管理

#### フェーズ2: 学習システム
- **モデル訓練**: 検出・分類モデルの学習
- **継続学習**: 既存モデルからの継続学習
- **ハイパーパラメータ最適化**: 自動調整機能
- **性能評価**: 詳細な評価レポート生成

#### フェーズ3: 統合システム
- **牌譜生成**: 天鳳JSON形式での出力
- **品質検証**: 出力牌譜の妥当性検証と信頼度評価
- **パフォーマンス最適化**: メモリ・CPU・GPU使用率の最適化
- **統合システム**: エンドツーエンドの自動処理

#### フェーズ4: 完成システム
- **Webインターフェース**: ブラウザベースの操作画面
- **統合テスト**: 全工程の自動テスト
- **包括的ドキュメント**: 完全な使用ガイド

### 技術仕様

- **言語**: Python 3.9+
- **主要ライブラリ**: OpenCV, NumPy, PyTorch, Pandas, Flask
- **対応形式**: MP4, AVI, MOV等の動画ファイル
- **出力形式**: JSON (天鳳形式)
- **目標精度**: 95%以上
- **処理速度**: 約2-5 FPS（CPU）、10-20 FPS（GPU）

---

## アーキテクチャ

### システム構成図

```
麻雀牌譜作成システム
├── 動画処理層 (src/video/)
│   ├── VideoProcessor: フレーム抽出・前処理
│   └── FrameExtractor: 教師データ用フレーム抽出
├── AI/ML処理層 (src/detection/, src/classification/)
│   ├── TileDetector: 牌検出 (YOLO/Detectron2)
│   ├── TileClassifier: 牌分類 (CNN/ViT)
│   └── AIPipeline: AI処理統合
├── ゲーム状態管理層 (src/game/, src/tracking/)
│   ├── GameState: 状態管理
│   ├── StateTracker: 状態追跡
│   ├── ActionDetector: アクション検出
│   └── GamePipeline: ゲーム処理統合
├── 学習システム層 (src/training/)
│   ├── DatasetManager: データセット管理
│   ├── TrainingManager: 学習管理
│   ├── ModelTrainer: モデル訓練
│   ├── ModelEvaluator: 性能評価
│   └── LearningScheduler: スケジューリング
├── 最適化・検証層 (src/optimization/, src/validation/)
│   ├── PerformanceOptimizer: パフォーマンス最適化
│   ├── QualityValidator: 品質検証
│   ├── MemoryOptimizer: メモリ最適化
│   └── GPUOptimizer: GPU最適化
├── 統合・出力層 (src/integration/, src/output/)
│   ├── SystemIntegrator: システム統合
│   └── TenhouJsonFormatter: 天鳳JSON出力
├── Webインターフェース (web_interface/)
│   ├── Flask アプリケーション
│   ├── データ管理画面
│   ├── ラベリング画面
│   └── 学習管理画面
└── ユーティリティ (src/utils/)
    ├── ConfigManager: 設定管理
    ├── Logger: ログ管理
    └── TileDefinitions: 牌定義
```

### データフロー

1. **動画入力** → フレーム抽出 → 前処理
2. **AI処理** → 牌検出 → 牌分類 → バッチ処理
3. **ゲーム状態** → 状態追跡 → 履歴管理 → ルール適用
4. **牌譜生成** → 天鳳JSON形式変換 → 品質検証 → 出力

### コンポーネント間通信

- **設定管理**: 全コンポーネントで共有される`ConfigManager`
- **ログ管理**: 統一されたログシステム
- **データ形式**: 標準化されたデータ構造
- **エラーハンドリング**: 一貫したエラー処理

---

## インストール・セットアップ

### 自動インストール（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/your-username/mahjong-system.git
cd mahjong-system

# 自動インストールスクリプトを実行
./install.sh

# 開発用依存関係も含める場合
./install.sh --dev
```

### 手動インストール

#### 1. システム依存関係

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev
```

**macOS:**
```bash
brew install python@3.9 opencv git
```

#### 2. Python環境（uvを使用）

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# 仮想環境の有効化
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate     # Windows
```

#### 3. ディレクトリ作成

```bash
mkdir -p data/{input,output,temp,training} logs models web_interface/{uploads,logs}
```

### Docker環境

#### 基本的な使用

```bash
# イメージをビルド
docker build -t mahjong-system .

# コンテナを実行
docker run -v $(pwd)/data:/app/data mahjong-system python main.py status

# docker-composeを使用
docker-compose up mahjong-system
```

#### 開発環境

```bash
# 開発用コンテナを起動
docker-compose --profile dev up mahjong-system-dev
```

#### GPU対応

```bash
# GPU対応版を起動
docker-compose --profile gpu up mahjong-system-gpu
```

### 環境変数設定

```bash
export PYTHONPATH=/path/to/mahjong-system
export LOG_LEVEL=INFO
export GPU_ENABLED=1
```

---

## 使用方法

### コマンドライン使用

#### 基本的な使用方法

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# システム状態確認
python main.py status

# 単一動画を処理（天鳳JSON形式）
python main.py process input_video.mp4

# 出力パスを指定
python main.py process input_video.mp4 --output result.json

# バッチ処理
python main.py batch input_directory output_directory

# 牌譜の品質検証
python main.py validate record.json

# システム最適化
python main.py optimize
```

#### 詳細なオプション

```bash
# ヘルプ表示
python main.py --help

# 詳細ログ出力
python main.py --verbose process input_video.mp4

# 品質検証を無効化
python main.py process input_video.mp4 --no-validation

# 並列処理数を指定
python main.py batch input_dir output_dir --workers 8
```

### Webインターフェース使用

#### 起動方法

```bash
# Webインターフェースを起動
cd web_interface
python run.py

# ブラウザでアクセス
# http://localhost:5000
```

#### 機能

1. **データ管理**
   - 動画アップロード
   - データセット管理
   - 統計表示

2. **ラベリング**
   - 半自動ラベリング
   - 手動修正
   - 品質確認

3. **学習管理**
   - 学習設定
   - 進捗監視
   - 結果確認

### プログラマティック使用

#### 基本的な使用例

```python
from src.utils.config import ConfigManager
from src.integration.system_integrator import SystemIntegrator
from src.video.video_processor import VideoProcessor
from src.pipeline.ai_pipeline import AIPipeline
from src.pipeline.game_pipeline import GamePipeline

# 設定管理を初期化
config_manager = ConfigManager()

# コンポーネントを初期化
video_processor = VideoProcessor(config_manager)
ai_pipeline = AIPipeline(config_manager)
game_pipeline = GamePipeline()

# システム統合
integrator = SystemIntegrator(
    config_manager, video_processor, ai_pipeline, game_pipeline
)

# 動画を処理
result = integrator.process_video_complete(
    video_path="input_video.mp4",
    output_path="output_record.json"
)

print(f"処理結果: {result.success}")
print(f"品質スコア: {result.quality_score}")
```

#### 学習システムの使用

```python
from src.training.learning.training_manager import TrainingManager, TrainingConfig

# 学習管理を初期化
training_manager = TrainingManager(config_manager)

# 学習設定
config = TrainingConfig(
    model_type="classification",
    model_name="tile_classifier_v1",
    dataset_version_id="your_dataset_version_id",
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# 学習を開始
session_id = training_manager.start_training(config)
```

---

## API仕様

### REST API エンドポイント

#### 動画処理

```http
POST /api/process
Content-Type: multipart/form-data

Parameters:
- video_file: 動画ファイル
- enable_validation: 品質検証有効化 (optional, default: true)

Response:
{
  "success": true,
  "output_path": "path/to/output.json",
  "quality_score": 85.5,
  "processing_time": 120.5
}
```

#### バッチ処理

```http
POST /api/batch
Content-Type: application/json

{
  "input_directory": "path/to/input",
  "output_directory": "path/to/output",
  "max_workers": 4
}

Response:
{
  "success": true,
  "total_files": 10,
  "successful_count": 9,
  "success_rate": 0.9,
  "processing_time": 1200.0
}
```

#### 品質検証

```http
POST /api/validate
Content-Type: application/json

{
  "record_path": "path/to/record.json"
}

Response:
{
  "success": true,
  "overall_score": 85.5,
  "category_scores": {
    "structure": 90.0,
    "content": 85.0,
    "consistency": 80.0
  },
  "issues": [],
  "recommendations": []
}
```

### Python API

#### SystemIntegrator クラス

```python
class SystemIntegrator:
    def process_video_complete(
        self,
        video_path: str,
        output_path: str,
        format_type: str = "tenhou_json",
        enable_optimization: bool = True,
        enable_validation: bool = True
    ) -> IntegrationResult:
        """動画を完全処理"""

    def process_batch(
        self,
        video_files: List[str],
        output_directory: str,
        format_type: str = "tenhou_json",
        max_workers: int = None
    ) -> Dict[str, Any]:
        """バッチ処理"""

    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
```

#### TrainingManager クラス

```python
class TrainingManager:
    def start_training(self, config: TrainingConfig) -> str:
        """学習開始"""

    def continue_training(
        self,
        base_session_id: str,
        new_config: TrainingConfig
    ) -> str:
        """継続学習"""

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """学習状況取得"""

    def compare_models(self, session_ids: List[str]) -> Dict[str, Any]:
        """モデル比較"""
```

---

## 設定ガイド

### 設定ファイル (config.yaml)

```yaml
# 動画処理設定
video:
  frame_extraction:
    fps: 1  # フレーム抽出レート
    output_format: "jpg"
    quality: 95
    max_frames: 1000

# AI/ML設定
ai:
  detection:
    confidence_threshold: 0.5
    model_path: "models/tile_detector.pt"
  classification:
    confidence_threshold: 0.8
    model_path: "models/tile_classifier.pt"
    num_classes: 34

# 学習設定
training:
  training_root: "data/training"
  dataset_root: "data/training/dataset"
  database_path: "data/training/dataset.db"
  default_epochs: 100
  default_batch_size: 32
  default_learning_rate: 0.001
  default_validation_split: 0.2

# システム設定
system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: true
  optimization_level: "balanced"

# ディレクトリ設定
directories:
  input: "data/input"
  output: "data/output"
  temp: "data/temp"
  models: "models"
  logs: "logs"

# 品質検証設定
validation:
  min_confidence: 0.7
  max_error_rate: 0.05
  required_completeness: 0.9

# Webインターフェース設定
web:
  host: "0.0.0.0"
  port: 5000
  debug: false
  upload_folder: "web_interface/uploads"
  max_file_size: "100MB"

# 牌定義
tiles:
  manzu: ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
  pinzu: ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"]
  souzu: ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s"]
  jihai: ["東", "南", "西", "北", "白", "發", "中"]
```

### 環境別設定

#### 開発環境

```yaml
system:
  max_workers: 2
  memory_limit: "4GB"
  gpu_enabled: false

web:
  debug: true

logging:
  level: "DEBUG"
```

#### 本番環境

```yaml
system:
  max_workers: 8
  memory_limit: "16GB"
  gpu_enabled: true

web:
  debug: false

logging:
  level: "INFO"
```

---

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー

**症状:**
```
ERROR: Out of memory during processing
RuntimeError: CUDA out of memory
```

**解決方法:**
```yaml
# config.yamlで調整
ai:
  training:
    batch_size: 4  # デフォルト: 8から減らす

system:
  max_workers: 2  # デフォルト: 4から減らす
  memory_limit: "4GB"  # メモリ制限を設定
```

#### 2. GPU認識されない

**症状:**
```
WARNING: GPU not available, using CPU
```

**確認方法:**
```bash
# GPU状態確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDA環境確認
nvidia-smi
```

**解決方法:**
- CUDA環境の再インストール
- PyTorchのGPU版インストール
- 設定でGPUを有効化

#### 3. 動画読み込みエラー

**症状:**
```
ERROR: Failed to read video file
cv2.error: OpenCV Error
```

**確認方法:**
```bash
# OpenCVの確認
python -c "import cv2; print(cv2.__version__)"

# コーデックの確認
ffmpeg -codecs | grep h264
```

**解決方法:**
- OpenCVの再インストール
- ffmpegのインストール
- 動画ファイル形式の確認

#### 4. モデルファイルが見つからない

**症状:**
```
WARNING: Model file not found: models/tile_detector.pt
```

**解決方法:**
- モデルファイルの配置確認
- 設定ファイルのパス確認
- 学習済みモデルのダウンロード

#### 5. Webインターフェースエラー

**症状:**
```
ERROR: Failed to start web server
Address already in use
```

**解決方法:**
```bash
# ポート使用状況確認
lsof -i :5000

# 別ポートで起動
python run.py --port 5001
```

### ログ確認

```bash
# システムログ
tail -f logs/mahjong_system.log

# エラーログのみ
grep ERROR logs/mahjong_system.log

# Webインターフェースログ
tail -f web_interface/logs/web_interface.log
```

### デバッグモード

```bash
# 詳細ログで実行
python main.py --verbose process input_video.mp4

# デバッグ設定
export LOG_LEVEL=DEBUG
python main.py status
```

---

## パフォーマンス最適化

### ベンチマーク結果

- **処理速度**: 約2-5 FPS（CPU）、10-20 FPS（GPU）
- **メモリ使用量**: 2-8GB（設定により調整可能）
- **精度**: 検出精度 90%+、分類精度 95%+
- **対応動画**: 1080p、30分程度の動画を20-60分で処理

### 最適化のヒント

#### 1. ハードウェア最適化

```yaml
# GPU使用
system:
  gpu_enabled: true

# 並列処理
system:
  max_workers: 8  # CPU数に応じて調整
```

#### 2. メモリ最適化

```yaml
# バッチサイズ調整
ai:
  training:
    batch_size: 16  # メモリ使用量に応じて調整

# フレーム制限
video:
  frame_extraction:
    max_frames: 500  # 必要に応じて制限
```

#### 3. 処理最適化

```yaml
# フレームレート調整
video:
  frame_extraction:
    fps: 0.5  # 必要に応じて下げる

# 信頼度閾値調整
ai:
  detection:
    confidence_threshold: 0.7  # 高めに設定して処理量削減
```

#### 4. 自動最適化

```python
from src.optimization.performance_optimizer import PerformanceOptimizer

optimizer = PerformanceOptimizer(config_manager)

# システム最適化実行
result = optimizer.optimize_full_system()

# 推奨事項取得
recommendations = optimizer.get_optimization_recommendations()
```

---

## 品質保証

### テスト実行

#### 単体テスト

```bash
# 全テストを実行
pytest

# 特定のテストを実行
pytest tests/test_integration.py -v

# カバレッジ付きでテスト
pytest --cov=src --cov-report=html
```

#### 統合テスト

```bash
# 統合テストを実行
pytest tests/test_integration.py::TestEndToEndIntegration -v

# パフォーマンステスト
pytest tests/test_performance.py -v
```

#### エンドツーエンドテスト

```bash
# 完全ワークフローテスト
python demo_complete_workflow.py

# 学習システムテスト
python demo_learning_system.py

# Webインターフェーステスト
python -m pytest tests/test_web_interface.py -v
```

### コード品質チェック

```bash
# リンティング
flake8 src tests

# フォーマット
black src tests
isort src tests

# 型チェック
mypy src
```

### 品質メトリクス

#### コードカバレッジ

```bash
# カバレッジ測定
pytest --cov=src --cov-report=html --cov-report=term

# 目標: 80%以上
```

#### 静的解析

```bash
# 複雑度チェック
radon cc src --min B

# セキュリティチェック
bandit -r src
```

### 品質検証

```python
from src.validation.quality_validator import QualityValidator

validator = QualityValidator(config_manager)

# 牌譜品質検証
result = validator.validate_record_file("output_record.json")

print(f"品質スコア: {result.overall_score}")
print(f"問題点: {len(result.issues)}")
```

---

## 開発・拡張

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
./install.sh --dev

# または手動で
uv add --dev pytest pytest-cov flake8 black isort mypy
```

### コーディング規約

- **PEP 8**に準拠
- **型ヒント**を使用
- **docstring**を記述
- **テスト**を追加

### 新機能の追加

#### 1. 新しい検出モデルの追加

```python
# src/detection/new_detector.py
from .tile_detector import TileDetector

class NewTileDetector(TileDetector):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        # 新しい実装

    def detect_tiles(self, frame):
        # 新しい検出ロジック
        pass
```

#### 2. 新しい出力形式の追加

```python
# src/output/new_formatter.py
from .tenhou_json_formatter import TenhouJsonFormatter

class NewFormatter(TenhouJsonFormatter):
    def format_game_record(self, game_record):
        # 新しい形式での出力
        pass
```

#### 3. 新しいWebページの追加

```python
# web_interface/routes/new_route.py
from flask import Blueprint, render_template

new_bp = Blueprint('new', __name__)

@new_bp.route('/new')
def new_page():
    return render_template('new.html')
```

### 貢献ガイドライン

#### 1. 開発フロー

1. フォークしてクローン
2. 機能ブランチを作成: `git checkout -b feature/new-feature`
3. 変更をコミット: `git commit -am 'Add new feature'`
4. テストを実行: `pytest`
5. ブランチにプッシュ: `git push origin feature/new-feature`
6. プルリクエストを作成

#### 2. コミットメッセージ

```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント更新
style: コードスタイル修正
refactor: リファクタリング
test: テスト追加・修正
chore: その他の変更
```

#### 3. テスト要件

- 新機能には必ずテストを追加
- カバレッジ80%以上を維持
- 統合テストも更新

---

## 付録

### サンプルデータ

#### 天鳳JSON形式の例

```json
{
  "game_info": {
    "rule": "東南戦",
    "players": ["Player1", "Player2", "Player3", "Player4"],
    "start_time": "2024-01-01T12:00:00Z"
  },
  "rounds": [
    {
      "round_number": 1,
      "round_name": "東1局",
      "dealer": 0,
      "actions": [
        {
          "player": 0,
          "action": "draw",
          "tiles": ["1m"],
          "timestamp": 1.5
        },
        {
          "player": 0,
          "action": "discard",
          "tiles": ["9p"],
          "timestamp": 2.0
        }
      ]
    }
  ]
}
```

### 設定テンプレート

#### 最小設定

```yaml
directories:
  input: "data/input"
  output: "data/output"
  temp: "data/temp"

system:
  max_workers: 2
  gpu_enabled: false
```

#### 完全設定

```yaml
# 完全な設定例は上記の「設定ガイド」を参照
```

### FAQ

#### Q: 処理が遅い場合はどうすればよいですか？

A: 以下を確認してください：
1. GPU使用の有効化
2. バッチサイズの調整
3. 並列処理数の増加
4. フレーム抽出レートの調整

#### Q: 精度が低い場合はどうすればよいですか？

A: 以下を試してください：
1. 信頼度閾値の調整
2. 追加の教師データでの再学習
3. データ拡張の有効化
4. モデルアーキテクチャの変更

#### Q: メモリ不足エラーが発生します

A: 以下を調整してください：
1. バッチサイズを小さくする
2. 並列処理数を減らす
3. フレーム数制限を設定
4. メモリ制限を設定

---

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## サポート

- **Issues**: [GitHub Issues](https://github.com/your-username/mahjong-system/issues)
- **Wiki**: [GitHub Wiki](https://github.com/your-username/mahjong-system/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/mahjong-system/discussions)

## 謝辞

- OpenCV コミュニティ
- PyTorch チーム
- 麻雀AI研究コミュニティ

---

**注意**: このシステムは研究・教育目的で開発されています。商用利用の際は適切なライセンスを確認してください。
