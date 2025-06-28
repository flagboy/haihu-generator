# 麻雀牌譜作成システム

動画から麻雀の牌譜を自動生成するAIシステム

## 概要

このシステムは、麻雀の対局動画（MP4等）を入力として受け取り、動画から麻雀牌の動きや配置を認識して、標準的な牌譜形式で出力します。

### 主な機能

- **動画処理**: フレーム抽出、前処理、シーン検出
- **AI/ML処理**: 牌検出、牌分類、ゲーム状態追跡
- **牌譜生成**: 天鳳形式、MJSCORE形式での出力
- **品質検証**: 出力牌譜の妥当性検証と信頼度評価
- **パフォーマンス最適化**: メモリ・CPU・GPU使用率の最適化
- **統合システム**: エンドツーエンドの自動処理

### 技術仕様

- **言語**: Python 3.9+
- **パッケージマネージャー**: uv（高速なPythonパッケージ管理）
- **主要ライブラリ**: OpenCV, NumPy, PyTorch, Pandas
- **対応形式**: MP4, AVI, MOV等の動画ファイル
- **出力形式**: JSON (MJSCORE), XML (天鳳)
- **目標精度**: 95%以上
- **最新機能**:
  - バッチ処理最適化
  - セキュリティ強化（XSS/CSRF対策、ファイルアップロード検証）
  - パフォーマンス最適化（メモリ管理、GPU最適化）

## インストール

### uvを使用したインストール（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/flagboy/haihu-generator.git
cd haihu-generator

# uvのインストール（まだインストールしていない場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync

# 開発用依存関係も含める場合
uv sync --dev
```

### 自動インストールスクリプト

```bash
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

#### 2. Python環境

```bash
# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt
```

#### 3. ディレクトリ作成

```bash
mkdir -p data/{input,output,temp} logs models
```

## 使用方法

### 基本的な使用方法

```bash
# uvを使用して実行（仮想環境の手動アクティベートは不要）
uv run python main.py status

# 単一動画を処理
uv run python main.py process input_video.mp4

# 出力形式を指定
uv run python main.py process input_video.mp4 --format tenhou --output result.xml

# バッチ処理（最適化済み）
uv run python main.py batch input_directory output_directory

# 牌譜の品質検証
uv run python main.py validate record.json

# システム最適化
uv run python main.py optimize
```

### 詳細なオプション

```bash
# ヘルプ表示
uv run python main.py --help

# 詳細ログ出力
uv run python main.py --verbose process input_video.mp4

# 品質検証を無効化
uv run python main.py process input_video.mp4 --no-validation

# 並列処理数を指定（最適化済み）
uv run python main.py batch input_dir output_dir --workers 8

# バッチサイズを自動最適化
uv run python main.py process input_video.mp4 --auto-optimize-batch
```

## 設定

### 設定ファイル (config.yaml)

```yaml
# 動画処理設定
video:
  frame_extraction:
    fps: 1  # フレーム抽出レート
    output_format: "jpg"
    quality: 95

# AI/ML設定
ai:
  detection:
    confidence_threshold: 0.5
  classification:
    confidence_threshold: 0.8

# システム設定
system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: true

# ディレクトリ設定
directories:
  input: "data/input"
  output: "data/output"
  temp: "data/temp"
  models: "models"
  logs: "logs"
```

### 環境変数

```bash
export PYTHONPATH=/path/to/mahjong-system
export LOG_LEVEL=INFO
export GPU_ENABLED=1
```

## Docker使用

### 基本的な使用

```bash
# イメージをビルド
docker build -t mahjong-system .

# コンテナを実行
docker run -v $(pwd)/data:/app/data mahjong-system python main.py status

# docker-composeを使用
docker-compose up mahjong-system
```

### 開発環境

```bash
# 開発用コンテナを起動
docker-compose --profile dev up mahjong-system-dev
```

### GPU対応

```bash
# GPU対応版を起動
docker-compose --profile gpu up mahjong-system-gpu
```

## 開発

### 開発環境のセットアップ

```bash
# 開発用依存関係をインストール
uv sync --dev

# または個別にインストール
uv add --dev pytest pytest-cov ruff mypy
```

### テスト実行

```bash
# 全テストを実行
uv run pytest

# 特定のテストを実行
uv run pytest tests/test_integration.py -v

# カバレッジ付きでテスト
uv run pytest --cov=src --cov-report=html

# パフォーマンステスト
uv run pytest tests/test_performance.py -v

# 最適化テスト
uv run pytest tests/optimization/test_batch_processing.py -v
```

### コード品質チェック

```bash
# リンティングとフォーマット（ruffを使用）
uv run ruff check src tests
uv run ruff format src tests

# 型チェック
uv run mypy src

# pre-commitフックの実行
uv run pre-commit run --all-files
```

## アーキテクチャ

### システム構成

```
麻雀牌譜作成システム
├── 動画処理モジュール (src/video/)
│   ├── フレーム抽出
│   ├── 前処理
│   └── シーン検出
├── AI/MLモジュール (src/detection/, src/classification/)
│   ├── 牌検出 (YOLO/Detectron2)
│   ├── 牌分類 (CNN/ViT)
│   └── バッチ処理最適化
├── ゲーム状態管理 (src/game/, src/tracking/)
│   ├── 状態追跡
│   ├── 履歴管理
│   └── ルールエンジン
├── 統合パイプライン (src/pipeline/)
│   ├── AIパイプライン
│   ├── 最適化AIパイプライン
│   └── ゲームパイプライン
├── 最適化・検証 (src/optimization/, src/validation/)
│   ├── パフォーマンス最適化
│   ├── バッチ処理最適化
│   ├── メモリ/GPU最適化
│   ├── 品質検証
│   └── システム統合
├── Webインターフェース (web_interface/)
│   ├── Flaskアプリケーション
│   ├── セキュリティ機能
│   └── REST API
└── ユーティリティ (src/utils/)
    ├── 設定管理
    ├── ログ管理
    ├── 牌定義
    └── デバイス管理
```

### データフロー

1. **動画入力** → フレーム抽出 → 前処理
2. **AI処理** → 牌検出 → 牌分類 → バッチ処理
3. **ゲーム状態** → 状態追跡 → 履歴管理 → ルール適用
4. **牌譜生成** → 形式変換 → 品質検証 → 出力

## パフォーマンス

### ベンチマーク結果

- **処理速度**:
  - CPU: 約2-5 FPS（最適化前）→ 5-10 FPS（最適化後）
  - GPU: 10-20 FPS（最適化前）→ 20-40 FPS（最適化後）
- **メモリ使用量**: 2-8GB（自動バッチサイズ調整により最適化）
- **精度**: 検出精度 90%+、分類精度 95%+
- **対応動画**: 1080p、30分程度の動画を10-30分で処理（最適化後）

### 最適化機能

1. **自動バッチサイズ最適化**:
   - メモリ使用量をリアルタイムで監視
   - 成功/失敗に基づき動的に調整
   - BatchSizeOptimizerによる最適バッチサイズ探索

2. **並列バッチ処理**:
   - CPUコア数に基づく自動ワーカー数設定
   - 非同期I/Oとバッチ処理の統合

3. **GPU/MPS最適化**:
   - CUDA/MPSデバイスの自動検出
   - メモリピンニングと非同期転送

4. **メモリ管理**:
   - 定期的なガベージコレクション
   - GPUメモリの明示的な解放

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー

```bash
# 自動最適化を有効にする
uv run python main.py process video.mp4 --auto-optimize-batch

# またはconfig.yamlで調整
ai:
  batch_processing:
    auto_optimize: true
    memory_fraction: 0.85  # 利用可能メモリの85%まで使用
system:
  max_workers: 2  # デフォルト: 4
```

#### 2. GPU認識されない

```bash
# GPU状態確認（自動デバイス検出）
uv run python -c "from src.utils.device_utils import get_device_info; print(get_device_info())"

# PyTorchのGPU確認
uv run python -c "import torch; print(torch.cuda.is_available())"

# CUDA環境確認
nvidia-smi

# Apple Siliconの場合（MPS）
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

#### 3. 動画読み込みエラー

```bash
# OpenCVの確認
python -c "import cv2; print(cv2.__version__)"

# コーデックの確認
ffmpeg -codecs | grep h264
```

### ログ確認

```bash
# システムログ
tail -f logs/mahjong_system.log

# エラーログのみ
grep ERROR logs/mahjong_system.log
```

### デバッグモード

```bash
# 詳細ログで実行
python main.py --verbose process input_video.mp4

# デバッグ設定
export LOG_LEVEL=DEBUG
```

## Webインターフェースとセキュリティ

### Webインターフェース起動

```bash
# Webインターフェースを起動
cd web_interface
uv run python app.py

# または
uv run python web_interface/run.py
```

アクセス: http://localhost:5000

### セキュリティ機能

1. **ファイルアップロード保護**:
   - ファイルタイプの厳密な検証（MIMEタイプとマジックナンバー）
   - ファイルサイズ制限（最大2GB）
   - パストラバーサル攻撃の防御

2. **XSS/CSRF対策**:
   - HTMLエスケープ処理
   - CSRFトークン検証
   - Content Security Policy (CSP)

3. **セキュリティヘッダー**:
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - X-XSS-Protection: 1; mode=block
   - Strict-Transport-Security (HTTPS環境)

### API仕様

主要なAPIエンドポイント:

```bash
# 動画アップロード
POST /api/upload_video
Content-Type: multipart/form-data

# フレーム抽出
POST /api/extract_frames
{
  "video_path": "path/to/video.mp4",
  "config": {
    "interval_seconds": 1.0,
    "quality_threshold": 0.5
  }
}

# データセット統計
GET /api/dataset/statistics

# 学習開始
POST /api/training/start
{
  "model_type": "detection",
  "epochs": 100,
  "batch_size": 32
}
```

詳細なAPI仕様書は`docs/api_specification.md`を参照してください。

## 貢献

### 開発への参加

1. フォークしてクローン
2. 機能ブランチを作成: `git checkout -b feature/new-feature`
3. 変更をコミット: `git commit -am 'Add new feature'`
4. ブランチにプッシュ: `git push origin feature/new-feature`
5. プルリクエストを作成

### コーディング規約

- PEP 8に準拠（ruffで自動チェック）
- 型ヒントを使用（mypyで検証）
- docstringを記述（Google Style）
- テストを追加（pytest、カバレッジ80%以上）
- pre-commitフックを使用

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 更新履歴

### v2.0.0 (2024-12-XX) - 最新
- バッチ処理最適化
  - 自動バッチサイズ調整
  - 並列バッチ処理
  - メモリ最適化
- セキュリティ強化
  - ファイルアップロード検証
  - XSS/CSRF対策
  - セキュリティヘッダー
- Webインターフェース改善
  - セキュアなファイル管理
  - リアルタイム進捗表示
- パフォーマンス向上（2-3倍高速化）

### v1.1.0 (2024-11-XX)
- パフォーマンス最適化
- 品質検証システム
- システム統合
- Docker対応
- CI/CD パイプライン

### v1.0.0 (2024-10-XX)
- 初回リリース
- 基本的な動画処理機能
- AI/ML パイプライン
- 牌譜生成機能

## サポート

- **Issues**: [GitHub Issues](https://github.com/flagboy/haihu-generator/issues)
- **Wiki**: [GitHub Wiki](https://github.com/flagboy/haihu-generator/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/flagboy/haihu-generator/discussions)

## 謝辞

- OpenCV コミュニティ
- PyTorch チーム
- 麻雀AI研究コミュニティ

---

**注意**: このシステムは研究・教育目的で開発されています。商用利用の際は適切なライセンスを確認してください。
