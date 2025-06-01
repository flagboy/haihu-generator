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
- **主要ライブラリ**: OpenCV, NumPy, PyTorch, Pandas
- **対応形式**: MP4, AVI, MOV等の動画ファイル
- **出力形式**: JSON (MJSCORE), XML (天鳳)
- **目標精度**: 95%以上

## インストール

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
# 仮想環境をアクティベート
source venv/bin/activate

# システム状態確認
python main.py status

# 単一動画を処理
python main.py process input_video.mp4

# 出力形式を指定
python main.py process input_video.mp4 --format tenhou --output result.xml

# バッチ処理
python main.py batch input_directory output_directory

# 牌譜の品質検証
python main.py validate record.json

# システム最適化
python main.py optimize
```

### 詳細なオプション

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
./install.sh --dev

# または手動で
pip install pytest pytest-cov flake8 black isort mypy
```

### テスト実行

```bash
# 全テストを実行
pytest

# 特定のテストを実行
pytest tests/test_integration.py -v

# カバレッジ付きでテスト
pytest --cov=src --cov-report=html

# パフォーマンステスト
pytest tests/test_performance.py -v
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
│   └── バッチ処理
├── ゲーム状態管理 (src/game/, src/tracking/)
│   ├── 状態追跡
│   ├── 履歴管理
│   └── ルールエンジン
├── 統合パイプライン (src/pipeline/)
│   ├── AIパイプライン
│   └── ゲームパイプライン
├── 最適化・検証 (src/optimization/, src/validation/)
│   ├── パフォーマンス最適化
│   ├── 品質検証
│   └── システム統合
└── ユーティリティ (src/utils/)
    ├── 設定管理
    ├── ログ管理
    └── 牌定義
```

### データフロー

1. **動画入力** → フレーム抽出 → 前処理
2. **AI処理** → 牌検出 → 牌分類 → バッチ処理
3. **ゲーム状態** → 状態追跡 → 履歴管理 → ルール適用
4. **牌譜生成** → 形式変換 → 品質検証 → 出力

## パフォーマンス

### ベンチマーク結果

- **処理速度**: 約2-5 FPS（CPU）、10-20 FPS（GPU）
- **メモリ使用量**: 2-8GB（設定により調整可能）
- **精度**: 検出精度 90%+、分類精度 95%+
- **対応動画**: 1080p、30分程度の動画を20-60分で処理

### 最適化のヒント

1. **バッチサイズ調整**: メモリ使用量に応じて調整
2. **並列処理**: CPU数に応じてワーカー数を調整
3. **GPU使用**: 可能な場合はGPUを有効化
4. **フレームレート**: 必要に応じてフレーム抽出レートを調整

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー

```bash
# バッチサイズを減らす
# config.yamlで調整
ai:
  training:
    batch_size: 4  # デフォルト: 8

# または並列数を減らす
system:
  max_workers: 2  # デフォルト: 4
```

#### 2. GPU認識されない

```bash
# GPU状態確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDA環境確認
nvidia-smi
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

## 貢献

### 開発への参加

1. フォークしてクローン
2. 機能ブランチを作成: `git checkout -b feature/new-feature`
3. 変更をコミット: `git commit -am 'Add new feature'`
4. ブランチにプッシュ: `git push origin feature/new-feature`
5. プルリクエストを作成

### コーディング規約

- PEP 8に準拠
- 型ヒントを使用
- docstringを記述
- テストを追加

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照

## 更新履歴

### v1.0.0 (2024-XX-XX)
- 初回リリース
- 基本的な動画処理機能
- AI/ML パイプライン
- 牌譜生成機能

### v1.1.0 (フェーズ4)
- パフォーマンス最適化
- 品質検証システム
- システム統合
- Docker対応
- CI/CD パイプライン

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
