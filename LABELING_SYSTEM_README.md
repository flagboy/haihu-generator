# 手牌学習データ作成システム

## 概要

手牌学習データ作成システムは、麻雀対局動画から手牌領域を抽出し、各牌を正確にラベリングすることで、AI学習用の高品質なデータセットを作成するシステムです。

## 主な機能

### 1. セッション管理
- 動画ごとのラベリングセッションを作成・管理
- 進捗の保存と再開
- 複数ユーザーによる協調作業対応

### 2. 手牌領域検出
- 4プレイヤー（自分、右、対面、左）の手牌領域を設定
- 手動設定と自動検出のハイブリッド方式
- 設定の保存と再利用

### 3. 牌分割・ラベリング
- 手牌領域から個々の牌を自動分割
- 直感的なラベリングインターフェース
- キーボードショートカットによる高速入力

### 4. データエクスポート
- COCO形式
- YOLO形式
- 天鳳JSON形式

## システム構成

```
src/training/labeling/
├── core/                      # コア機能
│   ├── hand_area_detector.py  # 手牌領域検出
│   ├── video_processor.py     # 動画処理
│   ├── tile_splitter.py       # 牌分割
│   └── labeling_session.py    # セッション管理
├── api/                       # API層
│   ├── routes.py             # RESTful API
│   └── websocket.py          # WebSocket通信
└── utils/                     # ユーティリティ

web_interface/
├── static/js/
│   ├── labeling-app.js       # メインアプリケーション
│   └── modules/              # モジュール群
└── templates/
    └── labeling.html         # UIテンプレート
```

## 使用方法

### 1. システムの起動

```bash
# Webインターフェースを起動
cd web_interface
python run.py
```

ブラウザで `http://localhost:5000` にアクセスし、「ラベリング」タブを選択します。

### 2. 新規セッションの作成

1. 動画ファイルを選択
2. 「新規セッション」ボタンをクリック
3. ユーザー名を入力

### 3. 手牌領域の設定

1. プレイヤーを選択（自分/右/対面/左）
2. 「手牌領域を設定」ボタンをクリック
3. キャンバス上でドラッグして領域を指定
4. 全プレイヤー分を設定

### 4. ラベリング作業

#### キーボードショートカット
- **数字キー（1-9）**: 牌の数字
- **Q/W/E**: 萬子/筒子/索子の切り替え
- **A/S/D/F/G/H/J**: 字牌（東南西北白發中）
- **矢印キー**: 牌/フレームの移動
- **Enter**: 現在のフレームを確定
- **Escape**: 操作のキャンセル

### 5. データのエクスポート

1. 「エクスポート」ボタンをクリック
2. 形式を選択（COCO/YOLO/天鳳）
3. ファイルがダウンロードされます

## API仕様

### RESTful API

#### セッション管理
- `GET /api/labeling/sessions/` - セッション一覧
- `POST /api/labeling/sessions/` - 新規セッション作成
- `GET /api/labeling/sessions/{id}` - セッション詳細

#### フレーム処理
- `POST /api/labeling/frames/{session_id}/extract` - フレーム抽出
- `GET /api/labeling/frames/{session_id}/{frame_number}` - フレーム取得
- `POST /api/labeling/frames/{session_id}/{frame_number}/tiles` - 牌分割

#### アノテーション
- `POST /api/labeling/annotations/{session_id}` - アノテーション追加
- `GET /api/labeling/annotations/{session_id}/export` - エクスポート

### WebSocket イベント

#### 送信イベント
- `join_session` - セッションに参加
- `frame_update` - フレーム更新通知
- `label_update` - ラベル更新通知
- `progress_update` - 進捗更新通知

#### 受信イベント
- `user_joined` - ユーザー参加通知
- `frame_updated` - フレーム更新通知
- `label_updated` - ラベル更新通知
- `progress_updated` - 進捗更新通知

## データ形式

### セッションデータ
```json
{
  "session_id": "uuid",
  "video_info": {
    "path": "/path/to/video.mp4",
    "fps": 30.0,
    "frame_count": 1000,
    "width": 1920,
    "height": 1080
  },
  "hand_regions": {
    "bottom": {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15},
    "right": {"x": 0.8, "y": 0.3, "w": 0.15, "h": 0.4},
    "top": {"x": 0.15, "y": 0.1, "w": 0.7, "h": 0.15},
    "left": {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.4}
  },
  "annotations": {
    "0": {
      "timestamp": 0.0,
      "players": {
        "bottom": {
          "tiles": [
            {"index": 0, "label": "1m", "x": 100, "y": 800, "w": 40, "h": 60}
          ]
        }
      }
    }
  }
}
```

## トラブルシューティング

### 動画が読み込めない
- 対応形式: MP4, AVI, MOV
- 推奨解像度: 1920x1080以上
- ファイルパスに日本語が含まれていないか確認

### 手牌領域が正しく検出されない
- 動画の明るさ・コントラストを確認
- 手動で領域を設定することを推奨

### ラベリングが保存されない
- セッションが正しく作成されているか確認
- ブラウザのコンソールでエラーを確認

## 開発者向け情報

### テストの実行
```bash
# 統合テスト
uv run pytest tests/test_labeling_integration.py -v

# 全テスト
uv run pytest tests/ -v
```

### データ移行
```bash
# 旧システムからのデータ移行
python scripts/migrate_labeling_data.py --old-dir . --new-dir data/training
```

## 今後の拡張予定

- AI自動ラベリング機能の強化
- リアルタイム動画処理
- クラウドストレージ対応
- モバイル対応UI
