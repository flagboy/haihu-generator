# API仕様書

麻雀牌譜作成システム Web API仕様書

## 概要

このドキュメントは、麻雀牌譜作成システムのWeb APIの仕様を定義します。

### ベースURL

```
http://localhost:5000/api
```

### 認証

現在のバージョンでは認証は実装されていませんが、本番環境では以下の実装を推奨します：
- JWT トークンベース認証
- API キー認証
- OAuth 2.0

### レスポンス形式

すべてのAPIレスポンスはJSON形式で返されます。

```json
{
  "success": true,
  "data": {},
  "message": "Success message"
}
```

エラーレスポンス：

```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## エンドポイント一覧

### 動画管理

#### 動画アップロード

動画ファイルをアップロードします。

```
POST /api/upload_video
```

**リクエスト**

- Content-Type: `multipart/form-data`
- Body:
  - `video`: 動画ファイル（必須）
  - `csrf_token`: CSRFトークン（本番環境では必須）

**レスポンス**

```json
{
  "success": true,
  "video_info": {
    "id": "uuid-string",
    "filename": "video_name.mp4",
    "filepath": "/path/to/video",
    "duration": 1800.5,
    "fps": 30.0,
    "frame_count": 54015,
    "width": 1920,
    "height": 1080,
    "upload_time": "2024-12-20T10:00:00"
  }
}
```

**エラーコード**

- `400`: ファイルが選択されていない、ファイル形式が不正
- `413`: ファイルサイズが制限を超過（最大2GB）
- `500`: サーバーエラー

#### 動画一覧取得

アップロード済みの動画一覧を取得します。

```
GET /api/videos
```

**レスポンス**

```json
[
  {
    "id": 1,
    "name": "対局動画1.mp4",
    "path": "/path/to/video",
    "upload_date": "2024-12-20",
    "frame_count": 1000,
    "annotation_count": 500,
    "status": "annotated"
  }
]
```

#### 動画詳細取得

特定の動画の詳細情報を取得します。

```
GET /api/videos/{video_id}
```

**パラメータ**

- `video_id`: 動画ID（必須）

**レスポンス**

```json
{
  "id": 1,
  "name": "対局動画1.mp4",
  "path": "/path/to/video",
  "upload_date": "2024-12-20",
  "frame_count": 1000,
  "annotation_count": 500,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration": 1800.5
}
```

#### 動画削除

動画とその関連データを削除します。

```
DELETE /api/videos/{video_id}
```

**パラメータ**

- `video_id`: 動画ID（必須）
- `delete_files`: 物理ファイルも削除するか（オプション、デフォルト: false）

**レスポンス**

```json
{
  "message": "Video deleted successfully"
}
```

### フレーム処理

#### フレーム抽出

動画からフレームを抽出します。

```
POST /api/extract_frames
```

**リクエスト**

```json
{
  "video_path": "/path/to/video.mp4",
  "config": {
    "interval_seconds": 1.0,
    "quality_threshold": 0.5,
    "max_frames": 1000,
    "resize_width": 1280
  }
}
```

**レスポンス**

```json
{
  "success": true,
  "session_id": "uuid-string"
}
```

進捗はWebSocketで通知されます。

#### フレーム一覧取得

動画のフレーム一覧を取得します。

```
GET /api/videos/{video_id}/frames
```

**パラメータ**

- `video_id`: 動画ID（必須）
- `page`: ページ番号（オプション、デフォルト: 1）
- `per_page`: ページあたりの件数（オプション、デフォルト: 50）

**レスポンス**

```json
{
  "frames": [
    {
      "id": 1,
      "video_id": 1,
      "frame_number": 100,
      "timestamp": 3.33,
      "path": "/path/to/frame.jpg",
      "annotation_count": 14,
      "annotated": true
    }
  ],
  "total": 1000,
  "page": 1,
  "per_page": 50,
  "pages": 20
}
```

### データセット管理

#### データセット統計取得

データセットの統計情報を取得します。

```
GET /api/dataset/statistics
```

**レスポンス**

```json
{
  "video_count": 10,
  "frame_count": 10000,
  "tile_count": 140000,
  "total_size": "5.2 GB",
  "last_updated": "2024-12-20T10:00:00"
}
```

#### データセットバージョン一覧

データセットのバージョン一覧を取得します。

```
GET /api/dataset/versions
```

**レスポンス**

```json
[
  {
    "id": "v1.0",
    "version": "1.0",
    "created_at": "2024-12-20T10:00:00",
    "frame_count": 10000,
    "tile_count": 140000,
    "description": "初回バージョン"
  }
]
```

#### データセットバージョン作成

新しいデータセットバージョンを作成します。

```
POST /api/dataset/create_version
```

**リクエスト**

```json
{
  "version": "1.1",
  "description": "アノテーション追加版",
  "include_all_data": true
}
```

**レスポンス**

```json
{
  "message": "Dataset version created successfully",
  "version": {
    "id": "v1.1",
    "version": "1.1",
    "created_at": "2024-12-20T10:00:00"
  }
}
```

#### データセットエクスポート

データセットを指定形式でエクスポートします。

```
POST /api/dataset/export
```

**リクエスト**

```json
{
  "format": "yolo",
  "version_id": "v1.0",
  "output_dir": "/path/to/output"
}
```

**パラメータ**

- `format`: エクスポート形式（"yolo", "coco", "voc"）
- `version_id`: バージョンID（オプション、指定しない場合は最新）
- `output_dir`: 出力ディレクトリ（オプション）

**レスポンス**

```json
{
  "message": "Dataset exported successfully in yolo format",
  "export_path": "/path/to/export",
  "version_id": "v1.0"
}
```

### 学習管理

#### 学習セッション一覧

学習セッションの一覧を取得します。

```
GET /api/training/sessions
```

**レスポンス**

```json
[
  {
    "id": "session-1",
    "model_type": "detection",
    "status": "completed",
    "start_time": "2024-12-20T10:00:00",
    "end_time": "2024-12-20T12:00:00",
    "epochs": 100,
    "best_accuracy": 0.95
  }
]
```

#### 学習開始

新しい学習セッションを開始します。

```
POST /api/training/start
```

**リクエスト**

```json
{
  "model_type": "detection",
  "model_name": "yolov5",
  "dataset_version_id": "v1.0",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "validation_split": 0.2,
  "test_split": 0.1,
  "early_stopping_patience": 10,
  "save_best_only": true,
  "use_data_augmentation": true,
  "transfer_learning": false,
  "gpu_enabled": true,
  "num_workers": 4,
  "seed": 42
}
```

**レスポンス**

```json
{
  "success": true,
  "session_id": "uuid-string"
}
```

進捗はWebSocketで通知されます。

### ラベリング

#### 手牌領域設定取得

手牌領域の設定を取得します。

```
GET /api/labeling/hand_areas
```

**レスポンス**

```json
{
  "regions": {
    "bottom": {"x": 640, "y": 900, "w": 640, "h": 120},
    "right": {"x": 1100, "y": 400, "w": 120, "h": 400},
    "top": {"x": 640, "y": 60, "w": 640, "h": 120},
    "left": {"x": 60, "y": 400, "w": 120, "h": 400}
  },
  "frame_size": [1920, 1080]
}
```

#### 牌分割

手牌領域から個々の牌を分割します。

```
POST /api/labeling/split_tiles
```

**リクエスト**

```json
{
  "video_id": "video-1",
  "frame_number": 1000,
  "player": "bottom"
}
```

**レスポンス**

```json
{
  "player": "bottom",
  "frame_number": 1000,
  "tiles": [
    {
      "index": 0,
      "bbox": {"x": 10, "y": 10, "w": 40, "h": 60},
      "label": null,
      "confidence": null
    }
  ]
}
```

#### 自動ラベリング

AIを使用して牌を自動的にラベリングします。

```
POST /api/labeling/auto_label
```

**リクエスト**

```json
{
  "video_id": "video-1",
  "frame_number": 1000,
  "player": "bottom"
}
```

**レスポンス**

```json
{
  "player": "bottom",
  "frame_number": 1000,
  "tiles": [
    {
      "index": 0,
      "label": "1m",
      "confidence": 0.95
    }
  ]
}
```

## WebSocket イベント

### 接続

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected');
});
```

### セッション参加

```javascript
socket.emit('join_session', {
  session_id: 'uuid-string'
});
```

### 進捗通知

#### フレーム抽出進捗

```javascript
socket.on('frame_extraction_progress', (data) => {
  console.log(data);
  // {
  //   session_id: 'uuid-string',
  //   status: 'processing',
  //   progress: 50,
  //   message: '500/1000 フレーム処理済み'
  // }
});
```

#### 学習進捗

```javascript
socket.on('training_progress', (data) => {
  console.log(data);
  // {
  //   session_id: 'uuid-string',
  //   training_session_id: 'training-1',
  //   status: 'training',
  //   progress: {
  //     epoch: 50,
  //     total_epochs: 100,
  //     loss: 0.123,
  //     accuracy: 0.95
  //   },
  //   message: '学習進捗: 50/100 エポック'
  // }
});
```

## エラーハンドリング

### HTTPステータスコード

- `200`: 成功
- `400`: 不正なリクエスト
- `401`: 認証エラー（将来実装）
- `403`: アクセス拒否
- `404`: リソースが見つからない
- `413`: ペイロードが大きすぎる
- `429`: レート制限（将来実装）
- `500`: サーバーエラー

### エラーレスポンス形式

```json
{
  "error": "エラーメッセージ",
  "code": "ERROR_CODE",
  "details": {
    "field": "エラーの詳細情報"
  }
}
```

## セキュリティ

### CSRF保護

本番環境では、すべてのPOST/PUT/DELETEリクエストにCSRFトークンが必要です。

```javascript
// トークン取得
const csrfToken = SecurityUtils.getCsrfToken();

// リクエストヘッダーに追加
headers: {
  'X-CSRF-Token': csrfToken
}
```

### ファイルアップロード

- 最大ファイルサイズ: 2GB
- 許可されるMIMEタイプ:
  - video/mp4
  - video/x-msvideo
  - video/quicktime
  - video/x-matroska
  - video/webm

### レート制限

本番環境では以下のレート制限が適用されます：

- ファイルアップロード: 10回/分
- API呼び出し: 100回/分
- WebSocket接続: 10接続/IPアドレス

## 使用例

### cURLを使用した例

```bash
# 動画アップロード
curl -X POST http://localhost:5000/api/upload_video \
  -F "video=@/path/to/video.mp4" \
  -F "csrf_token=your-csrf-token"

# 動画一覧取得
curl http://localhost:5000/api/videos

# フレーム抽出開始
curl -X POST http://localhost:5000/api/extract_frames \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "config": {
      "interval_seconds": 1.0
    }
  }'
```

### JavaScriptを使用した例

```javascript
// 動画アップロード
async function uploadVideo(file) {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('csrf_token', SecurityUtils.getCsrfToken());

  const response = await fetch('/api/upload_video', {
    method: 'POST',
    body: formData
  });

  return response.json();
}

// 動画一覧取得
async function getVideos() {
  const response = await fetch('/api/videos');
  return response.json();
}

// WebSocket接続
const socket = io();

socket.on('connect', () => {
  console.log('WebSocket connected');
});

socket.on('frame_extraction_progress', (data) => {
  updateProgressBar(data.progress);
});
```

## 更新履歴

### v2.0.0 (2024-12-XX)
- 初回API仕様公開
- セキュリティ機能追加
- バッチ処理最適化対応

## 注意事項

- このAPIは開発中であり、仕様は変更される可能性があります
- 本番環境では適切な認証・認可機構を実装してください
- ファイルアップロードは適切なウイルススキャンを実装することを推奨します
