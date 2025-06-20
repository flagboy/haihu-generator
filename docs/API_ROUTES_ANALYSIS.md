# APIルート分析レポート

## 現状の問題点

### 1. 大規模なルートファイル
- `scene_routes.py`: 711行 - 対局画面ラベリング関連のすべてのエンドポイント
- `routes.py`: 377行 - 一般的なラベリング機能
- `training_routes.py`: 253行 - 訓練関連のエンドポイント

### 2. scene_routes.pyの問題点

#### 2.1 責務の混在
- セッション管理
- フレーム処理
- ラベリング操作
- 自動ラベリング
- 画像処理
- エラーハンドリング

#### 2.2 エンドポイント一覧（13個）
1. `POST /sessions/clear` - 全セッションクリア
2. `POST /sessions` - セッション作成/再開
3. `GET /sessions/<session_id>` - セッション情報取得
4. `GET /sessions/<session_id>/frame/<frame_number>` - フレーム取得
5. `POST /sessions/<session_id>/label` - ラベル付け
6. `POST /sessions/<session_id>/batch_label` - バッチラベル付け
7. `GET /sessions/<session_id>/next_unlabeled` - 次の未ラベルフレーム
8. `GET /sessions/<session_id>/uncertainty_frame` - 不確実性の高いフレーム
9. `POST /sessions/<session_id>/auto_label` - 自動ラベリング
10. `DELETE /sessions/<session_id>` - セッション削除
11. `GET /sessions/<session_id>/segments` - セグメント取得
12. `POST /sessions/<session_id>/close` - セッション終了
13. `GET /sessions` - セッション一覧

#### 2.3 コードの問題
- グローバル変数での状態管理（`_sessions`、`_classifier`）
- 長いメソッド（create_sessionは180行以上）
- エラーハンドリングの不統一
- ビジネスロジックとルートハンドラーの混在

## リファクタリング設計

### 新しいアーキテクチャ

```
src/training/game_scene/labeling/api/
├── routes/
│   ├── __init__.py          # Blueprintの集約
│   ├── session_routes.py     # セッション管理（CRUD）
│   ├── frame_routes.py       # フレーム関連操作
│   ├── labeling_routes.py    # ラベリング操作
│   └── auto_label_routes.py  # 自動ラベリング
├── middleware/
│   ├── __init__.py
│   ├── error_handler.py      # エラーハンドリング
│   ├── session_validator.py  # セッション検証
│   └── request_logger.py     # リクエストログ
├── services/
│   ├── __init__.py
│   ├── session_service.py    # セッション管理ロジック
│   ├── frame_service.py      # フレーム処理ロジック
│   ├── labeling_service.py   # ラベリングロジック
│   └── auto_label_service.py # 自動ラベリングロジック
└── schemas/
    ├── __init__.py
    ├── request_schemas.py    # リクエストバリデーション
    └── response_schemas.py   # レスポンス形式定義
```

### 責務の分離

1. **Routes層**
   - HTTPリクエスト/レスポンスの処理
   - バリデーション
   - サービス層の呼び出し

2. **Services層**
   - ビジネスロジック
   - データ処理
   - 外部サービスとの連携

3. **Middleware層**
   - 横断的関心事の処理
   - エラーハンドリング
   - ログ記録
   - 認証・認可

4. **Schemas層**
   - データバリデーション
   - シリアライゼーション
   - API仕様の明確化

### RESTful設計の適用

#### セッションリソース
- `GET /api/scene_labeling/sessions` - 一覧取得
- `POST /api/scene_labeling/sessions` - 作成
- `GET /api/scene_labeling/sessions/{id}` - 詳細取得
- `DELETE /api/scene_labeling/sessions/{id}` - 削除
- `PUT /api/scene_labeling/sessions/{id}/close` - 終了

#### フレームリソース
- `GET /api/scene_labeling/sessions/{id}/frames` - フレーム一覧
- `GET /api/scene_labeling/sessions/{id}/frames/{frame_number}` - フレーム取得
- `GET /api/scene_labeling/sessions/{id}/frames/next_unlabeled` - 次の未ラベル
- `GET /api/scene_labeling/sessions/{id}/frames/uncertain` - 不確実性の高いフレーム

#### ラベルリソース
- `POST /api/scene_labeling/sessions/{id}/labels` - ラベル作成
- `POST /api/scene_labeling/sessions/{id}/labels/batch` - バッチ作成
- `POST /api/scene_labeling/sessions/{id}/labels/auto` - 自動ラベリング

### 改善点

1. **保守性の向上**
   - ファイルサイズの削減（各ルートファイル100-200行程度）
   - 責務の明確化
   - テストの書きやすさ

2. **拡張性の向上**
   - 新しいエンドポイントの追加が容易
   - ミドルウェアの追加が簡単
   - サービスの再利用

3. **エラーハンドリングの統一**
   - 中央集権的なエラー処理
   - 一貫性のあるエラーレスポンス
   - ログの標準化

4. **パフォーマンスの向上**
   - セッション管理の改善
   - キャッシュの活用
   - 非同期処理の導入可能性
