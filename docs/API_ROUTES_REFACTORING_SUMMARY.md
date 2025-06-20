# APIルートリファクタリング完了報告

## 概要

Phase 3として、`scene_routes.py`（711行）の大規模なAPIルートファイルを、責任を分離した複数のモジュールにリファクタリングしました。

## 実施内容

### 1. 問題分析
- 単一ファイルに13個のエンドポイントが混在
- グローバル変数によるセッション管理
- エラーハンドリングの不統一
- ビジネスロジックとHTTP処理の混在

### 2. 新しいアーキテクチャ

```
src/training/game_scene/labeling/api/
├── middleware/          # 横断的関心事
│   ├── __init__.py
│   ├── error_handler.py    # 統一エラーハンドリング
│   ├── request_logger.py   # リクエストログ
│   └── session_validator.py # セッション検証
├── schemas/            # データ検証
│   ├── __init__.py
│   ├── request_schemas.py  # リクエストスキーマ
│   └── response_schemas.py # レスポンススキーマ
├── services/           # ビジネスロジック
│   ├── __init__.py
│   ├── session_service.py  # セッション管理
│   ├── frame_service.py    # フレーム処理
│   ├── labeling_service.py # ラベリング
│   └── auto_label_service.py # 自動ラベリング
├── routes/             # HTTPエンドポイント
│   ├── __init__.py
│   ├── session_routes.py   # セッション関連
│   ├── frame_routes.py     # フレーム関連
│   ├── labeling_routes.py  # ラベリング関連
│   └── auto_label_routes.py # 自動ラベリング
└── scene_routes.py     # レガシー互換性
```

### 3. 主な改善点

#### エラーハンドリングの統一
```python
# カスタムエラークラス
class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400, payload: dict = None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

# 統一されたエラーレスポンス
{
    "error": {
        "message": "エラーメッセージ",
        "code": "ERROR_CODE",
        "details": {}
    }
}
```

#### RESTful設計の適用
- `GET /sessions` - セッション一覧
- `POST /sessions` - セッション作成
- `GET /sessions/{id}` - セッション詳細
- `DELETE /sessions/{id}` - セッション削除
- `GET /sessions/{id}/frames/{frame_number}` - フレーム取得
- `POST /sessions/{id}/label` - ラベル付け
- `POST /sessions/{id}/auto_label` - 自動ラベリング

#### ミドルウェアパターン
```python
@error_handler          # エラーハンドリング
@request_logger        # リクエストログ
@validate_session()    # セッション検証
def endpoint_handler():
    # 実際の処理
```

### 4. 後方互換性

既存のコードとの互換性を保つため、`scene_routes.py`に以下を実装：

```python
def setup_scene_labeling_api(app: Flask, classifier=None, use_legacy=False):
    """
    対局画面ラベリングAPIをセットアップ

    Args:
        use_legacy: True の場合は旧APIを使用（非推奨）
    """
    if use_legacy:
        # レガシーAPI（非推奨警告付き）
    else:
        # 新しいリファクタリングされたAPI
```

## テスト結果

- 単体テスト: 全て成功
- 統合テスト: 全て成功
- 後方互換性: 維持

## 次のステップ

1. 既存コードの段階的移行
2. パフォーマンステストの実施
3. APIドキュメントの自動生成（OpenAPI/Swagger）
4. レート制限の実装
5. 認証・認可の追加

## 移行ガイド

### 既存コードの更新

```python
# 旧コード
from src.training.game_scene.labeling.api.scene_routes import scene_labeling_bp
app.register_blueprint(scene_labeling_bp)

# 新コード
from src.training.game_scene.labeling.api.scene_routes import setup_scene_labeling_api
setup_scene_labeling_api(app, classifier=classifier)
```

### エラーハンドリング

```python
# 旧コード
if error:
    return jsonify({"error": error}), 400

# 新コード
from ..middleware.error_handler import ValidationError
raise ValidationError("エラーメッセージ", {"field": "value"})
```

## まとめ

このリファクタリングにより、コードの保守性、テスタビリティ、拡張性が大幅に向上しました。RESTful設計の採用により、APIの一貫性も改善されています。
