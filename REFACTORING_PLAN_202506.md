# リファクタリング計画書 - コードアーキテクチャ改善

## 概要
このドキュメントは、麻雀牌譜作成システムのコードアーキテクチャを改善するためのリファクタリング計画を記載します。

## 現状分析

### 1. 識別された問題点

#### 1.1 大規模ファイル
- `model_trainer.py` (833行) - 責務が多すぎる
- `dataset_manager.py` (775行) - データ管理とビジネスロジックが混在
- `scene_routes.py` (711行) - APIエンドポイントが肥大化
- `scene_labeling_session.py` (701行) - セッション管理が複雑

#### 1.2 アーキテクチャの問題
- **責務の分離不足**: 単一のクラスが複数の責任を持っている
- **依存関係の複雑化**: コンポーネント間の結合度が高い
- **テストの困難性**: モックが困難な設計
- **重複コード**: 類似機能が複数箇所に散在

#### 1.3 コード品質の問題
- **長いメソッド**: 100行を超えるメソッドが存在
- **深いネスト**: 条件分岐の入れ子が深い
- **マジックナンバー**: 定数化されていない数値
- **エラーハンドリング**: 一貫性のない例外処理

## リファクタリング方針

### 1. SOLID原則の適用
- **単一責任の原則 (SRP)**: 各クラスは1つの責任のみを持つ
- **開放閉鎖の原則 (OCP)**: 拡張に対して開き、修正に対して閉じる
- **リスコフの置換原則 (LSP)**: 派生クラスは基底クラスと置換可能
- **インターフェース分離の原則 (ISP)**: 不要な依存を避ける
- **依存性逆転の原則 (DIP)**: 抽象に依存する

### 2. デザインパターンの活用
- **Strategy Pattern**: アルゴリズムの切り替え
- **Factory Pattern**: オブジェクト生成の抽象化
- **Observer Pattern**: イベント駆動アーキテクチャ
- **Repository Pattern**: データアクセスの抽象化

## 実施計画

### Phase 1: ModelTrainerのリファクタリング（優先度: 高）

#### 1.1 責務の分離
```python
# Before: model_trainer.py (833行)
class ModelTrainer:
    # データローディング、訓練、評価、保存すべてを担当

# After: 責務ごとに分割
training/
├── trainer.py          # 訓練の調整のみ
├── data_loader.py      # データローディング
├── metrics.py          # メトリクス計算
├── checkpoint.py       # モデル保存/読み込み
└── callbacks.py        # コールバック処理
```

#### 1.2 実装内容
1. **DataLoaderFactory**: データローダー作成の責務を分離
2. **MetricsCalculator**: メトリクス計算を独立
3. **CheckpointManager**: チェックポイント管理を分離
4. **TrainingCallback**: 訓練中のイベント処理を抽象化

### Phase 2: DatasetManagerのリファクタリング（優先度: 高）

#### 2.1 Repository Patternの導入
```python
# Before: dataset_manager.py
class DatasetManager:
    # SQLite操作、データ変換、バージョン管理すべてを担当

# After: レイヤー分離
data/
├── repositories/
│   ├── dataset_repository.py    # データアクセス層
│   └── version_repository.py    # バージョン管理
├── services/
│   ├── dataset_service.py       # ビジネスロジック
│   └── export_service.py        # エクスポート処理
└── models/
    └── dataset.py               # データモデル
```

### Phase 3: APIルートの整理（優先度: 中）

#### 3.1 ルートの分割とミドルウェアの活用
```python
# Before: scene_routes.py (711行)
@bp.route('/api/scene/...')
def massive_endpoint():
    # 巨大な処理

# After: 機能ごとに分割
api/
├── routes/
│   ├── session_routes.py     # セッション管理
│   ├── labeling_routes.py    # ラベリング操作
│   └── export_routes.py      # エクスポート
├── middleware/
│   ├── auth.py               # 認証
│   └── validation.py         # バリデーション
└── handlers/
    └── error_handler.py      # エラーハンドリング
```

### Phase 4: 共通機能の抽出（優先度: 中）

#### 4.1 ベースクラスとミックスインの作成
```python
# 共通機能を抽出
common/
├── base/
│   ├── base_trainer.py       # 訓練の基底クラス
│   ├── base_dataset.py       # データセットの基底クラス
│   └── base_service.py       # サービスの基底クラス
├── mixins/
│   ├── logging_mixin.py      # ロギング機能
│   ├── validation_mixin.py   # バリデーション機能
│   └── cache_mixin.py        # キャッシュ機能
└── decorators/
    ├── retry.py              # リトライデコレータ
    ├── cache.py              # キャッシュデコレータ
    └── validate.py           # バリデーションデコレータ
```

## 実施スケジュール

### Week 1: Phase 1 - ModelTrainerのリファクタリング
- Day 1-2: 設計とインターフェース定義
- Day 3-4: 実装とユニットテスト
- Day 5: 統合テストと修正

### Week 2: Phase 2 - DatasetManagerのリファクタリング
- Day 1-2: Repository層の実装
- Day 3-4: Service層の実装
- Day 5: 移行とテスト

### Week 3: Phase 3 & 4 - API整理と共通機能
- Day 1-2: APIルートの分割
- Day 3-4: 共通機能の抽出
- Day 5: ドキュメント更新

## 成功基準

1. **コード品質の向上**
   - 各ファイルが500行以下
   - メソッドが50行以下
   - 循環的複雑度が10以下

2. **テストカバレッジ**
   - ユニットテストカバレッジ 80%以上
   - 統合テストの追加

3. **パフォーマンス**
   - 既存機能の性能劣化なし
   - メモリ使用量の削減

4. **保守性**
   - ドキュメントの完備
   - 依存関係の明確化
   - 拡張性の向上

## リスクと対策

### リスク1: 既存機能への影響
- **対策**: 段階的な移行とfeature flagの使用

### リスク2: パフォーマンスの劣化
- **対策**: ベンチマークテストの実施

### リスク3: 互換性の問題
- **対策**: 廃止予定APIの段階的な移行

## 次のステップ

1. この計画のレビューと承認
2. Phase 1の詳細設計の作成
3. 実装開始

---

作成日: 2025年6月19日
作成者: Claude Code Assistant
