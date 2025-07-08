# 推測フレームレビュー機能

## 概要

カメラ切り替えなどで欠落したアクションを推測した際に、その場面のフレームを保存し、後で人間が確認・修正できる機能です。

## 機能

1. **フレーム自動保存**
   - 推測が発生した場面のフレーム画像を自動保存
   - 推測情報（アクションタイプ、牌、信頼度）を画像に描画
   - 前巡と現巡の手牌情報も記録

2. **Webインターフェース**
   - 保存されたフレームをブラウザで確認
   - 推測が正しいかどうかを検証
   - 間違っている場合は修正情報を入力

3. **修正情報の管理**
   - 人間による修正情報を保存
   - 検証済み/未検証の管理
   - 修正情報のエクスポート

## 使用方法

### 1. 推測機能を有効にして処理を実行

```python
from src.tracking.simplified_action_detector import SimplifiedActionDetector
import cv2
import numpy as np

# フレーム保存を有効にした設定
config = {
    "enable_inference": True,    # 推測機能を有効化
    "enable_frame_save": True    # フレーム保存を有効化
}

detector = SimplifiedActionDetector(config)

# フレームを処理（実際の使用では動画から取得）
frame = cv2.imread("game_frame.jpg")  # または動画から取得
hand_tiles = ["1m", "2m", "3m", ...]  # 検出された手牌

# アクション検出（フレームも渡す）
result = detector.detect_hand_change(
    hand_tiles,
    frame_number=100,
    frame=frame
)

# 処理終了後、フレーム管理器を取得
frame_manager = detector.get_frame_manager()
if frame_manager:
    # レビュー用HTMLを生成
    html_path = frame_manager.generate_review_html()
    print(f"レビューページ: {html_path}")

    # 統計情報を表示
    stats = frame_manager.get_statistics()
    print(f"推測フレーム数: {stats['total_frames']}")
    print(f"未検証: {stats['total_frames'] - stats['verified_frames']}")
```

### 2. Webインターフェースでレビュー

```python
from src.web.inference_review_app import InferenceReviewApp

# レビューアプリを起動
app = InferenceReviewApp("inference_frames")
app.run(host="localhost", port=5001)

# ブラウザで http://localhost:5001 にアクセス
```

### 3. レビュー画面の操作

- **セッション一覧**: 処理したセッションの一覧が表示されます
- **フレーム確認**: 各フレームの画像と推測情報を確認
- **修正入力**:
  - アクションタイプの修正
  - 牌の修正
  - コメントの追加
- **検証済みマーク**: 確認が完了したフレームに検証済みマークを付与

#### キーボードショートカット
- `←` / `→`: 前後のフレームへ移動
- `u`: 次の未検証フレームへジャンプ
- `v`: 現在のフレームを検証済みにする

### 4. 修正情報のエクスポート

```python
# セッションを読み込み
frame_manager.load_session("20250708_120000")

# 修正情報をエクスポート
corrections = frame_manager.export_corrections()

# 修正情報の例
{
    "frame_id_1": {
        "original": {
            "action_type": "draw",
            "tile": "5z",
            "confidence": 0.8
        },
        "correction": {
            "action_type": "draw",
            "tile": "6z",
            "comment": "実際は6zをツモ"
        },
        "frame_info": {
            "frame_number": 100,
            "turn_number": 5,
            "player_index": 0
        }
    }
}
```

## ディレクトリ構造

```
inference_frames/
├── 20250708_120000/           # セッションディレクトリ
│   ├── images/                # フレーム画像
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── index.json            # フレーム情報のインデックス
│   └── review.html           # レビュー用HTML
├── 20250708_130000/
└── ...
```

## 設定オプション

```python
config = {
    # 推測機能の設定
    "enable_inference": True,      # 推測機能の有効化
    "enable_frame_save": True,     # フレーム保存の有効化
    "similarity_threshold": 0.8,   # 手番切り替えの閾値

    # フレーム保存の設定（InferenceFrameManager）
    "base_dir": "inference_frames",  # 保存先ディレクトリ
}
```

## 活用例

1. **モデル改善のための教師データ収集**
   - 推測が発生した場面を収集
   - 人間が正解を入力
   - 修正データを使ってモデルを再学習

2. **システムの精度評価**
   - 推測の正解率を測定
   - どのような場面で推測が発生するか分析
   - カメラ切り替えパターンの把握

3. **デバッグ・改善**
   - 推測ロジックの問題点を発見
   - エッジケースの収集
   - アルゴリズムの改善

## 注意事項

- フレーム画像は容量を使用するため、定期的にクリーンアップを推奨
- `frame_manager.cleanup_old_sessions(days=7)` で古いセッションを削除可能
- 大量のフレームがある場合、レビューページの読み込みが遅くなる可能性があります
