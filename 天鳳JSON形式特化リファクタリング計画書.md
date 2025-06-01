# 天鳳JSON形式特化リファクタリング計画書

## 概要

麻雀牌譜作成システムを天鳳JSON形式専用に特化するためのリファクタリング計画書です。
参考リポジトリ: https://github.com/Apricot-S/majiang-log

## 目標

- 複数出力形式（MJSCORE、天鳳XML）のサポートを削除
- 天鳳JSON形式のみに特化してコードを簡素化
- 出力品質の向上と保守性の改善

## フェーズ1: 設定ファイルとメインアプリケーションの修正

### ステップ1: config.yaml の修正

#### 削除する設定項目
- 複数出力形式の選択肢
- 不要な形式変換設定
- MJSCORE/天鳳XML関連設定

#### 新しい設定構造

```yaml
# 麻雀牌譜作成システム設定ファイル（天鳳JSON形式専用）

# 動画処理設定
video:
  frame_extraction:
    fps: 1
    output_format: "jpg"
    quality: 95
  preprocessing:
    target_width: 1920
    target_height: 1080
    normalize: true
    denoise: true

# 画像処理設定
image:
  tile_detection:
    min_tile_size: 20
    max_tile_size: 200
    confidence_threshold: 0.5
  preprocessing:
    gaussian_blur_kernel: 3
    brightness_adjustment: 1.0
    contrast_adjustment: 1.0

# AI/ML設定
ai:
  detection:
    model_type: "yolo"
    model_path: "models/tile_detector.pt"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    input_size: [640, 640]
  classification:
    model_type: "cnn"
    model_path: "models/tile_classifier.pt"
    confidence_threshold: 0.8
    input_size: [224, 224]
    num_classes: 37
  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 100
    device: "auto"

# 天鳳JSON出力設定
output:
  format: "tenhou_json"  # 固定値
  file_extension: ".json"
  include_metadata: true
  validation_enabled: true
  
# 天鳳形式固有設定
tenhou:
  rule_type: "hanchan"  # 東南戦
  aka_dora_enabled: true
  include_timestamps: true
  player_names: ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"]

# ログ設定
logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
  file_path: "logs/mahjong_system.log"
  rotation: "1 day"
  retention: "30 days"

# ディレクトリ設定
directories:
  input: "data/input"
  output: "data/output"
  temp: "data/temp"
  models: "models"
  logs: "logs"

# システム設定
system:
  max_workers: 4
  memory_limit: "8GB"
  gpu_enabled: true
```

### ステップ2: main.py の修正

#### 修正対象メソッド

##### 1. `process_video` メソッドの簡素化

**変更前:**
```python
def process_video(self, video_path: str, output_path: Optional[str] = None,
                 format_type: str = "mjscore", enable_validation: bool = True) -> Dict[str, Any]:
```

**変更後:**
```python
def process_video(self, video_path: str, output_path: Optional[str] = None,
                 enable_validation: bool = True) -> Dict[str, Any]:
    """
    動画を処理して天鳳JSON形式の牌譜を生成
    
    Args:
        video_path: 入力動画パス
        output_path: 出力パス（Noneの場合は自動生成）
        enable_validation: 品質検証を有効にするか
        
    Returns:
        処理結果
    """
```

##### 2. `batch_process` メソッドの簡素化

**変更前:**
```python
def batch_process(self, input_directory: str, output_directory: str,
                 format_type: str = "mjscore", max_workers: int = None) -> Dict[str, Any]:
```

**変更後:**
```python
def batch_process(self, input_directory: str, output_directory: str,
                 max_workers: int = None) -> Dict[str, Any]:
    """
    バッチ処理（天鳳JSON形式固定）
    
    Args:
        input_directory: 入力ディレクトリ
        output_directory: 出力ディレクトリ
        max_workers: 最大並列数
        
    Returns:
        バッチ処理結果
    """
```

##### 3. `_generate_output_path` メソッドの修正

**変更前:**
```python
def _generate_output_path(self, video_path: str, format_type: str) -> str:
    """出力パスを生成"""
    video_name = Path(video_path).stem
    output_dir = self.config_manager.get_config()['directories']['output']
    
    if format_type.lower() == 'tenhou':
        extension = '.xml'
    else:
        extension = '.json'
    
    return os.path.join(output_dir, f"{video_name}_record{extension}")
```

**変更後:**
```python
def _generate_output_path(self, video_path: str) -> str:
    """出力パスを生成（天鳳JSON形式固定）"""
    video_name = Path(video_path).stem
    output_dir = self.config_manager.get_config()['directories']['output']
    
    # 天鳳JSON形式固定
    extension = '.json'
    
    return os.path.join(output_dir, f"{video_name}_tenhou{extension}")
```

#### コマンドライン引数の修正

##### process コマンド

**変更前:**
```python
process_parser.add_argument(
    '--format', '-f',
    choices=['mjscore', 'tenhou'],
    default='mjscore',
    help='出力形式 (default: mjscore)'
)
```

**変更後:**
```python
# format引数を削除（天鳳JSON形式固定）
```

##### batch コマンド

**変更前:**
```python
batch_parser.add_argument(
    '--format', '-f',
    choices=['mjscore', 'tenhou'],
    default='mjscore',
    help='出力形式 (default: mjscore)'
)
```

**変更後:**
```python
# format引数を削除（天鳳JSON形式固定）
```

#### メイン実行部分の修正

**変更前:**
```python
if args.command == 'process':
    result = app.process_video(
        video_path=args.video_path,
        output_path=args.output,
        format_type=args.format,
        enable_validation=not args.no_validation
    )
```

**変更後:**
```python
if args.command == 'process':
    result = app.process_video(
        video_path=args.video_path,
        output_path=args.output,
        enable_validation=not args.no_validation
    )
```

## フェーズ2: パイプライン層の修正

### ステップ3: SystemIntegrator の修正

#### 修正対象ファイル: `src/integration/system_integrator.py`

**修正内容:**
- `format_type` パラメータの削除
- 天鳳JSON形式固定の処理フロー

### ステップ4: GamePipeline の修正

#### 修正対象ファイル: `src/pipeline/game_pipeline.py`

**修正内容:**
- `export_game_record` メソッドの簡素化
- 天鳳JSON形式専用の出力処理

```python
def export_game_record(self) -> str:
    """天鳳JSON形式で牌譜を出力"""
    # 天鳳JSON形式固定の実装
    pass
```

## フェーズ3: 天鳳JSON形式専用モジュールの実装

### ステップ5: 天鳳JSON フォーマッターの作成

#### 新規作成ファイル: `src/output/tenhou_json_formatter.py`

```python
"""
天鳳JSON形式フォーマッター
参考: https://github.com/Apricot-S/majiang-log
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class TenhouGameData:
    """天鳳ゲームデータ"""
    title: List[str]
    name: List[str]
    rule: Dict[str, Any]
    log: List[List[Dict[str, Any]]]

class TenhouJsonFormatter:
    """天鳳JSON形式フォーマッター"""
    
    def __init__(self, config_manager):
        self.config = config_manager.get_config()
        self.tenhou_config = self.config.get('tenhou', {})
    
    def format_game_record(self, game_data: Any) -> str:
        """ゲームデータを天鳳JSON形式に変換"""
        tenhou_data = self._convert_to_tenhou_format(game_data)
        return json.dumps(tenhou_data, ensure_ascii=False, indent=2)
    
    def _convert_to_tenhou_format(self, game_data: Any) -> Dict[str, Any]:
        """内部データ形式から天鳳形式に変換"""
        return {
            "title": self._generate_title(game_data),
            "name": self._get_player_names(),
            "rule": self._get_rule_settings(),
            "log": self._convert_game_log(game_data)
        }
    
    def _generate_title(self, game_data: Any) -> List[str]:
        """タイトル情報を生成"""
        # 実装詳細
        pass
    
    def _get_player_names(self) -> List[str]:
        """プレイヤー名を取得"""
        return self.tenhou_config.get('player_names', 
                                     ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"])
    
    def _get_rule_settings(self) -> Dict[str, Any]:
        """ルール設定を取得"""
        return {
            "disp": self.tenhou_config.get('rule_type', '東南戦'),
            "aka": 1 if self.tenhou_config.get('aka_dora_enabled', True) else 0
        }
    
    def _convert_game_log(self, game_data: Any) -> List[List[Dict[str, Any]]]:
        """ゲームログを天鳳形式に変換"""
        # 実装詳細
        pass
```

### ステップ6: 天鳳形式データモデルの作成

#### 新規作成ファイル: `src/models/tenhou_game_data.py`

```python
"""
天鳳形式データモデル
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

class TenhouActionType(Enum):
    """天鳳形式アクションタイプ"""
    START_KYOKU = "start_kyoku"
    TSUMO = "tsumo"
    DAHAI = "dahai"
    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    REACH = "reach"
    HORA = "hora"
    RYUKYOKU = "ryukyoku"
    END_KYOKU = "end_kyoku"

@dataclass
class TenhouAction:
    """天鳳形式アクション"""
    type: TenhouActionType
    actor: Optional[int] = None
    pai: Optional[str] = None
    target: Optional[int] = None
    consumed: Optional[List[str]] = None
    tsumogiri: Optional[bool] = None

@dataclass
class TenhouKyoku:
    """天鳳形式局データ"""
    bakaze: str  # 場風
    dora_marker: str  # ドラ表示牌
    kyoku: int  # 局数
    honba: int  # 本場
    kyotaku: int  # 供託
    oya: int  # 親
    scores: List[int]  # スコア
    tehais: List[List[str]]  # 手牌
    actions: List[TenhouAction]  # アクション履歴

@dataclass
class TenhouGameRecord:
    """天鳳形式ゲーム記録"""
    title: List[str]
    name: List[str]
    rule: Dict[str, Any]
    log: List[List[Dict[str, Any]]]
```

## フェーズ4: 既存コードの修正

### ステップ7: 不要なコードの削除

#### 削除対象
- MJSCORE形式関連コード
- 天鳳XML形式関連コード
- 出力形式選択ロジック

#### 修正対象ファイル
- `src/pipeline/game_pipeline.py`
- `src/integration/system_integrator.py`
- テストファイル群

## フェーズ5: テスト修正

### ステップ8: テストケースの修正

#### 修正対象テストファイル
- `tests/test_game_pipeline.py`
- `tests/test_integration.py`
- `tests/test_config.py`

#### 新規作成テストファイル
- `tests/test_tenhou_json_formatter.py`
- `tests/test_tenhou_game_data.py`

## 実装スケジュール

### 第1週: 基盤修正
- Day 1-2: `config.yaml` の修正
- Day 3-4: `main.py` の修正
- Day 5: 基本動作確認

### 第2週: パイプライン修正
- Day 1-2: `SystemIntegrator` の修正
- Day 3-4: `GamePipeline` の修正
- Day 5: 統合テスト

### 第3週: 天鳳形式実装
- Day 1-3: 天鳳JSONフォーマッターの実装
- Day 4-5: データモデルの実装

### 第4週: 最終調整
- Day 1-3: 不要コードの削除
- Day 4-5: テスト修正と最終確認

## 期待される効果

1. **コードの簡素化**: 出力形式選択ロジックの削除により、保守性向上
2. **品質向上**: 単一形式に集中することで、出力精度の向上
3. **標準準拠**: 天鳳形式という業界標準への完全対応
4. **処理効率**: 不要な変換処理の削除による性能向上

## リスク要因と対策

### リスク
- 既存機能の破綻
- テストケースの大幅修正
- 設定ファイルの互換性問題

### 対策
- 段階的な実装とテスト
- バックアップの作成
- 詳細な動作確認

## 次のステップ

1. この計画書の承認
2. Codeモードでの実装開始
3. 段階的な実装とテスト
4. 最終的な品質確認

---

**作成日**: 2024年6月1日  
**作成者**: Architect Mode  
**バージョン**: 1.0