# 対局動画からの教師データ作成戦略（ゼロベース版）

## 概要

本ドキュメントは、麻雀牌譜作成システムにおける牌認識AIを、外部データセットに依存せず、実際の対局動画のみから構築するための戦略と実装計画を定めたものです。

## 基本方針

### 1. ブートストラップアプローチ
- **初期段階**: 少数の高品質な手動ラベリングから開始
- **成長段階**: 半自動ラベリングによる効率的なデータ拡充
- **成熟段階**: アクティブラーニングによる継続的改善

### 2. 実環境特化型
- 実際に処理する対局動画と同じ環境のデータのみを使用
- ターゲット環境に最適化された高精度モデルの構築
- 外部データセットによるノイズを排除

### 3. 効率最優先
- 最小限のラベリング作業で最大の効果を追求
- スマートなサンプル選択による作業量削減
- 段階的な難易度設定による着実な精度向上

## 実装優先順位

### フェーズ1: 最小限データでの初期モデル構築（2週間）

#### 1.1 効率的な初期データ作成
**現状**: 学習済みモデルが存在しない
**対応**:
```python
# 初期データ収集戦略
class InitialDataCollector:
    def __init__(self):
        self.target_tiles_per_class = 30  # 各牌種30枚（計1,110枚）
        self.quality_threshold = 0.9

    def collect_clear_samples(self, video_paths):
        """明瞭なサンプルのみを収集"""
        samples = []
        for video in video_paths:
            # 1. 高品質フレームの抽出（手牌が綺麗に見える場面）
            frames = self.extract_high_quality_frames(video)

            # 2. 静止場面の優先（ブレなし）
            static_frames = self.filter_static_scenes(frames)

            # 3. 良好な照明条件のフレームを選択
            well_lit_frames = self.filter_by_lighting(static_frames)

            samples.extend(well_lit_frames)

        return samples
```

**スマートラベリング戦略**:
```python
# 効率的なラベリング順序
labeling_order = [
    # ステップ1: 識別しやすい字牌から開始（形状が特徴的）
    ["1z", "2z", "3z", "4z", "5z", "6z", "7z"],  # 東南西北白發中

    # ステップ2: 特徴的な数牌
    ["1m", "9m", "1p", "9p", "1s", "9s"],  # 端牌

    # ステップ3: 中間の数牌
    ["2m", "3m", "4m", "5m", "6m", "7m", "8m"],
    ["2p", "3p", "4p", "5p", "6p", "7p", "8p"],
    ["2s", "3s", "4s", "5s", "6s", "7s", "8s"],

    # ステップ4: 赤ドラと裏面
    ["0m", "0p", "0s", "back"]
]
```

**期待される成果**:
- 2週間で約1,200枚の高品質ラベリングデータ
- 各牌種の基本的な特徴を学習したモデル
- 限定的ながら半自動ラベリングの開始が可能（精度40-50%）

#### 1.2 データ拡張による少数データの最大活用
**現状**: 初期データが少ない
**対応**:
```python
# 積極的なデータ拡張戦略
class AggressiveAugmentation:
    def __init__(self):
        self.augmentation_factor = 20  # 1枚から20枚生成

    def create_pipeline(self):
        return A.Compose([
            # 基本的な変換
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),

            # 照明条件のシミュレーション
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.7
            ),

            # 撮影条件のシミュレーション
            A.OpticalDistortion(distort_limit=0.1, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),

            # ノイズとブラー
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ])
```

**期待される成果**:
- 1,200枚の実データから24,000枚の訓練データを生成
- 様々な環境条件に対する頑健性の向上

### フェーズ2: 半自動ラベリングによる規模拡大（1ヶ月）

#### 2.1 初期モデルを活用した効率的データ収集
**現状**: 基本的な認識が可能な初期モデルが完成
**対応**:
```python
# 半自動ラベリング戦略
class SemiAutoLabelingStrategy:
    def __init__(self, initial_model):
        self.model = initial_model
        self.confidence_threshold = 0.7
        self.human_review_threshold = 0.9

    def process_batch(self, frames):
        results = []
        for frame in frames:
            predictions = self.model.predict(frame)

            # 高信頼度の予測は自動承認
            auto_approved = [
                pred for pred in predictions
                if pred.confidence > self.human_review_threshold
            ]

            # 中程度の信頼度は人間がレビュー
            needs_review = [
                pred for pred in predictions
                if self.confidence_threshold < pred.confidence <= self.human_review_threshold
            ]

            # 低信頼度は手動ラベリング
            manual_required = [
                area for area in frame.tile_areas
                if not any(pred.overlaps(area) for pred in predictions)
            ]

            results.append({
                'auto_approved': auto_approved,
                'needs_review': needs_review,
                'manual_required': manual_required
            })

        return results
```

#### 2.2 難易度別データ収集
```python
# 段階的な難易度向上
difficulty_stages = [
    {
        'name': 'clear_tiles',
        'description': '明瞭な牌',
        'target_count': 5000,
        'criteria': {
            'min_size': 100,  # ピクセル
            'min_sharpness': 0.8,
            'max_occlusion': 0.0
        }
    },
    {
        'name': 'small_tiles',
        'description': '小さい牌（遠景）',
        'target_count': 3000,
        'criteria': {
            'min_size': 50,
            'max_size': 100,
            'min_sharpness': 0.6
        }
    },
    {
        'name': 'partially_occluded',
        'description': '部分的に隠れた牌',
        'target_count': 2000,
        'criteria': {
            'min_occlusion': 0.1,
            'max_occlusion': 0.3
        }
    }
]
```

**期待される成果**:
- 10,000枚の多様なラベリングデータ
- 精度70-80%の実用的なモデル
- ラベリング効率3倍向上

### フェーズ3: 実戦環境への対応と特殊ケース（1ヶ月）

#### 3.1 赤ドラと裏面牌の集中学習
**現状**: 基本的な牌は認識可能だが、特殊牌が未対応
**対応**:
```python
# 特殊牌の効率的収集
class SpecialTileCollector:
    def __init__(self, base_model):
        self.model = base_model
        self.special_tiles = {
            'red_dora': ['0m', '0p', '0s'],
            'back': ['back']
        }

    def find_red_dora_candidates(self, frame):
        """赤ドラ候補の自動検出"""
        # 1. まず5の牌を検出
        five_tiles = self.model.predict_tiles(frame, classes=['5m', '5p', '5s'])

        candidates = []
        for tile in five_tiles:
            # 2. 色分析で赤ドラ候補を抽出
            tile_img = frame.crop(tile.bbox)
            if self._has_red_component(tile_img):
                candidates.append({
                    'bbox': tile.bbox,
                    'base_class': tile.class_name,
                    'suggested_class': tile.class_name.replace('5', '0')
                })

        return candidates

    def find_back_tiles(self, frame, game_context):
        """ゲーム文脈から裏面牌を推定"""
        # 暗槓・暗刻の可能性がある領域を特定
        player_hands = game_context.get_player_hand_regions()

        candidates = []
        for region in player_hands:
            # 規則的に並んだ同じ見た目の領域を検出
            uniform_regions = self._find_uniform_regions(region)
            candidates.extend(uniform_regions)

        return candidates
```

#### 3.2 遮蔽とノイズへの対応
```python
# 実戦データの問題に対する対策
class RealWorldDataHandler:
    def __init__(self):
        self.occlusion_handler = OcclusionHandler()
        self.noise_handler = NoiseHandler()

    def create_occlusion_training_data(self, clean_tiles):
        """クリーンなデータから遮蔽データを人工生成"""
        occluded_data = []

        for tile in clean_tiles:
            # 1. 手による遮蔽をシミュレート
            hand_occluded = self._simulate_hand_occlusion(tile)

            # 2. 他の牌による遮蔽をシミュレート
            tile_occluded = self._simulate_tile_overlap(tile)

            # 3. 影による部分的な暗さをシミュレート
            shadow_affected = self._simulate_shadows(tile)

            occluded_data.extend([
                hand_occluded,
                tile_occluded,
                shadow_affected
            ])

        return occluded_data

    def enhance_low_quality_frames(self, frames):
        """低品質フレームの前処理強化"""
        enhanced = []
        for frame in frames:
            # 1. デノイジング
            denoised = cv2.fastNlMeansDenoisingColored(frame)

            # 2. コントラスト強調
            enhanced_contrast = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8,8)
            ).apply(denoised)

            # 3. シャープネス向上
            sharpened = self._unsharp_mask(enhanced_contrast)

            enhanced.append(sharpened)

        return enhanced
```

**期待される成果**:
- 全38クラス（赤ドラ・裏面含む）の認識対応
- 遮蔽牌の認識精度60%以上
- 実戦環境での総合精度85%

### フェーズ4: アクティブラーニングによる継続改善（継続的）

#### 4.1 アクティブラーニングの実装
```python
class SmartActiveLearning:
    def __init__(self, model):
        self.model = model
        self.learning_history = []

    def select_valuable_samples(self, video_frames, budget=100):
        """学習価値の高いサンプルを選択"""

        # 1. 不確実性サンプリング
        uncertainty_samples = self._uncertainty_sampling(video_frames, n=budget//3)

        # 2. 多様性サンプリング
        diversity_samples = self._diversity_sampling(video_frames, n=budget//3)

        # 3. エラー予測サンプリング
        error_prone_samples = self._error_prediction_sampling(video_frames, n=budget//3)

        return {
            'uncertainty': uncertainty_samples,
            'diversity': diversity_samples,
            'error_prone': error_prone_samples
        }

    def _uncertainty_sampling(self, frames, n):
        """予測が不確実なサンプルを選択"""
        predictions = self.model.predict_with_probability(frames)

        # エントロピーが高い（確信度が低い）サンプルを選択
        entropy_scores = self._calculate_entropy(predictions)
        uncertain_indices = np.argsort(entropy_scores)[-n:]

        return frames[uncertain_indices]

    def _diversity_sampling(self, frames, n):
        """既存データと異なる特徴を持つサンプルを選択"""
        features = self.model.extract_features(frames)

        # K-means++アルゴリズムで多様なサンプルを選択
        selected_indices = self._kmeans_plus_plus_selection(features, n)

        return frames[selected_indices]

    def _error_prediction_sampling(self, frames, n):
        """エラーを起こしやすいと予測されるサンプルを選択"""
        # 過去のエラーパターンから学習
        error_predictor = self._train_error_predictor()
        error_scores = error_predictor.predict(frames)

        high_error_indices = np.argsort(error_scores)[-n:]

        return frames[high_error_indices]

    def update_learning_history(self, selected_samples, labels, performance):
        """学習履歴を更新して次回の選択を改善"""
        self.learning_history.append({
            'samples': selected_samples,
            'labels': labels,
            'performance_gain': performance,
            'timestamp': datetime.now()
        })
```

#### 4.2 品質管理システム
```python
class AnnotationQualityChecker:
    def __init__(self):
        self.checks = [
            self._check_bbox_validity,
            self._check_class_consistency,
            self._check_temporal_consistency,
            self._check_annotator_agreement
        ]

    def validate_annotations(self, annotations):
        issues = []
        for check in self.checks:
            issues.extend(check(annotations))

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "quality_score": self._calculate_quality_score(issues)
        }
```

## 実装スケジュール

### 第1-2週: 初期ブートストラップ
- [ ] 高品質フレーム抽出ツールの実装
- [ ] 字牌と端牌の集中ラベリング（500枚）
- [ ] 基礎的なデータ拡張パイプライン構築
- [ ] 初期モデルの訓練（精度目標: 40%）

### 第3-4週: 基礎モデル完成
- [ ] 全牌種の基本データ収集（1,200枚）
- [ ] 積極的データ拡張の実装（20倍拡張）
- [ ] 初期モデルの改善（精度目標: 50%）
- [ ] 半自動ラベリングツールのテスト

### 第2ヶ月: 規模拡大
- [ ] 半自動ラベリングによるデータ収集（10,000枚）
- [ ] 難易度別データ収集の実施
- [ ] モデルの継続的改善（精度目標: 75%）
- [ ] 実戦環境でのテスト開始

### 第3ヶ月: 特殊ケース対応
- [ ] 赤ドラ・裏面牌の集中学習
- [ ] 遮蔽データの人工生成と学習
- [ ] 低品質フレームへの対応強化
- [ ] 総合精度85%達成

### 第4ヶ月以降: 継続的改善
- [ ] アクティブラーニングの本格運用
- [ ] 月次でのモデル更新サイクル確立
- [ ] 新規配信環境への適応
- [ ] 精度90%以上を維持

## 成功指標

### 技術指標
- **牌検出精度**: 95%以上（mAP@0.5）
- **牌分類精度**: 98%以上（Top-1 Accuracy）
- **赤ドラ識別精度**: 95%以上
- **遮蔽牌の検出率**: 90%以上

### 運用指標
- **ラベリング効率**: 1時間あたり500枚以上
- **データ品質スコア**: 平均85点以上
- **モデル更新頻度**: 月1回以上
- **処理速度**: 30FPS以上（GPU使用時）

## 初期データ作成の具体的手順

### 1. 最初の100枚（3日間）
```python
# 最初の100枚のラベリング計画
first_100_plan = {
    'day1': {
        'tiles': ['1z', '2z', '3z', '4z'],  # 東南西北
        'count': 10_per_tile,
        'source': '明瞭な手牌シーン',
        'time': '2-3時間'
    },
    'day2': {
        'tiles': ['5z', '6z', '7z'],  # 白發中
        'count': 10_per_tile,
        'source': '鳴き牌も含む',
        'time': '2時間'
    },
    'day3': {
        'tiles': ['1m', '9m', '1p', '9p', '1s', '9s'],  # 端牌
        'count': 5_per_tile,
        'source': '様々な角度',
        'time': '2時間'
    }
}
```

### 2. モデル評価基準
```python
# 段階的な成功基準
evaluation_milestones = [
    {
        'data_count': 100,
        'expected_accuracy': 0.3,
        'focus': '字牌の識別'
    },
    {
        'data_count': 500,
        'expected_accuracy': 0.5,
        'focus': '基本的な牌の識別'
    },
    {
        'data_count': 1200,
        'expected_accuracy': 0.6,
        'focus': '全牌種の基礎認識'
    },
    {
        'data_count': 10000,
        'expected_accuracy': 0.8,
        'focus': '実用レベルの認識'
    }
]
```

## リスクと対策

### リスク1: 初期の低精度による作業効率低下
**対策**:
- 最初は完全手動でも品質重視
- 識別しやすい牌から段階的に
- データ拡張で見かけ上のデータ量を増加

### リスク2: ラベリング作業の負担
**対策**:
- 1日2-3時間の持続可能なペース
- 明瞭なサンプルから開始
- 早期の半自動化移行

### リスク3: 特殊牌（赤ドラ・裏面）の不足
**対策**:
- 意図的なシーン選択
- 色分析による自動候補抽出
- 人工的なデータ生成

## ゼロベースアプローチの利点

外部データセットに依存しない本アプローチには以下の利点があります：

### 1. **完全な環境適応**
- 実際に処理する動画と同じ条件のデータのみで学習
- 配信環境特有の問題（照明、画質、アングル）に最初から対応
- 無駄な汎化性能を追求せず、ターゲット環境に特化

### 2. **品質管理の徹底**
- すべてのデータを自前で作成するため品質を完全にコントロール
- ラベリングの一貫性を保証
- 問題のあるデータの即座の修正が可能

### 3. **段階的な成長**
- 小さく始めて着実に成長
- 各段階での問題を確実に解決してから次へ
- 継続的な改善サイクルの確立

## まとめ

本戦略は、外部データセットに頼らず、実際の対局動画のみから高精度な牌認識システムを構築する現実的なアプローチです。

**成功の鍵**：
1. **初期の集中的な手動ラベリング**（最初の2週間が勝負）
2. **スマートなデータ選択**（質の高いサンプルから開始）
3. **積極的なデータ拡張**（少ないデータを最大活用）
4. **早期の半自動化移行**（効率を段階的に向上）

最初は労力がかかりますが、2-3ヶ月で実用レベルの精度（85%以上）に到達可能です。その後はアクティブラーニングによる継続的な改善により、最終的に90%以上の精度を達成できます。
