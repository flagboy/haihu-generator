# 初期モデル構築のための機能実装計画

## 概要

本ドキュメントは、2週間で初期モデルを完成させるために必要な機能の実装計画を定めたものです。特に、現在不足している「データ拡張機能」「ラベリング効率化機能」「YOLOv8対応」の3つの重要機能について、具体的な実装手順とスケジュールを示します。

## 実装優先順位と日程

### 全体スケジュール（7日間の開発 + 7日間のデータ作成）

```
Week 1: 機能開発フェーズ
- Day 1-3: データ拡張機能の実装
- Day 4-5: ラベリング効率化機能の実装  
- Day 6-7: YOLOv8対応とテスト

Week 2: データ作成フェーズ
- Day 8-10: 初期500枚の手動ラベリング
- Day 11-12: データ拡張による20倍増幅
- Day 13-14: 初期モデル訓練と評価
```

## Day 1-3: データ拡張機能の実装 ✅ 完了

### 1.1 Albumentations統合（Day 1）✅ 実装済み

#### 実装ファイル: `src/training/augmentation/advanced_augmentor.py`

**実装内容:**
- ✅ Albumentationsライブラリの統合完了
- ✅ 20種類以上の多様な変換パイプライン実装
- ✅ バウンディングボックスの自動変換対応
- ✅ クラスバランスを考慮したデータセット生成機能

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import List, Dict, Any

class AdvancedAugmentor:
    """20倍以上のデータ拡張を実現する高度な拡張クラス"""

    def __init__(self, augmentation_factor: int = 20):
        self.augmentation_factor = augmentation_factor
        self.pipelines = self._create_augmentation_pipelines()

    def _create_augmentation_pipelines(self) -> List[A.Compose]:
        """多様な拡張パイプラインを作成"""

        # 基本的な幾何学的変換
        geometric_light = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.7
            ),
        ])

        # 中程度の幾何学的変換
        geometric_medium = A.Compose([
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.5
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.5
            ),
            A.OpticalDistortion(
                distort_limit=0.5,
                shift_limit=0.5,
                p=0.5
            ),
        ])

        # 透視変換（カメラアングルのシミュレーション）
        perspective = A.Compose([
            A.Perspective(
                scale=(0.05, 0.1),
                keep_size=True,
                p=0.7
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-10, 10),
                p=0.7
            ),
        ])

        # 照明条件のシミュレーション
        lighting = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.7
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.7
            ),
        ])

        # 影のシミュレーション
        shadows = A.Compose([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=0.5
            ),
            A.RandomToneCurve(
                scale=0.1,
                p=0.3
            ),
        ])

        # ノイズとブラー（実環境のシミュレーション）
        noise_blur = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    per_channel=True,
                    p=1
                ),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1),
                A.MedianBlur(blur_limit=5, p=1),
                A.GaussianBlur(blur_limit=(3, 7), p=1),
                A.DefocusBlur(radius=(1, 3), alias_blur=0.1, p=1),
            ], p=0.5),
        ])

        # 画質劣化のシミュレーション
        quality_degradation = A.Compose([
            A.Downscale(
                scale_min=0.5,
                scale_max=0.9,
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=60,
                quality_upper=95,
                compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                p=0.5
            ),
        ])

        # 天候条件のシミュレーション（オプション）
        weather = A.Compose([
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3,
                    brightness_coefficient=0.7,
                    p=1
                ),
                A.RandomFog(
                    fog_coef_lower=0.3,
                    fog_coef_upper=0.5,
                    alpha_coef=0.08,
                    p=1
                ),
            ], p=0.1),
        ])

        # パイプラインの組み合わせ
        return [
            # 軽度の変換（高頻度）
            A.Compose([geometric_light, lighting, noise_blur]),
            # 中程度の変換
            A.Compose([geometric_medium, lighting, shadows]),
            # 強度の変換
            A.Compose([perspective, lighting, quality_degradation]),
            # 特殊条件
            A.Compose([geometric_light, weather, noise_blur]),
        ]

    def augment_single_image(self, image: np.ndarray, bbox: List[float],
                            class_id: int) -> List[Dict[str, Any]]:
        """1枚の画像から複数の拡張画像を生成"""
        augmented_samples = []

        # 各パイプラインで複数回拡張
        samples_per_pipeline = self.augmentation_factor // len(self.pipelines)

        for pipeline in self.pipelines:
            for i in range(samples_per_pipeline):
                # Albumentationsのbbox形式に変換
                transformed = pipeline(
                    image=image,
                    bboxes=[bbox],
                    class_labels=[class_id]
                )

                augmented_samples.append({
                    'image': transformed['image'],
                    'bbox': transformed['bboxes'][0] if transformed['bboxes'] else bbox,
                    'class_id': class_id,
                    'augmentation_info': {
                        'pipeline_idx': self.pipelines.index(pipeline),
                        'iteration': i
                    }
                })

        return augmented_samples

    def create_balanced_dataset(self, original_data: Dict[str, List],
                              target_per_class: int = 1000) -> Dict[str, List]:
        """クラスバランスを考慮したデータセット作成"""
        balanced_data = {}

        for class_name, samples in original_data.items():
            if len(samples) >= target_per_class:
                # 十分なデータがある場合はサンプリング
                balanced_data[class_name] = np.random.choice(
                    samples, target_per_class, replace=False
                ).tolist()
            else:
                # データが不足している場合は拡張で補完
                augmentation_needed = target_per_class // len(samples) + 1
                augmented_samples = []

                for sample in samples:
                    augmented = self.augment_single_image(
                        sample['image'],
                        sample['bbox'],
                        sample['class_id']
                    )
                    augmented_samples.extend(augmented[:augmentation_needed])

                balanced_data[class_name] = augmented_samples[:target_per_class]

        return balanced_data
```

### 1.2 赤ドラ検出用の色分析拡張（Day 2）✅ 実装済み

#### 実装ファイル: `src/training/augmentation/color_augmentor.py`

**実装内容:**
- ✅ 赤ドラ生成のための4つの手法実装（オーバーレイ、色置換、グラデーション、混合）
- ✅ 色統計に基づく赤ドラ判定機能
- ✅ 通常の5と赤5の訓練ペア生成機能
- ✅ 多様な赤色バリエーション生成パイプライン

```python
class RedDoraAugmentor:
    """赤ドラ検出のための特殊な色拡張"""

    def __init__(self):
        self.red_enhancement_pipeline = self._create_red_enhancement_pipeline()

    def _create_red_enhancement_pipeline(self) -> A.Compose:
        """赤色を強調する拡張パイプライン"""
        return A.Compose([
            # 赤色チャンネルの強調
            A.ChannelShuffle(p=0.3),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.7
            ),
            # 赤色の彩度を上げる
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=50,
                val_shift_limit=20,
                p=0.8
            ),
            # 赤色領域のコントラスト強調
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(4, 4),
                p=0.5
            ),
        ])

    def create_red_dora_variations(self, base_five_tile: np.ndarray,
                                  n_variations: int = 50) -> List[np.ndarray]:
        """通常の5の牌から赤ドラのバリエーションを生成"""
        variations = []

        for i in range(n_variations):
            # 赤色オーバーレイの追加
            red_overlay = np.zeros_like(base_five_tile)
            red_overlay[:, :, 2] = np.random.randint(100, 200)  # Red channel

            # アルファブレンディング
            alpha = np.random.uniform(0.3, 0.7)
            red_tinted = cv2.addWeighted(base_five_tile, 1-alpha, red_overlay, alpha, 0)

            # 追加の色拡張
            augmented = self.red_enhancement_pipeline(image=red_tinted)['image']
            variations.append(augmented)

        return variations
```

### 1.3 データ拡張の統合とテスト（Day 3）✅ 実装済み

#### 実装ファイル: `src/training/augmentation/unified_augmentor.py`

**実装内容:**
- ✅ すべての拡張機能を統合したUnifiedAugmentorクラス
- ✅ データセット全体の一括処理機能
- ✅ 訓練/検証データの自動分割
- ✅ YOLO形式でのデータ出力
- ✅ 詳細なレポート生成（JSON/Markdown）
- ✅ 包括的なテストコード（tests/test_augmentation.py）

```python
from .advanced_augmentor import AdvancedAugmentor
from .color_augmentor import RedDoraAugmentor

class UnifiedAugmentor:
    """すべての拡張機能を統合したクラス"""

    def __init__(self, config: Dict[str, Any]):
        self.advanced = AdvancedAugmentor(
            augmentation_factor=config.get('augmentation_factor', 20)
        )
        self.red_dora = RedDoraAugmentor()
        self.config = config

    def augment_dataset(self, dataset_path: str, output_path: str):
        """データセット全体の拡張"""
        # 実装詳細...
```

## Day 4-5: ラベリング効率化機能の実装 ✅ 完了

### 2.1 キーボードショートカットの拡充（Day 4）✅ 実装済み

#### 実装ファイル: `web_interface/static/js/enhanced_shortcuts.js`

**実装内容:**
- ✅ 30種類以上の包括的なキーボードショートカット
- ✅ ビジュアルフィードバック機能
- ✅ ヘルプオーバーレイ（Hキーで表示）
- ✅ クイックラベリングモードの実装

```javascript
class EnhancedShortcutManager {
    constructor(labelingInterface) {
        this.interface = labelingInterface;
        this.shortcuts = this.defineShortcuts();
        this.setupEventListeners();
    }

    defineShortcuts() {
        return {
            // ナビゲーション
            'ArrowRight': () => this.interface.nextFrame(),
            'ArrowLeft': () => this.interface.previousFrame(),
            'Space': () => this.interface.togglePlayPause(),

            // ラベリング操作
            'Enter': () => this.interface.confirmCurrentBox(),
            'Escape': () => this.interface.cancelCurrentBox(),
            'Delete': () => this.interface.deleteSelectedBox(),
            'Ctrl+C': () => this.interface.copySelectedBox(),
            'Ctrl+V': () => this.interface.pasteBox(),

            // 牌の種類選択（数字キー）
            '1': () => this.interface.selectTileType('manzu'),
            '2': () => this.interface.selectTileType('pinzu'),
            '3': () => this.interface.selectTileType('souzu'),
            '4': () => this.interface.selectTileType('jihai'),

            // 牌の番号選択（テンキー）
            'Numpad1': () => this.interface.selectTileNumber(1),
            'Numpad2': () => this.interface.selectTileNumber(2),
            // ... Numpad3-9

            // 特殊牌
            'R': () => this.interface.toggleRedDora(),
            'B': () => this.interface.selectBackTile(),

            // 表示制御
            'G': () => this.interface.toggleGrid(),
            'L': () => this.interface.toggleLabels(),
            'H': () => this.interface.toggleHelp(),

            // バッチ操作
            'Ctrl+A': () => this.interface.selectAllBoxes(),
            'Shift+Click': () => this.interface.multiSelect(),
            'Alt+C': () => this.interface.copyPreviousFrame(),

            // クイックアクション
            'Q': () => this.interface.quickLabelMode(),
            'W': () => this.interface.switchToNextUnlabeled(),
            'S': () => this.interface.saveProgress(),
        };
    }

    setupEventListeners() {
        document.addEventListener('keydown', (e) => {
            const key = this.getKeyCombo(e);
            const action = this.shortcuts[key];

            if (action && !this.interface.isInputFocused()) {
                e.preventDefault();
                action();
                this.showShortcutFeedback(key);
            }
        });
    }

    getKeyCombo(event) {
        let combo = '';
        if (event.ctrlKey) combo += 'Ctrl+';
        if (event.altKey) combo += 'Alt+';
        if (event.shiftKey) combo += 'Shift+';
        combo += event.key;
        return combo;
    }

    showShortcutFeedback(key) {
        // ビジュアルフィードバックの表示
        const feedback = document.createElement('div');
        feedback.className = 'shortcut-feedback';
        feedback.textContent = key;
        document.body.appendChild(feedback);

        setTimeout(() => feedback.remove(), 500);
    }
}

// クイックラベリングモード
class QuickLabelingMode {
    constructor(interface) {
        this.interface = interface;
        this.enabled = false;
        this.lastTileType = null;
    }

    enable() {
        this.enabled = true;
        this.interface.showMessage('クイックラベリングモード: ON');
        this.setupQuickMode();
    }

    setupQuickMode() {
        // クリックだけでバウンディングボックスを作成
        this.interface.canvas.addEventListener('click', (e) => {
            if (!this.enabled) return;

            const rect = this.interface.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // 最後に使用した牌種で自動的にボックスを作成
            if (this.lastTileType) {
                this.interface.createBoxAtPoint(x, y, this.lastTileType);
            }
        });
    }
}
```

### 2.2 バッチラベリング機能（Day 5）✅ 実装済み

#### 実装ファイル: `src/training/labeling/batch_labeler.py`

**実装内容:**
- ✅ テンプレートベースの一括適用
- ✅ オプティカルフローによる自動追跡
- ✅ スマートな前方伝播機能
- ✅ フレーム間の類似度計算
- ✅ WebインターフェースアダプターによるUIレベルの統合

```python
class BatchLabeler:
    """複数フレームの一括ラベリング機能"""

    def __init__(self, interface):
        self.interface = interface
        self.templates = {}

    def create_template_from_frame(self, frame_id: str) -> Dict[str, Any]:
        """現在のフレームからテンプレートを作成"""
        annotations = self.interface.get_frame_annotations(frame_id)

        template = {
            'tile_positions': [],
            'tile_types': [],
            'created_from': frame_id,
            'timestamp': datetime.now()
        }

        for ann in annotations:
            template['tile_positions'].append({
                'bbox': ann['bbox'],
                'relative_position': self._calculate_relative_position(ann['bbox'])
            })
            template['tile_types'].append(ann['tile_type'])

        return template

    def apply_template_to_frames(self, template: Dict[str, Any],
                               frame_ids: List[str],
                               auto_adjust: bool = True):
        """テンプレートを複数フレームに適用"""
        results = []

        for frame_id in frame_ids:
            if auto_adjust:
                # フレーム間の位置ずれを自動補正
                adjusted_template = self._adjust_template_for_frame(
                    template, frame_id
                )
            else:
                adjusted_template = template

            # アノテーションの適用
            annotations = self._apply_template(adjusted_template, frame_id)
            results.append({
                'frame_id': frame_id,
                'annotations': annotations,
                'success': len(annotations) > 0
            })

        return results

    def smart_propagation(self, start_frame: int, end_frame: int,
                         confidence_threshold: float = 0.8):
        """スマートな前方伝播"""
        current_annotations = self.interface.get_frame_annotations(start_frame)

        for frame_idx in range(start_frame + 1, end_frame + 1):
            # 前フレームとの差分を計算
            frame_diff = self._calculate_frame_difference(frame_idx - 1, frame_idx)

            if frame_diff < 0.1:  # ほぼ同じフレーム
                # アノテーションをそのままコピー
                self._copy_annotations(frame_idx - 1, frame_idx)
            else:
                # オプティカルフローで位置を追跡
                adjusted_annotations = self._track_tiles_optical_flow(
                    current_annotations, frame_idx - 1, frame_idx
                )

                # 信頼度の高いものだけを適用
                filtered = [
                    ann for ann in adjusted_annotations
                    if ann['confidence'] > confidence_threshold
                ]

                self.interface.set_frame_annotations(frame_idx, filtered)
                current_annotations = filtered

    def _track_tiles_optical_flow(self, annotations: List[Dict],
                                 prev_frame_idx: int,
                                 curr_frame_idx: int) -> List[Dict]:
        """オプティカルフローを使用した牌の追跡"""
        prev_frame = self.interface.get_frame(prev_frame_idx)
        curr_frame = self.interface.get_frame(curr_frame_idx)

        # オプティカルフロー計算
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        adjusted_annotations = []
        for ann in annotations:
            # バウンディングボックスの中心点を追跡
            cx, cy = self._get_bbox_center(ann['bbox'])
            dx, dy = flow[int(cy), int(cx)]

            # 新しい位置を計算
            new_bbox = [
                ann['bbox'][0] + dx,
                ann['bbox'][1] + dy,
                ann['bbox'][2] + dx,
                ann['bbox'][3] + dy
            ]

            adjusted_annotations.append({
                **ann,
                'bbox': new_bbox,
                'confidence': self._calculate_tracking_confidence(flow, ann['bbox'])
            })

        return adjusted_annotations
```

## Day 6-7: YOLOv8対応 ✅ 完了

### 3.1 YOLOv8統合（Day 6）✅ 実装済み

#### 実装ファイル: `src/detection/yolov8_detector.py`

**実装内容:**
- ✅ ultralyticsライブラリの完全統合
- ✅ YOLOv8形式のデータセット自動変換
- ✅ 最適化された訓練パラメータ
- ✅ バッチ予測とモデル評価機能
- ✅ 可視化機能とエクスポート機能

```python
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path

class YOLOv8TileDetector:
    """YOLOv8を使用した麻雀牌検出器"""

    def __init__(self, model_path: str = None, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.class_names = self._setup_class_names()

    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _load_model(self, model_path: str) -> YOLO:
        if model_path and Path(model_path).exists():
            # 既存のモデルを読み込み
            return YOLO(model_path)
        else:
            # 新規モデルの作成
            return YOLO('yolov8n.yaml')  # nano版から開始

    def _setup_class_names(self) -> List[str]:
        """麻雀牌のクラス名を設定"""
        tiles = []

        # 数牌
        for suit in ['m', 'p', 's']:
            for num in range(1, 10):
                tiles.append(f"{num}{suit}")

        # 字牌
        tiles.extend(['1z', '2z', '3z', '4z', '5z', '6z', '7z'])

        # 赤ドラ
        tiles.extend(['0m', '0p', '0s'])

        # 裏面
        tiles.append('back')

        return tiles

    def prepare_training_data(self, dataset_path: str, output_path: str):
        """YOLOv8形式のデータセット準備"""
        # ディレクトリ構造の作成
        output_dir = Path(output_path)
        (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        # データセット設定ファイルの作成
        yaml_content = f"""
path: {output_dir.absolute()}
train: images/train
val: images/val

names:
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(self.class_names))}

nc: {len(self.class_names)}
"""

        with open(output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)

        # アノテーションの変換
        self._convert_annotations(dataset_path, output_dir)

    def _convert_annotations(self, input_path: str, output_dir: Path):
        """既存のアノテーションをYOLO形式に変換"""
        from ..dataset_manager import DatasetManager

        dm = DatasetManager()
        annotations = dm.get_all_annotations()

        for ann in annotations:
            # 画像のコピー
            img_path = Path(ann['image_path'])
            split = 'train' if np.random.random() < 0.8 else 'val'

            new_img_path = output_dir / 'images' / split / img_path.name
            shutil.copy(img_path, new_img_path)

            # ラベルファイルの作成
            label_path = output_dir / 'labels' / split / f"{img_path.stem}.txt"

            with open(label_path, 'w') as f:
                for tile in ann['tiles']:
                    # YOLO形式に変換 (class_id, x_center, y_center, width, height)
                    class_id = self.class_names.index(tile['class'])
                    x_center = (tile['bbox'][0] + tile['bbox'][2]) / 2 / ann['width']
                    y_center = (tile['bbox'][1] + tile['bbox'][3]) / 2 / ann['height']
                    width = (tile['bbox'][2] - tile['bbox'][0]) / ann['width']
                    height = (tile['bbox'][3] - tile['bbox'][1]) / ann['height']

                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    def train(self, data_yaml: str, epochs: int = 100, batch_size: int = 16):
        """YOLOv8モデルの訓練"""
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=self.device,
            project='models/yolov8',
            name='mahjong_tiles',
            exist_ok=True,

            # 最適化設定
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,

            # データ拡張（YOLOv8内蔵）
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,

            # その他の設定
            close_mosaic=10,
            amp=True,  # 自動混合精度
            patience=50,  # 早期停止
            save=True,
            save_period=10,
            val=True,
            plots=True,
        )

        return results

    def predict(self, image: np.ndarray, conf_threshold: float = 0.5):
        """画像から麻雀牌を検出"""
        results = self.model(
            image,
            conf=conf_threshold,
            iou=0.45,
            max_det=300,
            classes=None,  # すべてのクラスを検出
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                        'confidence': float(boxes.conf[i]),
                        'class_id': int(boxes.cls[i]),
                        'class_name': self.class_names[int(boxes.cls[i])]
                    }
                    detections.append(detection)

        return detections
```

### 3.2 訓練パイプラインの統合（Day 7）✅ 実装済み

#### 実装ファイル: `src/training/unified_trainer.py`

**実装内容:**
- ✅ 完全自動化された訓練パイプライン
- ✅ データ拡張→YOLO変換→訓練→評価の統合
- ✅ 詳細なレポート生成（JSON/Markdown）
- ✅ ファインチューニング機能
- ✅ バッチ推論機能

```python
class UnifiedModelTrainer:
    """データ拡張、ラベリング、YOLOv8を統合した訓練システム"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.augmentor = UnifiedAugmentor(self.config['augmentation'])
        self.detector = YOLOv8TileDetector()
        self.batch_labeler = BatchLabeler(None)  # インターフェースは後で設定

    def create_initial_model(self,
                           raw_data_path: str,
                           output_model_path: str,
                           target_accuracy: float = 0.5):
        """初期モデル作成の完全なワークフロー"""

        # Step 1: データ拡張
        print("Step 1: データ拡張を実行中...")
        augmented_path = self._augment_initial_data(raw_data_path)

        # Step 2: YOLOv8形式への変換
        print("Step 2: YOLOv8形式に変換中...")
        yolo_dataset_path = self._prepare_yolo_dataset(augmented_path)

        # Step 3: 初期訓練
        print("Step 3: 初期モデルの訓練を開始...")
        self._train_initial_model(yolo_dataset_path, output_model_path)

        # Step 4: 評価とレポート生成
        print("Step 4: モデルの評価中...")
        metrics = self._evaluate_model(output_model_path)

        if metrics['mAP'] >= target_accuracy:
            print(f"✅ 目標精度 {target_accuracy} を達成: mAP = {metrics['mAP']:.3f}")
        else:
            print(f"⚠️ 目標精度未達: mAP = {metrics['mAP']:.3f} < {target_accuracy}")

        # レポートの生成
        self._generate_training_report(metrics, output_model_path)

        return metrics

    def _augment_initial_data(self, raw_data_path: str) -> str:
        """初期データの拡張（1,200枚 → 24,000枚）"""
        output_path = Path(raw_data_path).parent / 'augmented_data'

        # データ読み込み
        raw_data = self._load_raw_data(raw_data_path)

        # クラスバランスを考慮した拡張
        balanced_data = self.augmentor.advanced.create_balanced_dataset(
            raw_data,
            target_per_class=650  # 37クラス × 650 ≈ 24,000
        )

        # 保存
        self._save_augmented_data(balanced_data, output_path)

        return str(output_path)

    def _prepare_yolo_dataset(self, augmented_path: str) -> str:
        """YOLOv8形式のデータセット準備"""
        output_path = Path(augmented_path).parent / 'yolo_dataset'

        self.detector.prepare_training_data(
            augmented_path,
            str(output_path)
        )

        return str(output_path / 'dataset.yaml')

    def _train_initial_model(self, data_yaml: str, output_path: str):
        """初期モデルの訓練"""
        # 軽量な設定で高速に訓練
        results = self.detector.train(
            data_yaml=data_yaml,
            epochs=50,  # 初期は少なめ
            batch_size=32,
            imgsz=416,  # 初期は小さめの画像サイズ
        )

        # 最良のモデルを保存
        best_model = Path('models/yolov8/mahjong_tiles/weights/best.pt')
        if best_model.exists():
            shutil.copy(best_model, output_path)

    def _evaluate_model(self, model_path: str) -> Dict[str, float]:
        """モデルの評価"""
        self.detector.model = YOLO(model_path)

        # 検証データでの評価
        metrics = self.detector.model.val()

        return {
            'mAP': float(metrics.box.map),
            'mAP50': float(metrics.box.map50),
            'mAP75': float(metrics.box.map75),
            'precision': float(metrics.box.p),
            'recall': float(metrics.box.r),
            'classes': self._evaluate_per_class(metrics)
        }

    def _generate_training_report(self, metrics: Dict, model_path: str):
        """訓練レポートの生成"""
        report_path = Path(model_path).parent / 'training_report.md'

        report_content = f"""
# 初期モデル訓練レポート

## 概要
- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- モデルパス: {model_path}

## 全体的なメトリクス
- mAP@0.5: {metrics['mAP50']:.3f}
- mAP@0.5:0.95: {metrics['mAP']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}

## クラス別性能
| クラス | AP@0.5 | Precision | Recall |
|--------|--------|-----------|--------|
"""

        for class_name, class_metrics in metrics['classes'].items():
            report_content += f"| {class_name} | {class_metrics['ap50']:.3f} | "
            report_content += f"{class_metrics['precision']:.3f} | "
            report_content += f"{class_metrics['recall']:.3f} |\n"

        report_content += """
## 推奨事項
"""

        # 性能に基づく推奨事項
        if metrics['mAP'] < 0.3:
            report_content += "- より多くのデータが必要です\n"
            report_content += "- データ拡張をさらに強化してください\n"
        elif metrics['mAP'] < 0.5:
            report_content += "- 半自動ラベリングを開始できます\n"
            report_content += "- 難しいサンプルを追加してください\n"
        else:
            report_content += "- 実用レベルに近づいています\n"
            report_content += "- 実環境でのテストを開始してください\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
```

## テストとデバッグ計画

### 機能別テスト項目

#### 1. データ拡張テスト
```python
def test_augmentation():
    # 1枚の画像から20枚生成されることを確認
    augmentor = AdvancedAugmentor(augmentation_factor=20)
    test_image = cv2.imread('test_tile.jpg')
    augmented = augmentor.augment_single_image(
        test_image,
        [100, 100, 200, 200],
        class_id=0
    )
    assert len(augmented) == 20

    # 各拡張画像が異なることを確認
    hashes = [imagehash.average_hash(Image.fromarray(a['image']))
              for a in augmented]
    assert len(set(hashes)) > 15  # 少なくとも15種類は異なる
```

#### 2. キーボードショートカットテスト
```javascript
// Cypressを使用したE2Eテスト
describe('Keyboard Shortcuts', () => {
    it('should navigate frames with arrow keys', () => {
        cy.visit('/labeling');
        cy.get('body').type('{rightarrow}');
        cy.get('#frame-counter').should('contain', '2');
    });

    it('should copy previous frame with Alt+C', () => {
        cy.get('body').type('{alt}c');
        cy.get('.annotation-box').should('have.length.gt', 0);
    });
});
```

#### 3. YOLOv8統合テスト
```python
def test_yolov8_integration():
    detector = YOLOv8TileDetector()

    # データ準備のテスト
    detector.prepare_training_data('test_data/', 'output/')
    assert Path('output/dataset.yaml').exists()
    assert Path('output/images/train').exists()

    # 予測のテスト
    test_image = cv2.imread('test_frame.jpg')
    detections = detector.predict(test_image)
    assert isinstance(detections, list)
```

## 成功基準

### Week 1終了時点（機能開発完了）✅ 達成
- ✅ データ拡張で1枚→20枚の生成が可能
- ✅ 主要なキーボードショートカットが動作
- ✅ YOLOv8でのデータ準備・訓練が可能
- ✅ 統合テストがすべてパス

### Week 2終了時点（初期モデル完成）
- [ ] 1,200枚の手動ラベリングデータ
- [ ] 24,000枚の拡張データ
- [ ] mAP 40%以上の初期モデル
- [ ] 半自動ラベリングの動作確認

## 実装完了状況サマリー

### ✅ 完了した機能（7日間）

1. **データ拡張システム（Day 1-3）**
   - Albumentations統合による20種類以上の変換
   - 赤ドラ検出用の特殊な色拡張
   - 統合拡張システムとYOLO形式出力

2. **ラベリング効率化（Day 4-5）**
   - 30種類以上のキーボードショートカット
   - クイックラベリングモード
   - オプティカルフローによるバッチラベリング
   - Webインターフェースアダプター

3. **YOLOv8統合（Day 6-7）**
   - ultralytics公式ライブラリ統合
   - 自動データセット変換
   - 統合訓練パイプライン
   - ファインチューニング・バッチ推論対応

### 🚀 次のステップ

1. **実データでのテスト**
   - 実際の対局動画からフレーム抽出
   - 初期1,200枚のラベリング実施
   - 初期モデルの訓練と評価

2. **半自動化への移行**
   - 初期モデルによる予測
   - 人間による修正と承認
   - データセットの段階的拡大

3. **継続的改善**
   - アクティブラーニングの実装
   - 特殊ケース（赤ドラ・裏面）の強化
   - 実環境での性能評価

## リスクと対策

### リスク1: データ拡張の品質
**対策**:
- 拡張パラメータの慎重な調整
- 人間による品質チェック（サンプリング）
- 過度な変形を避ける

### リスク2: YOLOv8の学習が収束しない
**対策**:
- 学習率の調整
- より小さなモデル（yolov8n）から開始
- バッチサイズの最適化

### リスク3: ラベリング作業の遅延
**対策**:
- 効率化ツールの優先実装
- 複数人での作業分担
- 品質より量を重視（初期段階）

## まとめ

この実装計画に従うことで、2週間で初期モデルの構築が可能になります。特に重要なのは：

1. **最初の3日間でデータ拡張を完成**させること
2. **ラベリング効率を最大化**すること
3. **YOLOv8を早期に動作**させること

これらの機能が揃えば、少ないデータからでも実用的な初期モデルが構築できます。
