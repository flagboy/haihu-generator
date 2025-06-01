"""
半自動ラベリングシステム

既存モデルを使った初期予測と、ユーザー修正インターフェースの基盤を提供
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..classification.tile_classifier import ClassificationResult, TileClassifier
from ..detection.tile_detector import DetectionResult, TileDetector
from ..utils.config import ConfigManager
from ..utils.file_io import FileIOHelper
from ..utils.logger import LoggerMixin
from .annotation_data import AnnotationData, BoundingBox, FrameAnnotation, TileAnnotation


class PredictionResult:
    """予測結果クラス"""

    def __init__(
        self,
        frame_annotation: FrameAnnotation,
        detection_results: list[DetectionResult],
        classification_results: list[ClassificationResult],
    ):
        """
        初期化

        Args:
            frame_annotation: フレームアノテーション
            detection_results: 検出結果
            classification_results: 分類結果
        """
        self.frame_annotation = frame_annotation
        self.detection_results = detection_results
        self.classification_results = classification_results
        self.confidence_scores = self._calculate_confidence_scores()

    def _calculate_confidence_scores(self) -> dict[str, float]:
        """信頼度スコアを計算"""
        if not self.frame_annotation.tiles:
            return {"overall": 0.0, "detection": 0.0, "classification": 0.0}

        detection_confidences = [tile.confidence for tile in self.frame_annotation.tiles]

        return {
            "overall": np.mean(detection_confidences) if detection_confidences else 0.0,
            "detection": np.mean([det.confidence for det in self.detection_results])
            if self.detection_results
            else 0.0,
            "classification": np.mean([cls.confidence for cls in self.classification_results])
            if self.classification_results
            else 0.0,
            "tile_count": len(self.frame_annotation.tiles),
        }


class VisualizationGenerator:
    """可視化生成クラス"""

    def __init__(self):
        """初期化"""
        self.colors = {
            "high_confidence": (0, 255, 0),  # 緑
            "medium_confidence": (0, 255, 255),  # 黄
            "low_confidence": (0, 0, 255),  # 赤
            "hand": (255, 0, 0),  # 青
            "discard": (0, 165, 255),  # オレンジ
            "call": (255, 0, 255),  # マゼンタ
            "unknown": (128, 128, 128),  # グレー
        }

    def generate_prediction_visualization(
        self, image_path: str, prediction_result: PredictionResult
    ) -> np.ndarray:
        """
        予測結果の可視化画像を生成

        Args:
            image_path: 元画像のパス
            prediction_result: 予測結果

        Returns:
            可視化画像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        vis_image = image.copy()

        # 牌アノテーションを描画
        for tile in prediction_result.frame_annotation.tiles:
            self._draw_tile_annotation(vis_image, tile)

        # 統計情報を描画
        self._draw_statistics(vis_image, prediction_result.confidence_scores)

        return vis_image

    def _draw_tile_annotation(self, image: np.ndarray, tile: TileAnnotation):
        """牌アノテーションを描画"""
        bbox = tile.bbox

        # 信頼度に基づく色選択
        if tile.confidence >= 0.8:
            color = self.colors["high_confidence"]
        elif tile.confidence >= 0.5:
            color = self.colors["medium_confidence"]
        else:
            color = self.colors["low_confidence"]

        # バウンディングボックスを描画
        cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)

        # エリアタイプに基づく色でボーダーを追加
        area_color = self.colors.get(tile.area_type, self.colors["unknown"])
        cv2.rectangle(image, (bbox.x1 - 1, bbox.y1 - 1), (bbox.x2 + 1, bbox.y2 + 1), area_color, 1)

        # ラベルテキスト
        label = f"{tile.tile_id} ({tile.confidence:.2f})"
        if tile.area_type != "unknown":
            label += f" [{tile.area_type}]"

        # テキスト背景
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            image, (bbox.x1, bbox.y1 - text_height - 5), (bbox.x1 + text_width, bbox.y1), color, -1
        )

        # テキスト描画
        cv2.putText(
            image, label, (bbox.x1, bbox.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # 遮蔽情報
        if tile.is_occluded:
            cv2.putText(
                image,
                "OCCLUDED",
                (bbox.x1, bbox.y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

    def _draw_statistics(self, image: np.ndarray, confidence_scores: dict[str, float]):
        """統計情報を描画"""
        h, w = image.shape[:2]

        # 背景矩形
        cv2.rectangle(image, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (300, 100), (255, 255, 255), 2)

        # 統計テキスト
        stats_text = [
            f"Overall: {confidence_scores.get('overall', 0):.3f}",
            f"Detection: {confidence_scores.get('detection', 0):.3f}",
            f"Classification: {confidence_scores.get('classification', 0):.3f}",
            f"Tiles: {confidence_scores.get('tile_count', 0)}",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(
                image, text, (15, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

    def generate_comparison_view(
        self,
        image_path: str,
        prediction_result: PredictionResult,
        ground_truth: FrameAnnotation | None = None,
    ) -> np.ndarray:
        """
        予測結果と正解の比較表示を生成

        Args:
            image_path: 元画像のパス
            prediction_result: 予測結果
            ground_truth: 正解アノテーション

        Returns:
            比較画像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        h, w = image.shape[:2]

        if ground_truth is None:
            # 予測結果のみ表示
            return self.generate_prediction_visualization(image_path, prediction_result)

        # 左右分割表示
        comparison_image = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # 左側：予測結果
        pred_vis = self.generate_prediction_visualization(image_path, prediction_result)
        comparison_image[:, :w] = pred_vis

        # 右側：正解
        gt_vis = image.copy()
        for tile in ground_truth.tiles:
            bbox = tile.bbox
            cv2.rectangle(gt_vis, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
            cv2.putText(
                gt_vis,
                tile.tile_id,
                (bbox.x1, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        comparison_image[:, w:] = gt_vis

        # 分割線
        cv2.line(comparison_image, (w, 0), (w, h), (255, 255, 255), 2)

        # ラベル
        cv2.putText(
            comparison_image,
            "Prediction",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison_image,
            "Ground Truth",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return comparison_image


class SemiAutoLabeler(LoggerMixin):
    """半自動ラベリングクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()

        # 既存モデルを初期化
        self.tile_detector = TileDetector(self.config_manager)
        self.tile_classifier = TileClassifier(self.config_manager)
        self.visualization_generator = VisualizationGenerator()

        # 設定
        training_config = self.config_manager.get_config().get("training", {})
        labeling_config = training_config.get("semi_auto_labeling", {})

        self.confidence_threshold = labeling_config.get("confidence_threshold", 0.5)
        self.auto_area_classification = labeling_config.get("auto_area_classification", True)
        self.enable_occlusion_detection = labeling_config.get("enable_occlusion_detection", True)

        # 出力設定
        self.output_dir = Path(training_config.get("labeling_output_dir", "data/training/labeling"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_dir = self.output_dir / "predictions"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.corrections_dir = self.output_dir / "corrections"

        for dir_path in [self.predictions_dir, self.visualizations_dir, self.corrections_dir]:
            dir_path.mkdir(exist_ok=True)

        self.logger.info("SemiAutoLabeler初期化完了")

    def predict_frame_annotations(self, frame_annotation: FrameAnnotation) -> PredictionResult:
        """
        フレームの自動アノテーション予測

        Args:
            frame_annotation: 入力フレームアノテーション

        Returns:
            予測結果
        """
        image_path = frame_annotation.image_path
        if not Path(image_path).exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        self.logger.info(f"フレーム予測開始: {frame_annotation.frame_id}")

        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        # 牌検出
        detection_results = self.tile_detector.detect_tiles(image)

        # エリア分類
        if self.auto_area_classification:
            area_classified_detections = self.tile_detector.classify_tile_areas(
                image, detection_results
            )
        else:
            area_classified_detections = {"unknown": detection_results}

        # 牌分類
        classification_results = []
        predicted_tiles = []

        for area_type, detections in area_classified_detections.items():
            for detection in detections:
                # 牌領域を切り出し
                x1, y1, x2, y2 = detection.bbox
                tile_image = image[y1:y2, x1:x2]

                if tile_image.size > 0:
                    # 牌分類
                    classification_result = self.tile_classifier.classify_tile(tile_image)
                    classification_results.append(classification_result)

                    # 信頼度チェック
                    if classification_result.confidence >= self.confidence_threshold:
                        # 遮蔽検出
                        is_occluded, occlusion_ratio = (
                            self._detect_occlusion(tile_image)
                            if self.enable_occlusion_detection
                            else (False, 0.0)
                        )

                        # TileAnnotationを作成
                        bbox = BoundingBox(x1, y1, x2, y2)
                        tile_annotation = TileAnnotation(
                            tile_id=classification_result.tile_name,
                            bbox=bbox,
                            confidence=min(detection.confidence, classification_result.confidence),
                            area_type=area_type,
                            is_face_up=True,  # デフォルト
                            is_occluded=is_occluded,
                            occlusion_ratio=occlusion_ratio,
                            annotator="semi_auto_labeler",
                            notes=f"Detection: {detection.confidence:.3f}, Classification: {classification_result.confidence:.3f}",
                        )
                        predicted_tiles.append(tile_annotation)

        # フレームアノテーションを更新
        updated_frame = FrameAnnotation(
            frame_id=frame_annotation.frame_id,
            image_path=frame_annotation.image_path,
            image_width=frame_annotation.image_width,
            image_height=frame_annotation.image_height,
            timestamp=frame_annotation.timestamp,
            tiles=predicted_tiles,
            quality_score=frame_annotation.quality_score,
            is_valid=frame_annotation.is_valid,
            scene_type=frame_annotation.scene_type,
            game_phase=frame_annotation.game_phase,
            annotated_at=datetime.now(),
            annotator="semi_auto_labeler",
            notes=f"Auto-predicted {len(predicted_tiles)} tiles",
        )

        prediction_result = PredictionResult(
            updated_frame, detection_results, classification_results
        )

        self.logger.info(f"フレーム予測完了: {len(predicted_tiles)}個の牌を検出")
        return prediction_result

    def batch_predict_annotations(
        self, frame_annotations: list[FrameAnnotation]
    ) -> list[PredictionResult]:
        """
        複数フレームの一括予測

        Args:
            frame_annotations: フレームアノテーションのリスト

        Returns:
            予測結果のリスト
        """
        prediction_results = []

        for i, frame_annotation in enumerate(frame_annotations):
            try:
                prediction_result = self.predict_frame_annotations(frame_annotation)
                prediction_results.append(prediction_result)

                # 進捗表示
                if (i + 1) % 10 == 0:
                    self.logger.info(f"一括予測進捗: {i + 1}/{len(frame_annotations)}")

            except Exception as e:
                self.logger.error(f"フレーム予測エラー {frame_annotation.frame_id}: {e}")
                # エラーの場合は空の予測結果を作成
                empty_frame = FrameAnnotation(
                    frame_id=frame_annotation.frame_id,
                    image_path=frame_annotation.image_path,
                    image_width=frame_annotation.image_width,
                    image_height=frame_annotation.image_height,
                    timestamp=frame_annotation.timestamp,
                    tiles=[],
                    quality_score=frame_annotation.quality_score,
                    is_valid=False,
                    scene_type=frame_annotation.scene_type,
                    game_phase=frame_annotation.game_phase,
                    annotated_at=datetime.now(),
                    annotator="semi_auto_labeler",
                    notes=f"Prediction failed: {str(e)}",
                )
                prediction_results.append(PredictionResult(empty_frame, [], []))

        self.logger.info(f"一括予測完了: {len(prediction_results)}フレーム")
        return prediction_results

    def save_predictions(
        self, prediction_results: list[PredictionResult], output_name: str = "predictions"
    ) -> bool:
        """
        予測結果を保存

        Args:
            prediction_results: 予測結果のリスト
            output_name: 出力ファイル名

        Returns:
            保存成功かどうか
        """
        try:
            # アノテーションデータを作成
            annotation_data = AnnotationData()

            # 仮の動画IDを作成
            video_id = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 動画アノテーションを作成（仮の情報）
            from .annotation_data import VideoAnnotation

            video_annotation = VideoAnnotation(
                video_id=video_id,
                video_path="",
                video_name=output_name,
                duration=0.0,
                fps=30.0,
                width=1920,
                height=1080,
                frames=[result.frame_annotation for result in prediction_results],
            )

            annotation_data.video_annotations[video_id] = video_annotation

            # JSONファイルに保存
            output_path = self.predictions_dir / f"{output_name}.json"
            annotation_data.save_to_json(str(output_path))

            # 統計情報も保存
            stats = self._calculate_prediction_statistics(prediction_results)
            stats_path = self.predictions_dir / f"{output_name}_stats.json"
            # default=str to handle any non-serializable objects
            stats_json = json.loads(json.dumps(stats, default=str))
            FileIOHelper.save_json(stats_json, stats_path, pretty=True)

            self.logger.info(f"予測結果を保存: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"予測結果の保存に失敗: {e}")
            return False

    def generate_visualizations(
        self, prediction_results: list[PredictionResult], output_name: str = "visualizations"
    ) -> bool:
        """
        予測結果の可視化を生成

        Args:
            prediction_results: 予測結果のリスト
            output_name: 出力ディレクトリ名

        Returns:
            生成成功かどうか
        """
        try:
            vis_output_dir = self.visualizations_dir / output_name
            vis_output_dir.mkdir(exist_ok=True)

            for i, prediction_result in enumerate(prediction_results):
                frame_id = prediction_result.frame_annotation.frame_id
                image_path = prediction_result.frame_annotation.image_path

                # 可視化画像を生成
                vis_image = self.visualization_generator.generate_prediction_visualization(
                    image_path, prediction_result
                )

                # 保存
                output_path = vis_output_dir / f"{frame_id}_prediction.jpg"
                cv2.imwrite(str(output_path), vis_image)

                # 進捗表示
                if (i + 1) % 50 == 0:
                    self.logger.info(f"可視化生成進捗: {i + 1}/{len(prediction_results)}")

            self.logger.info(f"可視化生成完了: {vis_output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"可視化生成に失敗: {e}")
            return False

    def _detect_occlusion(self, tile_image: np.ndarray) -> tuple[bool, float]:
        """
        牌の遮蔽を検出

        Args:
            tile_image: 牌画像

        Returns:
            (遮蔽されているか, 遮蔽率)
        """
        if tile_image.size == 0:
            return True, 1.0

        # 簡易的な遮蔽検出
        # エッジ密度で判定
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        # 閾値以下なら遮蔽とみなす
        occlusion_threshold = 0.1
        is_occluded = edge_ratio < occlusion_threshold
        occlusion_ratio = max(0.0, 1.0 - (edge_ratio / occlusion_threshold))

        return is_occluded, occlusion_ratio

    def _calculate_prediction_statistics(
        self, prediction_results: list[PredictionResult]
    ) -> dict[str, Any]:
        """予測統計を計算"""
        if not prediction_results:
            return {}

        total_frames = len(prediction_results)
        total_tiles = sum(len(result.frame_annotation.tiles) for result in prediction_results)

        confidence_scores = []
        tile_types = set()
        area_distribution = {}

        for result in prediction_results:
            confidence_scores.append(result.confidence_scores.get("overall", 0.0))

            for tile in result.frame_annotation.tiles:
                tile_types.add(tile.tile_id)
                area_type = tile.area_type
                area_distribution[area_type] = area_distribution.get(area_type, 0) + 1

        return {
            "total_frames": total_frames,
            "total_tiles": total_tiles,
            "avg_tiles_per_frame": total_tiles / total_frames if total_frames > 0 else 0.0,
            "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0.0,
            "unique_tile_types": len(tile_types),
            "tile_types": sorted(list(tile_types)),
            "area_distribution": area_distribution,
            "high_confidence_frames": sum(1 for score in confidence_scores if score >= 0.8),
            "low_confidence_frames": sum(1 for score in confidence_scores if score < 0.5),
        }

    def create_correction_template(self, prediction_result: PredictionResult) -> dict[str, Any]:
        """
        修正用テンプレートを作成

        Args:
            prediction_result: 予測結果

        Returns:
            修正テンプレート
        """
        template = {
            "frame_id": prediction_result.frame_annotation.frame_id,
            "image_path": prediction_result.frame_annotation.image_path,
            "timestamp": prediction_result.frame_annotation.timestamp,
            "predictions": [],
            "corrections": [],
            "validation_status": "pending",
            "corrected_by": "",
            "corrected_at": "",
            "notes": "",
        }

        # 予測結果を追加
        for tile in prediction_result.frame_annotation.tiles:
            prediction_entry = {
                "tile_id": tile.tile_id,
                "bbox": {
                    "x1": tile.bbox.x1,
                    "y1": tile.bbox.y1,
                    "x2": tile.bbox.x2,
                    "y2": tile.bbox.y2,
                },
                "confidence": tile.confidence,
                "area_type": tile.area_type,
                "is_face_up": tile.is_face_up,
                "is_occluded": tile.is_occluded,
                "needs_correction": tile.confidence < 0.8,  # 低信頼度は修正が必要
            }
            template["predictions"].append(prediction_entry)

        return template

    def get_labeling_statistics(self) -> dict[str, Any]:
        """ラベリング統計情報を取得"""
        stats = {
            "output_directory": str(self.output_dir),
            "confidence_threshold": self.confidence_threshold,
            "auto_area_classification": self.auto_area_classification,
            "enable_occlusion_detection": self.enable_occlusion_detection,
        }

        # 出力ディレクトリの統計
        if self.predictions_dir.exists():
            prediction_files = list(self.predictions_dir.glob("*.json"))
            stats["prediction_files"] = len(
                [f for f in prediction_files if not f.name.endswith("_stats.json")]
            )

        if self.visualizations_dir.exists():
            vis_dirs = [d for d in self.visualizations_dir.iterdir() if d.is_dir()]
            total_vis_images = sum(len(list(d.glob("*.jpg"))) for d in vis_dirs)
            stats["visualization_images"] = total_vis_images

        return stats
