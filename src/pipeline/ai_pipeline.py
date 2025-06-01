"""
AI/MLデータ処理パイプライン
検出→分類の統合パイプラインとバッチ処理を提供
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..classification.tile_classifier import ClassificationResult, TileClassifier
from ..detection.tile_detector import DetectionResult, TileDetector
from ..utils.config import ConfigManager
from ..utils.logger import get_logger


@dataclass
class PipelineResult:
    """パイプライン処理結果を格納するデータクラス"""

    frame_id: int
    detections: list[DetectionResult]
    classifications: list[tuple[DetectionResult, ClassificationResult]]
    processing_time: float
    tile_areas: dict[str, list[DetectionResult]]
    confidence_scores: dict[str, float]


@dataclass
class BatchProcessingConfig:
    """バッチ処理設定"""

    batch_size: int = 8
    max_workers: int = 4
    enable_parallel: bool = True
    confidence_filter: float = 0.5


class AIPipeline:
    """AI/MLデータ処理パイプライン"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # AI/MLモジュールを初期化
        self.detector = TileDetector(config_manager)
        self.classifier = TileClassifier(config_manager)

        # バッチ処理設定
        self.batch_config = self._setup_batch_config()

        # 統計情報
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_classifications": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
        }

        self.logger.info("AIPipeline initialized")

    def _setup_batch_config(self) -> BatchProcessingConfig:
        """バッチ処理設定の初期化"""
        ai_config = self.config.get_config().get("ai", {})
        training_config = ai_config.get("training", {})
        system_config = self.config.get_config().get("system", {})

        return BatchProcessingConfig(
            batch_size=training_config.get("batch_size", 8),
            max_workers=system_config.get("max_workers", 4),
            enable_parallel=True,
            confidence_filter=ai_config.get("detection", {}).get("confidence_threshold", 0.5),
        )

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> PipelineResult:
        """
        単一フレームを処理

        Args:
            frame: 入力フレーム
            frame_id: フレームID

        Returns:
            処理結果
        """
        start_time = time.time()

        try:
            # 1. 牌検出
            detections = self.detector.detect_tiles(frame)
            self.logger.debug(f"Frame {frame_id}: Detected {len(detections)} tiles")

            # 2. エリア分類
            tile_areas = self.detector.classify_tile_areas(frame, detections)

            # 3. 牌分類
            classifications = []
            if detections:
                # 検出された牌画像を切り出し
                tile_images = self._extract_tile_images(frame, detections)

                # バッチ分類
                if len(tile_images) > 1:
                    classification_results = self.classifier.classify_tiles_batch(tile_images)
                else:
                    classification_results = (
                        [self.classifier.classify_tile(tile_images[0])] if tile_images else []
                    )

                # 検出結果と分類結果をペアリング
                for detection, classification in zip(
                    detections, classification_results, strict=False
                ):
                    classifications.append((detection, classification))

            # 4. 信頼度スコア計算
            confidence_scores = self._calculate_confidence_scores(classifications)

            # 5. 結果フィルタリング
            filtered_detections, filtered_classifications = self._filter_results(
                detections, classifications, confidence_scores
            )

            processing_time = time.time() - start_time

            # 統計更新
            self._update_stats(
                1, len(filtered_detections), len(filtered_classifications), processing_time
            )

            return PipelineResult(
                frame_id=frame_id,
                detections=filtered_detections,
                classifications=filtered_classifications,
                processing_time=processing_time,
                tile_areas=tile_areas,
                confidence_scores=confidence_scores,
            )

        except Exception as e:
            self.logger.error(f"Frame processing failed for frame {frame_id}: {e}")
            return PipelineResult(
                frame_id=frame_id,
                detections=[],
                classifications=[],
                processing_time=time.time() - start_time,
                tile_areas={},
                confidence_scores={},
            )

    def process_frames_batch(
        self, frames: list[np.ndarray], start_frame_id: int = 0
    ) -> list[PipelineResult]:
        """
        複数フレームをバッチ処理

        Args:
            frames: 入力フレームのリスト
            start_frame_id: 開始フレームID

        Returns:
            処理結果のリスト
        """
        if not frames:
            return []

        results = []

        if self.batch_config.enable_parallel and len(frames) > 1:
            # 並列処理
            results = self._process_frames_parallel(frames, start_frame_id)
        else:
            # 逐次処理
            for i, frame in enumerate(frames):
                result = self.process_frame(frame, start_frame_id + i)
                results.append(result)

        return results

    def _process_frames_parallel(
        self, frames: list[np.ndarray], start_frame_id: int
    ) -> list[PipelineResult]:
        """並列フレーム処理"""
        results = [None] * len(frames)

        with ThreadPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            # フレーム処理タスクを投入
            future_to_index = {
                executor.submit(self.process_frame, frame, start_frame_id + i): i
                for i, frame in enumerate(frames)
            }

            # 結果を収集
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(
                        f"Parallel processing failed for frame {start_frame_id + index}: {e}"
                    )
                    results[index] = PipelineResult(
                        frame_id=start_frame_id + index,
                        detections=[],
                        classifications=[],
                        processing_time=0.0,
                        tile_areas={},
                        confidence_scores={},
                    )

        return results

    def _extract_tile_images(
        self, frame: np.ndarray, detections: list[DetectionResult]
    ) -> list[np.ndarray]:
        """検出結果から牌画像を切り出し"""
        tile_images = []

        # 設定から最小タイルサイズを取得
        min_tile_size = (
            self.config.get_config().get("system", {}).get("constants", {}).get("min_tile_size", 10)
        )

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # バウンディングボックスを画像範囲内にクリップ
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # 牌画像を切り出し
            tile_image = frame[y1:y2, x1:x2]

            # 最小サイズチェック
            if tile_image.shape[0] > min_tile_size and tile_image.shape[1] > min_tile_size:
                tile_images.append(tile_image)
            else:
                # サイズが小さすぎる場合はダミー画像
                dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)
                tile_images.append(dummy_image)

        return tile_images

    def _calculate_confidence_scores(
        self, classifications: list[tuple[DetectionResult, ClassificationResult]]
    ) -> dict[str, float]:
        """信頼度スコアを計算"""
        if not classifications:
            return {}

        detection_confidences = [det.confidence for det, _ in classifications]
        classification_confidences = [cls.confidence for _, cls in classifications]

        scores = {
            "avg_detection_confidence": np.mean(detection_confidences),
            "avg_classification_confidence": np.mean(classification_confidences),
            "min_detection_confidence": np.min(detection_confidences),
            "min_classification_confidence": np.min(classification_confidences),
            "combined_confidence": np.mean(
                [det.confidence * cls.confidence for det, cls in classifications]
            ),
        }

        return scores

    def _filter_results(
        self,
        detections: list[DetectionResult],
        classifications: list[tuple[DetectionResult, ClassificationResult]],
        confidence_scores: dict[str, float],
    ) -> tuple[list[DetectionResult], list[tuple[DetectionResult, ClassificationResult]]]:
        """結果をフィルタリング"""
        threshold = self.batch_config.confidence_filter

        # 信頼度によるフィルタリング
        filtered_detections = [det for det in detections if det.confidence >= threshold]

        filtered_classifications = [
            (det, cls)
            for det, cls in classifications
            if det.confidence >= threshold and cls.confidence >= threshold
        ]

        return filtered_detections, filtered_classifications

    def post_process_results(self, results: list[PipelineResult]) -> dict[str, Any]:
        """結果の後処理"""
        if not results:
            return {}

        # 全体統計
        total_detections = sum(len(r.detections) for r in results)
        total_classifications = sum(len(r.classifications) for r in results)
        total_time = sum(r.processing_time for r in results)

        # 牌種類別統計
        tile_counts = {}
        confidence_stats = []

        for result in results:
            for _, classification in result.classifications:
                tile_name = classification.tile_name
                tile_counts[tile_name] = tile_counts.get(tile_name, 0) + 1
                confidence_stats.append(classification.confidence)

        # エリア別統計
        area_stats = {"hand_tiles": 0, "discarded_tiles": 0, "called_tiles": 0}

        for result in results:
            for area, detections in result.tile_areas.items():
                if area in area_stats:
                    area_stats[area] += len(detections)

        processed_results = {
            "summary": {
                "total_frames": len(results),
                "total_detections": total_detections,
                "total_classifications": total_classifications,
                "total_processing_time": total_time,
                "average_processing_time": total_time / len(results) if results else 0,
                "fps": len(results) / total_time if total_time > 0 else 0,
            },
            "tile_statistics": tile_counts,
            "area_statistics": area_stats,
            "confidence_statistics": {
                "mean": np.mean(confidence_stats) if confidence_stats else 0,
                "std": np.std(confidence_stats) if confidence_stats else 0,
                "min": np.min(confidence_stats) if confidence_stats else 0,
                "max": np.max(confidence_stats) if confidence_stats else 0,
            },
        }

        return processed_results

    def _update_stats(
        self, frames: int, detections: int, classifications: int, processing_time: float
    ):
        """統計情報を更新"""
        self.stats["total_frames"] += frames
        self.stats["total_detections"] += detections
        self.stats["total_classifications"] += classifications
        self.stats["total_processing_time"] += processing_time

        if self.stats["total_frames"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_frames"]
            )

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        return self.stats.copy()

    def reset_statistics(self):
        """統計情報をリセット"""
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_classifications": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
        }

    def visualize_results(self, frame: np.ndarray, result: PipelineResult) -> np.ndarray:
        """結果を可視化"""
        vis_frame = frame.copy()

        # 検出結果を描画
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 信頼度を表示
            label = f"Det: {detection.confidence:.2f}"
            cv2.putText(
                vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # 分類結果を描画
        for detection, classification in result.classifications:
            x1, y1, x2, y2 = detection.bbox

            # 牌名と信頼度を表示
            label = f"{classification.tile_name}: {classification.confidence:.2f}"
            cv2.putText(
                vis_frame, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        # フレーム情報を表示
        info_text = f"Frame: {result.frame_id}, Time: {result.processing_time:.3f}s"
        cv2.putText(
            vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return vis_frame

    def save_results(self, results: list[PipelineResult], output_path: str, format: str = "json"):
        """結果を保存"""
        import json
        import pickle

        try:
            if format == "json":
                # JSON形式で保存（シリアライズ可能な部分のみ）
                serializable_results = []
                for result in results:
                    serializable_result = {
                        "frame_id": result.frame_id,
                        "processing_time": result.processing_time,
                        "detections": [
                            {
                                "bbox": det.bbox,
                                "confidence": det.confidence,
                                "class_id": det.class_id,
                                "class_name": det.class_name,
                            }
                            for det in result.detections
                        ],
                        "classifications": [
                            {
                                "detection": {
                                    "bbox": det.bbox,
                                    "confidence": det.confidence,
                                    "class_id": det.class_id,
                                    "class_name": det.class_name,
                                },
                                "classification": {
                                    "tile_name": cls.tile_name,
                                    "confidence": cls.confidence,
                                    "class_id": cls.class_id,
                                },
                            }
                            for det, cls in result.classifications
                        ],
                        "confidence_scores": result.confidence_scores,
                    }
                    serializable_results.append(serializable_result)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            elif format == "pickle":
                # Pickle形式で保存（完全なオブジェクト）
                with open(output_path, "wb") as f:
                    pickle.dump(results, f)

            self.logger.info(f"Results saved to {output_path} in {format} format")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
