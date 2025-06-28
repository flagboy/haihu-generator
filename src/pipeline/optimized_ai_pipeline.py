"""
最適化されたAIパイプライン
バッチ処理とメモリ管理の改善版
"""

import time
from typing import Any

import numpy as np
import torch

from src.classification.tile_classifier import ClassificationResult, TileClassifier
from src.detection.tile_detector import DetectionResult, TileDetector
from src.optimization.batch_processing import (
    BatchSizeOptimizer,
    OptimizedBatchProcessor,
    ParallelBatchProcessor,
)
from src.pipeline.ai_pipeline import BatchConfig, PipelineResult
from src.utils.config import ConfigManager
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizedAIPipeline:
    """最適化されたAIパイプライン"""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        batch_config: BatchConfig | None = None,
        auto_optimize: bool = True,
    ):
        """
        Args:
            config_manager: 設定マネージャー
            batch_config: バッチ処理設定
            auto_optimize: 自動最適化を有効にするか
        """
        self.config = config_manager or ConfigManager()
        self.batch_config = batch_config or BatchConfig()
        self.auto_optimize = auto_optimize
        self.device = get_device()

        # AI モデルの初期化
        self.detector = TileDetector(self.config)
        self.classifier = TileClassifier(self.config)

        # バッチ処理最適化の初期化
        self._init_batch_processors()

        # 統計情報
        self.stats = {
            "frames_processed": 0,
            "tiles_detected": 0,
            "tiles_classified": 0,
            "total_processing_time": 0.0,
            "batch_sizes_used": [],
        }

        logger.info("OptimizedAIPipeline初期化完了")

    def _init_batch_processors(self) -> None:
        """バッチプロセッサーを初期化"""
        # 基本バッチプロセッサー
        self.batch_processor = OptimizedBatchProcessor(
            batch_size=self.batch_config.batch_size,
            auto_optimize=self.auto_optimize,
            device=self.device,
        )

        # 並列バッチプロセッサー（並列処理用）
        if self.batch_config.enable_parallel:
            self.parallel_processor = ParallelBatchProcessor(
                max_workers=self.batch_config.max_workers,
                batch_size=self.batch_config.parallel_batch_size,
                auto_optimize=self.auto_optimize,
                device=self.device,
            )

        # バッチサイズ最適化器
        if self.auto_optimize:
            self.batch_optimizer = BatchSizeOptimizer(
                initial_batch_size=self.batch_config.batch_size,
                min_batch_size=8,
                max_batch_size=128,
            )

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> PipelineResult:
        """単一フレームを処理（既存互換性のため）"""
        return self.process_frames_batch([frame], start_frame_id=frame_id)[0]

    def process_frames_batch(
        self, frames: list[np.ndarray], start_frame_id: int = 0
    ) -> list[PipelineResult]:
        """
        複数フレームを最適化されたバッチ処理で実行

        Args:
            frames: 入力フレームのリスト
            start_frame_id: 開始フレームID

        Returns:
            処理結果のリスト
        """
        if not frames:
            return []

        start_time = time.time()

        # メモリ最適化
        self.batch_processor.optimize_memory()

        # バッチサイズの動的調整
        if self.auto_optimize and hasattr(self, "batch_optimizer"):
            try:
                # テスト実行で最適なバッチサイズを探索
                optimal_size = self.batch_optimizer.find_optimal_batch_size(
                    lambda size: self._test_batch_size(frames[: min(size, len(frames))])
                )
                self.batch_processor.batch_size = optimal_size
                if hasattr(self, "parallel_processor"):
                    self.parallel_processor.batch_size = optimal_size
            except Exception as e:
                logger.warning(f"バッチサイズ最適化エラー: {e}")

        # 並列処理の選択
        if (
            self.batch_config.enable_parallel
            and len(frames) > self.batch_config.parallel_batch_size
            and hasattr(self, "parallel_processor")
        ):
            results = self._process_parallel_optimized(frames, start_frame_id)
        else:
            results = self._process_sequential_optimized(frames, start_frame_id)

        # 統計更新
        processing_time = time.time() - start_time
        self._update_global_stats(len(frames), results, processing_time)

        return results

    def _process_sequential_optimized(
        self, frames: list[np.ndarray], start_frame_id: int
    ) -> list[PipelineResult]:
        """逐次処理（最適化版）"""
        results = []

        # バッチ単位で処理
        for batch_frames in self.batch_processor.stream_batches(frames):
            batch_start_id = start_frame_id + len(results)

            # 検出フェーズ（バッチ処理）
            all_detections = self._batch_detect_tiles(batch_frames)

            # 分類フェーズ（バッチ処理）
            all_classifications = self._batch_classify_tiles(batch_frames, all_detections)

            # 結果の作成
            for i, (frame, detections, classifications) in enumerate(
                zip(batch_frames, all_detections, all_classifications, strict=False)
            ):
                frame_id = batch_start_id + i
                result = self._create_result(frame_id, frame, detections, classifications)
                results.append(result)

            # 定期的なメモリ最適化
            if len(results) % 100 == 0:
                self.batch_processor.optimize_memory()

        return results

    def _process_parallel_optimized(
        self, frames: list[np.ndarray], start_frame_id: int
    ) -> list[PipelineResult]:
        """並列処理（最適化版）"""

        def process_chunk(chunk_data: tuple[list[np.ndarray], int]) -> list[PipelineResult]:
            chunk_frames, chunk_start_id = chunk_data
            return self._process_sequential_optimized(chunk_frames, chunk_start_id)

        # チャンクに分割
        chunk_size = self.batch_config.parallel_batch_size
        chunks = []
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i : i + chunk_size]
            chunks.append((chunk, start_frame_id + i))

        # 並列処理
        all_results = []
        with self.parallel_processor.executor as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

            for future in futures:
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"並列処理エラー: {e}")

        return all_results

    def _batch_detect_tiles(self, frames: list[np.ndarray]) -> list[list[DetectionResult]]:
        """バッチ単位で牌検出"""
        all_detections = []

        # 検出器がバッチ処理をサポートしている場合
        if hasattr(self.detector, "detect_tiles_batch"):
            # GPUメモリ効率を考慮したサブバッチ処理
            sub_batch_size = min(self.batch_processor.batch_size, 16)
            for i in range(0, len(frames), sub_batch_size):
                sub_batch = frames[i : i + sub_batch_size]
                sub_detections = self.detector.detect_tiles_batch(sub_batch)
                all_detections.extend(sub_detections)
        else:
            # フォールバック：逐次処理
            for frame in frames:
                detections = self.detector.detect_tiles(frame)
                all_detections.append(detections)

        return all_detections

    def _batch_classify_tiles(
        self, frames: list[np.ndarray], all_detections: list[list[DetectionResult]]
    ) -> list[list[tuple[DetectionResult, ClassificationResult]]]:
        """バッチ単位で牌分類"""
        all_classifications = []

        # 全フレームから牌画像を収集
        all_tile_images = []
        tile_to_frame_map = []  # (frame_idx, detection_idx)

        for frame_idx, (frame, detections) in enumerate(zip(frames, all_detections, strict=False)):
            tile_images = self._extract_tile_images(frame, detections)
            all_tile_images.extend(tile_images)
            tile_to_frame_map.extend([(frame_idx, i) for i in range(len(tile_images))])

        # バッチ分類
        if all_tile_images:
            # デバイスへの転送を最適化
            if torch.cuda.is_available():
                # GPU転送をバッチ化
                classification_results = self.classifier.classify_tiles_batch(
                    all_tile_images, device=self.device
                )
            else:
                classification_results = self.classifier.classify_tiles_batch(all_tile_images)

            # 結果を各フレームに割り当て
            frame_classifications = [[] for _ in range(len(frames))]
            for tile_idx, (frame_idx, det_idx) in enumerate(tile_to_frame_map):
                detection = all_detections[frame_idx][det_idx]
                classification = classification_results[tile_idx]
                frame_classifications[frame_idx].append((detection, classification))

            all_classifications = frame_classifications
        else:
            all_classifications = [[] for _ in range(len(frames))]

        return all_classifications

    def _extract_tile_images(
        self, frame: np.ndarray, detections: list[DetectionResult]
    ) -> list[np.ndarray]:
        """検出結果から牌画像を切り出し（最適化版）"""
        tile_images = []
        h, w = frame.shape[:2]
        min_tile_size = 10

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # バウンディングボックスを画像範囲内にクリップ（ベクトル化）
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # サイズチェック
            if (x2 - x1) > min_tile_size and (y2 - y1) > min_tile_size:
                tile_image = frame[y1:y2, x1:x2]
                tile_images.append(tile_image)

        return tile_images

    def _create_result(
        self,
        frame_id: int,
        frame: np.ndarray,
        detections: list[DetectionResult],
        classifications: list[tuple[DetectionResult, ClassificationResult]],
    ) -> PipelineResult:
        """処理結果を作成"""
        # エリア分類
        tile_areas = self.detector.classify_tile_areas(frame, detections) if detections else {}

        # 信頼度スコア計算
        confidence_scores = self._calculate_confidence_scores(classifications)

        # 結果フィルタリング
        filtered_detections, filtered_classifications = self._filter_results(
            detections, classifications
        )

        return PipelineResult(
            frame_id=frame_id,
            detections=filtered_detections,
            classifications=filtered_classifications,
            processing_time=0.0,  # バッチ処理では個別計測しない
            tile_areas=tile_areas,
            confidence_scores=confidence_scores,
        )

    def _calculate_confidence_scores(
        self, classifications: list[tuple[DetectionResult, ClassificationResult]]
    ) -> dict[str, float]:
        """信頼度スコアを計算（最適化版）"""
        if not classifications:
            return {}

        # NumPy配列で一括計算
        det_confs = np.array([det.confidence for det, _ in classifications])
        cls_confs = np.array([cls.confidence for _, cls in classifications])
        combined = det_confs * cls_confs

        return {
            "avg_detection_confidence": float(np.mean(det_confs)),
            "avg_classification_confidence": float(np.mean(cls_confs)),
            "min_detection_confidence": float(np.min(det_confs)),
            "min_classification_confidence": float(np.min(cls_confs)),
            "combined_confidence": float(np.mean(combined)),
        }

    def _filter_results(
        self,
        detections: list[DetectionResult],
        classifications: list[tuple[DetectionResult, ClassificationResult]],
    ) -> tuple[list[DetectionResult], list[tuple[DetectionResult, ClassificationResult]]]:
        """結果をフィルタリング（最適化版）"""
        threshold = self.batch_config.confidence_filter

        # NumPy配列でベクトル化フィルタリング
        if detections:
            det_confs = np.array([det.confidence for det in detections])
            det_mask = det_confs >= threshold
            filtered_detections = [
                det for det, keep in zip(detections, det_mask, strict=False) if keep
            ]
        else:
            filtered_detections = []

        if classifications:
            cls_mask = [
                det.confidence >= threshold and cls.confidence >= threshold
                for det, cls in classifications
            ]
            filtered_classifications = [
                cls for cls, keep in zip(classifications, cls_mask, strict=False) if keep
            ]
        else:
            filtered_classifications = []

        return filtered_detections, filtered_classifications

    def _test_batch_size(self, test_frames: list[np.ndarray]) -> bool:
        """バッチサイズのテスト実行"""
        try:
            # 検出テスト
            _ = self._batch_detect_tiles(test_frames[:2])
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            return False
        except Exception as e:
            logger.warning(f"バッチサイズテストエラー: {e}")
            return False

    def _update_global_stats(
        self, num_frames: int, results: list[PipelineResult], processing_time: float
    ) -> None:
        """グローバル統計を更新"""
        self.stats["frames_processed"] += num_frames
        self.stats["total_processing_time"] += processing_time

        total_detections = sum(len(r.detections) for r in results)
        total_classifications = sum(len(r.classifications) for r in results)

        self.stats["tiles_detected"] += total_detections
        self.stats["tiles_classified"] += total_classifications
        self.stats["batch_sizes_used"].append(self.batch_processor.batch_size)

        # メモリ使用状況をログ
        memory_stats = self.batch_processor.get_memory_stats()
        if memory_stats:
            logger.info(
                f"バッチ処理完了: frames={num_frames}, "
                f"batch_size={self.batch_processor.batch_size}, "
                f"memory_used={memory_stats.get('used_percent', 0):.1f}%, "
                f"time={processing_time:.2f}s"
            )

    def get_stats(self) -> dict[str, Any]:
        """統計情報を取得"""
        stats = self.stats.copy()
        if stats["frames_processed"] > 0:
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["frames_processed"]
            )
            stats["avg_detections_per_frame"] = stats["tiles_detected"] / stats["frames_processed"]
            stats["avg_batch_size"] = (
                np.mean(stats["batch_sizes_used"]) if stats["batch_sizes_used"] else 0
            )

        return stats
