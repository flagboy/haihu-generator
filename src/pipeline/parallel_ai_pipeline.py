"""
並列処理に最適化されたAIパイプライン
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from .ai_pipeline import AIPipeline


@dataclass
class BatchPipelineResult:
    """バッチパイプライン処理結果を格納するデータクラス"""

    success: bool
    frame_results: list[dict[str, Any]]
    total_frames: int
    total_detections: int
    total_classifications: int
    processing_time: float
    average_confidence: float
    confidence_distribution: dict[str, Any]


class ParallelAIPipeline(AIPipeline):
    """並列処理に最適化されたAIパイプライン"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        super().__init__(config_manager)

        # 設定を取得
        config = self.config.get_config()
        self.system_config = config.get("system", {})
        self.ai_config = config.get("ai", {})
        self.performance_config = config.get("performance", {}).get("processing", {})

        # 並列処理設定
        self.max_workers = self.system_config.get("max_workers", mp.cpu_count())
        self.use_gpu_parallel = self.ai_config.get("enable_gpu_parallel", False)
        self.parallel_batch_size = self.performance_config.get("parallel_batch_size", 16)

        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"ParallelAIPipeline initialized with {self.max_workers} workers")

    def process_frames_parallel_batches(
        self, frames: list[np.ndarray], batch_start_frame: int = 0
    ) -> BatchPipelineResult:
        """
        フレームを並列バッチ処理

        Args:
            frames: 処理するフレームのリスト
            batch_start_frame: バッチの開始フレーム番号

        Returns:
            パイプライン処理結果
        """
        start_time = time.time()

        try:
            # バッチを作成
            batches = self._create_parallel_batches(frames)
            self.logger.info(f"Processing {len(frames)} frames in {len(batches)} parallel batches")

            # 並列処理エグゼキュータの選択
            executor_class = (
                ProcessPoolExecutor if not self.use_gpu_parallel else ThreadPoolExecutor
            )

            with executor_class(max_workers=self.max_workers) as executor:
                # バッチを並列で処理
                futures = []
                for idx, batch in enumerate(batches):
                    batch_offset = idx * self.parallel_batch_size
                    future = executor.submit(
                        self._process_batch_wrapper, batch, batch_start_frame + batch_offset
                    )
                    futures.append((future, batch_offset))

                # 結果を収集
                all_results = []
                total_detections = 0
                total_classifications = 0

                for future, offset in futures:
                    try:
                        batch_result = future.result(timeout=60)  # 60秒のタイムアウト
                        all_results.extend(batch_result.frame_results)
                        total_detections += batch_result.total_detections
                        total_classifications += batch_result.total_classifications
                    except Exception as e:
                        self.logger.error(f"Batch processing failed at offset {offset}: {e}")
                        # 失敗したバッチの分だけ空の結果を追加
                        for i in range(self.parallel_batch_size):
                            if batch_start_frame + offset + i < batch_start_frame + len(frames):
                                all_results.append(
                                    self._create_empty_frame_result(batch_start_frame + offset + i)
                                )

            # 結果をフレーム番号順にソート
            all_results.sort(key=lambda r: r.get("frame_id", 0))

            processing_time = time.time() - start_time

            return BatchPipelineResult(
                success=True,
                frame_results=all_results,
                total_frames=len(frames),
                total_detections=total_detections,
                total_classifications=total_classifications,
                processing_time=processing_time,
                average_confidence=self._calculate_average_confidence(all_results),
                confidence_distribution=self._calculate_confidence_distribution(all_results),
            )

        except Exception as e:
            self.logger.error(f"Parallel batch processing failed: {e}")
            return BatchPipelineResult(
                success=False,
                frame_results=[],
                total_frames=len(frames),
                total_detections=0,
                total_classifications=0,
                processing_time=time.time() - start_time,
                average_confidence=0.0,
                confidence_distribution={},
            )

    def _create_parallel_batches(self, frames: list[np.ndarray]) -> list[list[np.ndarray]]:
        """
        フレームを並列処理用のバッチに分割

        Args:
            frames: フレームリスト

        Returns:
            バッチのリスト
        """
        batches = []
        for i in range(0, len(frames), self.parallel_batch_size):
            batch = frames[i : i + self.parallel_batch_size]
            batches.append(batch)

        return batches

    def _process_batch_wrapper(
        self, batch: list[np.ndarray], batch_start_frame: int
    ) -> BatchPipelineResult:
        """
        バッチ処理のラッパー（プロセス間での実行用）

        Args:
            batch: フレームのバッチ
            batch_start_frame: バッチの開始フレーム番号

        Returns:
            バッチ処理結果
        """
        # 各プロセスで独自のロガーを設定
        self.logger = get_logger(f"{self.__class__.__name__}_worker")

        try:
            # 親クラスのバッチ処理を呼び出し
            results = self.process_frames_batch(batch, batch_start_frame)

            # PipelineResult のリストを BatchPipelineResult に変換
            frame_results = []
            total_detections = 0
            total_classifications = 0

            for result in results:
                frame_data = {
                    "frame_id": result.frame_id,
                    "detections": result.detections,
                    "classifications": result.classifications,
                    "processing_time": result.processing_time,
                    "tile_areas": result.tile_areas,
                    "confidence_scores": result.confidence_scores,
                }
                frame_results.append(frame_data)
                total_detections += len(result.detections)
                total_classifications += len(result.classifications)

            return BatchPipelineResult(
                success=True,
                frame_results=frame_results,
                total_frames=len(batch),
                total_detections=total_detections,
                total_classifications=total_classifications,
                processing_time=sum(r.processing_time for r in results),
                average_confidence=self._calculate_average_confidence(frame_results),
                confidence_distribution=self._calculate_confidence_distribution(frame_results),
            )

        except Exception as e:
            self.logger.error(f"Batch wrapper processing failed: {e}")
            # 失敗時は空の結果を返す
            empty_results = [
                self._create_empty_frame_result(batch_start_frame + i) for i in range(len(batch))
            ]
            return BatchPipelineResult(
                success=False,
                frame_results=empty_results,
                total_frames=len(batch),
                total_detections=0,
                total_classifications=0,
                processing_time=0.0,
                average_confidence=0.0,
                confidence_distribution={},
            )

    def _create_empty_frame_result(self, frame_id: int) -> dict[str, Any]:
        """
        空のフレーム結果を作成

        Args:
            frame_id: フレームID

        Returns:
            空のフレーム結果
        """
        return {
            "frame_id": frame_id,
            "detections": [],
            "classifications": [],
            "tile_areas": {},
            "confidence_scores": {},
            "processing_time": 0.0,
            "error": "Processing failed",
        }

    def process_frames_with_prefetch(
        self, frames: list[np.ndarray], prefetch_size: int = 2
    ) -> BatchPipelineResult:
        """
        プリフェッチを使用した非同期処理

        Args:
            frames: 処理するフレームのリスト
            prefetch_size: プリフェッチするバッチ数

        Returns:
            パイプライン処理結果
        """
        start_time = time.time()

        try:
            batches = self._create_parallel_batches(frames)

            with ThreadPoolExecutor(max_workers=min(prefetch_size, len(batches))) as executor:
                # 最初のバッチをプリフェッチ
                futures = []
                for i in range(min(prefetch_size, len(batches))):
                    future = executor.submit(
                        self._prefetch_and_process_batch, batches[i], i * self.parallel_batch_size
                    )
                    futures.append(future)

                # 結果を収集しながら新しいバッチをキューに追加
                all_results = []
                batch_index = prefetch_size

                while futures:
                    # 完了したFutureを取得
                    done_future = futures.pop(0)
                    result = done_future.result()
                    all_results.extend(result.frame_results)

                    # 次のバッチがあればキューに追加
                    if batch_index < len(batches):
                        future = executor.submit(
                            self._prefetch_and_process_batch,
                            batches[batch_index],
                            batch_index * self.parallel_batch_size,
                        )
                        futures.append(future)
                        batch_index += 1

            # 結果をソート
            all_results.sort(key=lambda r: r.get("frame_id", 0))

            return BatchPipelineResult(
                success=True,
                frame_results=all_results,
                total_frames=len(frames),
                total_detections=sum(len(r.get("detections", [])) for r in all_results),
                total_classifications=sum(len(r.get("classifications", [])) for r in all_results),
                processing_time=time.time() - start_time,
                average_confidence=self._calculate_average_confidence(all_results),
                confidence_distribution=self._calculate_confidence_distribution(all_results),
            )

        except Exception as e:
            self.logger.error(f"Prefetch processing failed: {e}")
            return BatchPipelineResult(
                success=False,
                frame_results=[],
                total_frames=len(frames),
                total_detections=0,
                total_classifications=0,
                processing_time=time.time() - start_time,
                average_confidence=0.0,
                confidence_distribution={},
            )

    def _prefetch_and_process_batch(
        self, batch: list[np.ndarray], batch_start_frame: int
    ) -> BatchPipelineResult:
        """
        バッチのプリフェッチと処理

        Args:
            batch: フレームのバッチ
            batch_start_frame: バッチの開始フレーム番号

        Returns:
            バッチ処理結果
        """
        # データのプリフェッチ（例：前処理）
        preprocessed_batch = []
        for frame in batch:
            preprocessed = self._preprocess_frame_optimized(frame)
            preprocessed_batch.append(preprocessed)

        # 処理を実行して結果を変換
        results = self.process_frames_batch(preprocessed_batch, batch_start_frame)

        # PipelineResult のリストを BatchPipelineResult に変換
        frame_results = []
        total_detections = 0
        total_classifications = 0

        for result in results:
            frame_data = {
                "frame_id": result.frame_id,
                "detections": result.detections,
                "classifications": result.classifications,
                "processing_time": result.processing_time,
                "tile_areas": result.tile_areas,
                "confidence_scores": result.confidence_scores,
            }
            frame_results.append(frame_data)
            total_detections += len(result.detections)
            total_classifications += len(result.classifications)

        return BatchPipelineResult(
            success=True,
            frame_results=frame_results,
            total_frames=len(batch),
            total_detections=total_detections,
            total_classifications=total_classifications,
            processing_time=sum(r.processing_time for r in results),
            average_confidence=self._calculate_average_confidence(frame_results),
            confidence_distribution=self._calculate_confidence_distribution(frame_results),
        )

    def _preprocess_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """
        最適化されたフレーム前処理

        Args:
            frame: 入力フレーム

        Returns:
            前処理済みフレーム
        """
        # NumPyの最適化された操作を使用
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # 必要に応じてリサイズ（最適化版）
        if hasattr(self, "target_size") and self.target_size:
            import cv2

            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        return frame

    def _calculate_average_confidence(self, frame_results: list[dict[str, Any]]) -> float:
        """
        平均信頼度を計算

        Args:
            frame_results: フレーム処理結果のリスト

        Returns:
            平均信頼度
        """
        all_confidences = []

        for result in frame_results:
            classifications = result.get("classifications", [])
            for classification_pair in classifications:
                if hasattr(classification_pair, "__iter__") and len(classification_pair) >= 2:
                    # (detection, classification) のタプル形式
                    _, classification = classification_pair
                    if hasattr(classification, "confidence"):
                        all_confidences.append(classification.confidence)

        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    def _calculate_confidence_distribution(
        self, frame_results: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        信頼度分布を計算

        Args:
            frame_results: フレーム処理結果のリスト

        Returns:
            信頼度分布
        """
        distribution = {"high": 0, "medium": 0, "low": 0}

        for result in frame_results:
            classifications = result.get("classifications", [])
            for classification_pair in classifications:
                if hasattr(classification_pair, "__iter__") and len(classification_pair) >= 2:
                    # (detection, classification) のタプル形式
                    _, classification = classification_pair
                    if hasattr(classification, "confidence"):
                        confidence = classification.confidence
                        if confidence >= 0.8:
                            distribution["high"] += 1
                        elif confidence >= 0.5:
                            distribution["medium"] += 1
                        else:
                            distribution["low"] += 1

        return distribution
