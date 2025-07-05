"""
バッチ処理対応の拡張ゲームパイプライン

複数フレームの並列処理とキャッシュ機能により
パフォーマンスを最適化
"""

import concurrent.futures
from typing import Any

import numpy as np

from ..detection.cached_scene_detector import CachedSceneDetector
from .enhanced_game_pipeline import EnhancedGamePipeline, EnhancedProcessingResult, ProcessingResult


class BatchEnhancedGamePipeline(EnhancedGamePipeline):
    """バッチ処理対応の拡張ゲームパイプライン"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        # 設定を保存
        self.config = config or {}

        # 拡張機能の設定
        enable_scene = self.config.get("enable_scene_detection", True)
        enable_score = self.config.get("enable_score_reading", True)
        enable_player = self.config.get("enable_player_detection", True)

        # 親クラスの初期化（game_idを渡す）
        super().__init__(
            game_id="batch_processing",
            enable_scene_detection=enable_scene,
            enable_score_reading=enable_score,
            enable_player_detection=enable_player,
        )

        # バッチ処理設定
        batch_config = self.config.get("batch_processing", {})
        self.batch_size = batch_config.get("batch_size", 10)
        self.max_workers = batch_config.get("max_workers", 4)
        self.enable_parallel = batch_config.get("enable_parallel", True)

        # キャッシュ付きシーン検出器に置き換え
        if enable_scene:
            self.scene_detector = CachedSceneDetector(self.config.get("scene_detection"))

        self.logger.info(
            f"BatchEnhancedGamePipeline初期化完了 "
            f"(batch_size: {self.batch_size}, max_workers: {self.max_workers})"
        )

    def process_frame(self, frame_data: dict[str, Any]) -> ProcessingResult:
        """
        フレームデータを処理（親クラスのインターフェース）

        Args:
            frame_data: フレーム検出データ

        Returns:
            ProcessingResult: 処理結果
        """
        # 親クラスのメソッドを呼び出す
        return super().process_frame(frame_data)

    def process_frame_batch(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> EnhancedProcessingResult:
        """
        単一フレームを処理（バッチ処理用）

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            拡張処理結果
        """
        return self.process_frame_enhanced(frame, frame_number, timestamp)

    def process_frames_batch(
        self, frames: list[tuple[np.ndarray, int, float]]
    ) -> list[EnhancedProcessingResult]:
        """
        複数フレームをバッチ処理

        Args:
            frames: (フレーム, フレーム番号, タイムスタンプ)のリスト

        Returns:
            処理結果のリスト
        """
        if not self.enable_parallel or len(frames) <= 1:
            # 並列処理無効または単一フレームの場合は逐次処理
            return [
                self.process_frame_batch(frame, frame_num, ts) for frame, frame_num, ts in frames
            ]

        results = []

        # バッチごとに処理
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            batch_results = self._process_batch_parallel(batch)
            results.extend(batch_results)

            # 進捗ログ
            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                self.logger.info(
                    f"バッチ処理進捗: {min(i + self.batch_size, len(frames))}/{len(frames)}"
                )

        return results

    def _process_batch_parallel(
        self, batch: list[tuple[np.ndarray, int, float]]
    ) -> list[EnhancedProcessingResult]:
        """バッチを並列処理"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 非同期タスクを作成
            future_to_index = {
                executor.submit(self.process_frame_batch, frame, frame_num, ts): i
                for i, (frame, frame_num, ts) in enumerate(batch)
            }

            # 結果を収集（インデックス順を保持）
            indexed_results: list[tuple[int, EnhancedProcessingResult]] = []
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    indexed_results.append((index, result))
                except Exception as e:
                    self.logger.error(f"フレーム処理エラー (index: {index}): {e}")
                    # エラーの場合は空の結果を作成
                    frame, frame_num, ts = batch[index]
                    error_result = self._create_error_result(frame_num, ts, str(e))
                    indexed_results.append((index, error_result))

        # インデックス順にソートして結果を返す
        indexed_results.sort(key=lambda x: x[0])
        return [result for _, result in indexed_results]

    def optimize_for_video(self, video_path: str):
        """
        特定の動画に対して最適化

        Args:
            video_path: 動画ファイルパス
        """
        self.logger.info(f"動画に対する最適化開始: {video_path}")

        # シーン境界を事前に検出してキャッシュ
        if self.scene_detector and hasattr(self.scene_detector, "detect_game_boundaries"):
            boundaries = self.scene_detector.detect_game_boundaries(video_path)
            self.logger.info(f"ゲーム境界を{len(boundaries)}個検出")

        # その他の最適化処理
        # - モデルのウォームアップ
        # - メモリの事前確保
        # など

    def clear_caches(self):
        """全てのキャッシュをクリア"""
        if self.scene_detector and hasattr(self.scene_detector, "clear_cache"):
            self.scene_detector.clear_cache()

        # その他のキャッシュもクリア
        self.logger.info("全てのキャッシュをクリアしました")

    def get_performance_stats(self) -> dict[str, Any]:
        """パフォーマンス統計を取得"""
        stats = {
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "parallel_enabled": self.enable_parallel,
        }

        # キャッシュ統計
        if self.scene_detector and hasattr(self.scene_detector, "_frame_hash_cache"):
            stats["scene_cache_size"] = len(self.scene_detector._frame_hash_cache)

        return stats

    def _create_error_result(
        self, frame_number: int, timestamp: float, error_message: str
    ) -> EnhancedProcessingResult:
        """エラー結果を作成"""
        result = EnhancedProcessingResult(
            success=False,
            frame_number=frame_number,
            actions_detected=0,
            confidence=0.0,
            processing_time=0.0,
        )
        result.errors.append(error_message)
        return result
