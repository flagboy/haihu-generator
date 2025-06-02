"""
動画処理のオーケストレーションを担当するクラス
"""

import time
from dataclasses import dataclass
from typing import Any

from ..pipeline.ai_pipeline import AIPipeline
from ..pipeline.game_pipeline import GamePipeline
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..video.video_processor import VideoProcessor


@dataclass
class ProcessingOptions:
    """処理オプションを格納するデータクラス"""

    enable_optimization: bool = True
    enable_validation: bool = True
    enable_gpu: bool = True
    batch_size: int | None = None
    max_workers: int | None = None


@dataclass
class ProcessingResult:
    """処理結果を格納するデータクラス"""

    success: bool
    video_path: str
    output_path: str | None = None
    frame_count: int = 0
    detected_tiles: int = 0
    processing_time: float = 0.0
    statistics: dict[str, Any] = None
    errors: list[str] = None
    warnings: list[str] = None
    game_data: Any | None = None

    def __post_init__(self):
        if self.statistics is None:
            self.statistics = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class VideoProcessingOrchestrator:
    """動画処理のオーケストレーションを行うクラス"""

    def __init__(
        self,
        config_manager: ConfigManager,
        video_processor: VideoProcessor,
        ai_pipeline: AIPipeline,
        game_pipeline: GamePipeline,
    ):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
            video_processor: 動画処理オブジェクト
            ai_pipeline: AI処理パイプライン
            game_pipeline: ゲーム処理パイプライン
        """
        self.config_manager = config_manager
        self.video_processor = video_processor
        self.ai_pipeline = ai_pipeline
        self.game_pipeline = game_pipeline
        self.logger = get_logger(self.__class__.__name__)

        # 設定の読み込み
        self.config = config_manager._config
        self.system_config = self.config.get("system", {})
        self.performance_config = self.config.get("performance", {})

    def process_video(self, video_path: str, options: ProcessingOptions) -> ProcessingResult:
        """
        動画を処理する

        Args:
            video_path: 入力動画のパス
            options: 処理オプション

        Returns:
            処理結果
        """
        start_time = time.time()

        try:
            # 1. 動画からフレームを抽出
            self.logger.info(f"フレーム抽出を開始: {video_path}")
            extraction_result = self.video_processor.extract_frames(video_path)

            if not extraction_result["success"]:
                return ProcessingResult(
                    success=False,
                    video_path=video_path,
                    errors=[
                        f"フレーム抽出に失敗: {extraction_result.get('error', 'Unknown error')}"
                    ],
                )

            frames = extraction_result["frames"]
            frame_count = len(frames)
            self.logger.info(f"抽出されたフレーム数: {frame_count}")

            # 2. AI処理（検出・分類）
            self.logger.info("AI処理を開始")
            ai_results = self._process_frames_with_ai(frames, options)

            # 3. ゲーム状態の追跡
            self.logger.info("ゲーム状態追跡を開始")
            game_results = self._process_game_state(ai_results, options)

            # 処理時間の計算
            processing_time = time.time() - start_time

            # 結果の集計
            return ProcessingResult(
                success=True,
                video_path=video_path,
                frame_count=frame_count,
                detected_tiles=self._count_detected_tiles(ai_results),
                processing_time=processing_time,
                game_data=game_results,
                statistics=self._collect_statistics(ai_results, game_results),
            )

        except Exception as e:
            self.logger.error(f"動画処理中にエラーが発生: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                video_path=video_path,
                errors=[str(e)],
                processing_time=time.time() - start_time,
            )

    def _process_frames_with_ai(
        self, frames: list[Any], options: ProcessingOptions
    ) -> list[dict[str, Any]]:
        """
        フレームをAIで処理

        Args:
            frames: フレームリスト
            options: 処理オプション

        Returns:
            AI処理結果のリスト
        """
        batch_size = options.batch_size or self.system_config.get("constants", {}).get(
            "default_batch_size", 32
        )
        results = []

        # バッチ処理
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_results = self.ai_pipeline.process_frames_batch(batch_frames, batch_start_frame=i)
            results.extend(batch_results.frame_results)

        return results

    def _process_game_state(
        self, ai_results: list[dict[str, Any]], options: ProcessingOptions
    ) -> Any:
        """
        ゲーム状態を処理

        Args:
            ai_results: AI処理結果
            options: 処理オプション

        Returns:
            ゲーム処理結果
        """
        # AI結果をゲームデータに変換
        game_data_list = []
        for result in ai_results:
            if result.get("classifications"):
                game_data = self._convert_ai_to_game_data(result)
                game_data_list.append(game_data)

        # ゲームパイプラインで処理
        return self.game_pipeline.process_game_data(game_data_list)

    def _convert_ai_to_game_data(self, ai_result: dict[str, Any]) -> dict[str, Any]:
        """
        AI結果をゲームデータ形式に変換

        Args:
            ai_result: AI処理結果

        Returns:
            ゲームデータ
        """
        game_data = {
            "frame_id": ai_result.get("frame_id"),
            "timestamp": ai_result.get("timestamp"),
            "tiles": [],
        }

        for det, cls in ai_result.get("classifications", []):
            tile_data = {
                "type": cls.label,
                "confidence": cls.confidence,
                "position": det.bbox,
                "area": self._classify_tile_area(det.bbox),
            }
            game_data["tiles"].append(tile_data)

        return game_data

    def _classify_tile_area(self, bbox: list[int]) -> str:
        """
        牌の位置から領域を分類

        Args:
            bbox: バウンディングボックス

        Returns:
            領域名
        """
        # 簡略化された領域分類
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2

        # 画面を9分割して判定
        if y_center < 360:
            return "player_top"
        elif y_center > 720:
            return "player_bottom"
        elif x_center < 640:
            return "player_left"
        elif x_center > 1280:
            return "player_right"
        else:
            return "center"

    def _count_detected_tiles(self, ai_results: list[dict[str, Any]]) -> int:
        """検出された牌の総数をカウント"""
        count = 0
        for result in ai_results:
            count += len(result.get("classifications", []))
        return count

    def _collect_statistics(
        self, ai_results: list[dict[str, Any]], game_results: Any
    ) -> dict[str, Any]:
        """
        統計情報を収集

        Args:
            ai_results: AI処理結果
            game_results: ゲーム処理結果

        Returns:
            統計情報
        """
        stats = {
            "total_frames": len(ai_results),
            "frames_with_detections": sum(1 for r in ai_results if r.get("classifications")),
            "total_detections": sum(len(r.get("classifications", [])) for r in ai_results),
            "average_confidence": self._calculate_average_confidence(ai_results),
            "processing_breakdown": {
                "video_extraction": "N/A",  # 実際の値は別途設定
                "ai_processing": "N/A",
                "game_tracking": "N/A",
            },
        }

        # ゲーム統計を追加
        if hasattr(game_results, "get_statistics"):
            stats["game_statistics"] = game_results.get_statistics()

        return stats

    def _calculate_average_confidence(self, ai_results: list[dict[str, Any]]) -> float:
        """平均信頼度を計算"""
        confidences = []
        for result in ai_results:
            for _, cls in result.get("classifications", []):
                confidences.append(cls.confidence)

        return sum(confidences) / len(confidences) if confidences else 0.0
