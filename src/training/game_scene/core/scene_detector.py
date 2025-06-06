"""
シーン検出器

動画から対局シーンのセグメントを検出する
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ....utils.logger import LoggerMixin
from .game_scene_classifier import GameSceneClassifier


@dataclass
class SceneSegment:
    """シーンセグメント"""

    start_frame: int
    end_frame: int
    confidence: float
    scene_type: str = "game"  # "game" or "non_game"

    @property
    def duration_frames(self) -> int:
        """フレーム数"""
        return self.end_frame - self.start_frame + 1


class SceneDetector(LoggerMixin):
    """シーン検出器"""

    def __init__(
        self,
        classifier: GameSceneClassifier | None = None,
        min_scene_duration: int = 30,  # 最小シーン長（フレーム数）
        confidence_threshold: float = 0.8,
        smoothing_window: int = 5,
    ):
        """
        初期化

        Args:
            classifier: 対局画面分類器
            min_scene_duration: 最小シーン長
            confidence_threshold: 信頼度閾値
            smoothing_window: スムージングウィンドウサイズ
        """
        self.classifier = classifier or GameSceneClassifier()
        self.min_scene_duration = min_scene_duration
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        self.logger.info("SceneDetector初期化完了")

    def detect_scenes(
        self,
        video_path: str,
        sample_interval: int = 30,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[SceneSegment]:
        """
        動画から対局シーンを検出

        Args:
            video_path: 動画ファイルパス
            sample_interval: サンプリング間隔（フレーム数）
            progress_callback: 進捗コールバック

        Returns:
            シーンセグメントのリスト
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"動画を開けません: {video_path}")
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(
                f"動画解析開始: {Path(video_path).name} "
                f"(総フレーム数: {total_frames}, FPS: {fps:.1f})"
            )

            # フレームごとの分類結果
            frame_classifications = []
            frame_numbers = []

            # サンプリングしながら分類
            for frame_idx in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # 分類
                is_game, confidence = self.classifier.classify_frame(frame)
                frame_classifications.append((is_game, confidence))
                frame_numbers.append(frame_idx)

                # 進捗通知
                if progress_callback:
                    progress = frame_idx / total_frames
                    progress_callback(progress)

            # 分類結果をスムージング
            smoothed_classifications = self._smooth_classifications(
                frame_classifications, frame_numbers
            )

            # セグメント検出
            segments = self._detect_segments(smoothed_classifications, frame_numbers, total_frames)

            # 短いセグメントをフィルタリング
            segments = self._filter_short_segments(segments)

            self.logger.info(f"検出完了: {len(segments)}個の対局シーンを検出")

            return segments

        finally:
            cap.release()

    def detect_scenes_from_frames(
        self, frame_paths: list[str], frame_numbers: list[int] | None = None
    ) -> list[SceneSegment]:
        """
        抽出済みフレームから対局シーンを検出

        Args:
            frame_paths: フレーム画像のパスリスト
            frame_numbers: フレーム番号リスト（Noneの場合は連番）

        Returns:
            シーンセグメントのリスト
        """
        if frame_numbers is None:
            frame_numbers = list(range(len(frame_paths)))

        # バッチ分類
        classifications = self.classifier.classify_batch(frame_paths)

        # スムージング
        smoothed_classifications = self._smooth_classifications(classifications, frame_numbers)

        # セグメント検出
        segments = self._detect_segments(
            smoothed_classifications,
            frame_numbers,
            frame_numbers[-1] + 1 if frame_numbers else len(frame_paths),
        )

        # 短いセグメントをフィルタリング
        segments = self._filter_short_segments(segments)

        return segments

    def _smooth_classifications(
        self, classifications: list[tuple[bool, float]], frame_numbers: list[int]
    ) -> list[tuple[bool, float]]:
        """
        分類結果をスムージング

        Args:
            classifications: 分類結果のリスト
            frame_numbers: フレーム番号のリスト

        Returns:
            スムージング後の分類結果
        """
        if len(classifications) <= self.smoothing_window:
            return classifications

        smoothed = []
        half_window = self.smoothing_window // 2

        for i in range(len(classifications)):
            # ウィンドウ内の分類結果を取得
            start_idx = max(0, i - half_window)
            end_idx = min(len(classifications), i + half_window + 1)

            window_classifications = classifications[start_idx:end_idx]

            # 信頼度の平均
            avg_confidence = np.mean([c[1] for c in window_classifications])

            # 多数決
            game_count = sum(1 for c in window_classifications if c[0])
            is_game = game_count > len(window_classifications) / 2

            smoothed.append((is_game, avg_confidence))

        return smoothed

    def _detect_segments(
        self, classifications: list[tuple[bool, float]], frame_numbers: list[int], total_frames: int
    ) -> list[SceneSegment]:
        """
        連続する同じ分類結果からセグメントを検出

        Args:
            classifications: 分類結果のリスト
            frame_numbers: フレーム番号のリスト
            total_frames: 総フレーム数

        Returns:
            セグメントのリスト
        """
        if not classifications:
            return []

        segments = []
        current_segment_start = 0
        current_is_game = classifications[0][0]
        confidence_sum = classifications[0][1]

        for i in range(1, len(classifications)):
            if classifications[i][0] != current_is_game:
                # セグメント終了
                avg_confidence = confidence_sum / (i - current_segment_start)

                # フレーム番号を補間
                start_frame = frame_numbers[current_segment_start]
                end_frame = frame_numbers[i - 1] if i - 1 < len(frame_numbers) else total_frames - 1

                if i < len(frame_numbers) - 1:
                    # 次のサンプル点との中間まで
                    end_frame = (frame_numbers[i - 1] + frame_numbers[i]) // 2

                segment = SceneSegment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    confidence=avg_confidence,
                    scene_type="game" if current_is_game else "non_game",
                )
                segments.append(segment)

                # 新しいセグメント開始
                current_segment_start = i
                current_is_game = classifications[i][0]
                confidence_sum = classifications[i][1]
            else:
                confidence_sum += classifications[i][1]

        # 最後のセグメント
        if current_segment_start < len(classifications):
            avg_confidence = confidence_sum / (len(classifications) - current_segment_start)
            start_frame = frame_numbers[current_segment_start]
            end_frame = total_frames - 1

            segment = SceneSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=avg_confidence,
                scene_type="game" if current_is_game else "non_game",
            )
            segments.append(segment)

        return segments

    def _filter_short_segments(self, segments: list[SceneSegment]) -> list[SceneSegment]:
        """
        短いセグメントをフィルタリング

        Args:
            segments: セグメントのリスト

        Returns:
            フィルタリング後のセグメントリスト
        """
        filtered = []

        for segment in segments:
            if segment.duration_frames >= self.min_scene_duration:
                filtered.append(segment)
            else:
                self.logger.debug(
                    f"短いセグメントをスキップ: "
                    f"{segment.scene_type} "
                    f"({segment.start_frame}-{segment.end_frame}, "
                    f"{segment.duration_frames}フレーム)"
                )

        return filtered

    def merge_nearby_segments(
        self,
        segments: list[SceneSegment],
        max_gap: int = 150,  # 最大ギャップ（フレーム数）
    ) -> list[SceneSegment]:
        """
        近接する同じタイプのセグメントをマージ

        Args:
            segments: セグメントのリスト
            max_gap: マージする最大ギャップ

        Returns:
            マージ後のセグメントリスト
        """
        if not segments:
            return []

        # タイプとフレーム番号でソート
        sorted_segments = sorted(segments, key=lambda s: (s.scene_type, s.start_frame))

        merged = []
        current = sorted_segments[0]

        for segment in sorted_segments[1:]:
            if (
                segment.scene_type == current.scene_type
                and segment.start_frame - current.end_frame <= max_gap
            ):
                # マージ
                current = SceneSegment(
                    start_frame=current.start_frame,
                    end_frame=segment.end_frame,
                    confidence=(current.confidence + segment.confidence) / 2,
                    scene_type=current.scene_type,
                )
            else:
                merged.append(current)
                current = segment

        merged.append(current)

        return merged
