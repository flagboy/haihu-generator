"""
スマートフレーム選択システム

効率的な教師データ作成のために、重要なフレームを自動選択し、
冗長なフレームをスキップする機能を提供
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..detection.scene_detector import SceneType
from ..utils.logger import LoggerMixin
from .frame_similarity_detector import FrameSimilarityDetector


@dataclass
class FrameImportance:
    """フレーム重要度情報"""

    frame_number: int
    importance_score: float
    reasons: list[str]
    is_key_frame: bool
    should_annotate: bool
    metadata: dict[str, Any]


class SmartFrameSelector(LoggerMixin):
    """スマートフレーム選択クラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}

        # 類似度検出器
        self.similarity_detector = FrameSimilarityDetector(
            self.config.get("similarity_detection", {})
        )

        # 選択戦略設定
        self.min_frame_interval = self.config.get("min_frame_interval", 5)  # 最小フレーム間隔
        self.max_frame_interval = self.config.get("max_frame_interval", 30)  # 最大フレーム間隔
        self.scene_change_threshold = self.config.get("scene_change_threshold", 0.3)
        self.motion_threshold = self.config.get("motion_threshold", 0.1)

        # 重要シーンの設定
        self.important_scenes = self.config.get(
            "important_scenes",
            [
                SceneType.GAME_START.value,
                SceneType.ROUND_START.value,
                SceneType.ROUND_END.value,
                SceneType.DORA_INDICATOR.value,
                SceneType.RIICHI.value,
                SceneType.TSUMO.value,
                SceneType.RON.value,
            ],
        )

        # キャッシュ
        self.frame_cache = []
        self.last_selected_frame = None
        self.frames_since_last_selection = 0

        self.logger.info("SmartFrameSelector初期化完了")

    def analyze_frame_importance(
        self,
        frame: np.ndarray,
        frame_number: int,
        scene_type: str | None = None,
        previous_frame: np.ndarray | None = None,
    ) -> FrameImportance:
        """
        フレームの重要度を分析

        Args:
            frame: 分析対象のフレーム
            frame_number: フレーム番号
            scene_type: シーンタイプ
            previous_frame: 前のフレーム

        Returns:
            フレーム重要度情報
        """
        importance_score = 0.0
        reasons = []
        is_key_frame = False

        # 1. シーンタイプによる重要度
        if scene_type in self.important_scenes:
            importance_score += 0.5
            reasons.append(f"重要シーン: {scene_type}")
            is_key_frame = True

        # 2. シーン変化の検出
        if previous_frame is not None:
            scene_change = self._detect_scene_change(previous_frame, frame)
            if scene_change > self.scene_change_threshold:
                importance_score += 0.3
                reasons.append(f"シーン変化検出: {scene_change:.2f}")
                is_key_frame = True

        # 3. モーション量の検出
        if previous_frame is not None:
            motion_amount = self._calculate_motion(previous_frame, frame)
            if motion_amount > self.motion_threshold:
                importance_score += 0.2
                reasons.append(f"高モーション: {motion_amount:.2f}")

        # 4. フレーム間隔による調整
        self.frames_since_last_selection += 1

        if self.frames_since_last_selection >= self.max_frame_interval:
            importance_score += 0.3
            reasons.append("最大間隔到達")
        elif self.frames_since_last_selection < self.min_frame_interval:
            importance_score -= 0.5
            reasons.append("最小間隔未満")

        # 5. 最終判定
        should_annotate = importance_score >= 0.5 or is_key_frame

        # 選択された場合はカウンタリセット
        if should_annotate:
            self.frames_since_last_selection = 0
            self.last_selected_frame = frame_number

        return FrameImportance(
            frame_number=frame_number,
            importance_score=min(importance_score, 1.0),
            reasons=reasons,
            is_key_frame=is_key_frame,
            should_annotate=should_annotate,
            metadata={
                "frames_since_last": self.frames_since_last_selection,
                "scene_type": scene_type,
            },
        )

    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """シーン変化を検出"""
        # ヒストグラム差分
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Bhattacharyya距離
        distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        return distance

    def _calculate_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """フレーム間のモーション量を計算"""
        # グレースケール変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # オプティカルフロー計算（簡易版）
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # モーション量の計算
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_amount = np.mean(magnitude)

        return motion_amount / 10.0  # 正規化

    def select_frames_from_video(
        self, video_path: str, target_frames: int | None = None, analyze_first: bool = True
    ) -> list[int]:
        """
        動画から効率的にフレームを選択

        Args:
            video_path: 動画パス
            target_frames: 目標フレーム数（Noneの場合は自動）
            analyze_first: 最初に全体を分析するか

        Returns:
            選択されたフレーム番号のリスト
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)  # TODO: 実装時に使用

        selected_frames = []

        if analyze_first:
            # 第1パス: 高速分析
            self.logger.info("第1パス: フレーム分析中...")
            importance_scores = self._analyze_video_pass(cap, total_frames)

            # 重要度に基づいてフレーム選択
            if target_frames:
                selected_frames = self._select_top_frames(importance_scores, target_frames)
            else:
                selected_frames = self._select_adaptive_frames(importance_scores)
        else:
            # リアルタイム選択
            selected_frames = self._select_realtime_frames(cap, total_frames, target_frames)

        cap.release()

        self.logger.info(
            f"フレーム選択完了: {len(selected_frames)}/{total_frames}フレーム "
            f"(選択率: {len(selected_frames) / total_frames * 100:.1f}%)"
        )

        return sorted(selected_frames)

    def _analyze_video_pass(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> list[FrameImportance]:
        """動画全体を分析"""
        importance_scores = []
        previous_frame = None

        # サンプリング間隔（高速化のため）
        sample_interval = max(1, total_frames // 1000)

        frame_number = 0
        while frame_number < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                break

            # 重要度分析
            importance = self.analyze_frame_importance(
                frame,
                frame_number,
                None,  # シーンタイプは別途検出
                previous_frame,
            )

            importance_scores.append(importance)
            previous_frame = frame

            frame_number += sample_interval

            if frame_number % 100 == 0:
                progress = frame_number / total_frames * 100
                self.logger.debug(f"分析進捗: {progress:.1f}%")

        return importance_scores

    def _select_top_frames(
        self, importance_scores: list[FrameImportance], target_count: int
    ) -> list[int]:
        """重要度の高いフレームを選択"""
        # 重要度でソート
        sorted_frames = sorted(importance_scores, key=lambda x: x.importance_score, reverse=True)

        # 上位N個を選択
        selected = []
        for frame_info in sorted_frames[:target_count]:
            selected.append(frame_info.frame_number)

        return selected

    def _select_adaptive_frames(self, importance_scores: list[FrameImportance]) -> list[int]:
        """適応的にフレームを選択"""
        selected = []

        for frame_info in importance_scores:
            if frame_info.should_annotate:
                selected.append(frame_info.frame_number)

        return selected

    def _select_realtime_frames(
        self, cap: cv2.VideoCapture, total_frames: int, target_frames: int | None
    ) -> list[int]:
        """リアルタイムでフレームを選択"""
        selected = []
        previous_frame = None

        # base_interval = total_frames / target_frames if target_frames else 30  # TODO: 実装時に使用

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 重要度分析
            importance = self.analyze_frame_importance(frame, frame_number, None, previous_frame)

            # 選択判定
            if importance.should_annotate:
                selected.append(frame_number)

            previous_frame = frame
            frame_number += 1

            # 目標数に達したら調整
            if target_frames and len(selected) >= target_frames:
                break

        return selected

    def get_statistics(self) -> dict[str, Any]:
        """選択統計を取得"""
        return {
            "total_analyzed": len(self.frame_cache),
            "last_selected_frame": self.last_selected_frame,
            "frames_since_last_selection": self.frames_since_last_selection,
            "min_interval": self.min_frame_interval,
            "max_interval": self.max_frame_interval,
        }
