"""
教師データ作成用フレーム抽出システム

既存のVideoProcessorを拡張して、教師データ作成に特化したフレーム抽出機能を提供
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from ..video.video_processor import VideoProcessor
from .annotation_data import AnnotationData, FrameAnnotation


class FrameQualityAnalyzer:
    """フレーム品質評価クラス"""

    def __init__(self):
        """初期化"""
        pass

    def analyze_frame_quality(self, frame: np.ndarray) -> dict[str, float]:
        """
        フレームの品質を分析

        Args:
            frame: 入力フレーム

        Returns:
            品質スコア辞書
        """
        scores = {}

        # ブラー検出
        scores["sharpness"] = self._calculate_sharpness(frame)

        # 明度評価
        scores["brightness"] = self._calculate_brightness_score(frame)

        # コントラスト評価
        scores["contrast"] = self._calculate_contrast_score(frame)

        # ノイズ評価
        scores["noise"] = self._calculate_noise_score(frame)

        # 総合スコア
        scores["overall"] = self._calculate_overall_score(scores)

        return scores

    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """シャープネス（鮮明度）を計算"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 正規化（経験的な値）
        max_var = 2000.0
        return min(laplacian_var / max_var, 1.0)

    def _calculate_brightness_score(self, frame: np.ndarray) -> float:
        """明度スコアを計算"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # 適切な明度範囲（80-180）でスコア化
        if 80 <= mean_brightness <= 180:
            return 1.0
        elif mean_brightness < 80:
            return max(mean_brightness / 80.0, 0.1)
        else:  # > 180
            return max((255 - mean_brightness) / 75.0, 0.1)

    def _calculate_contrast_score(self, frame: np.ndarray) -> float:
        """コントラストスコアを計算"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()

        # 正規化（経験的な値）
        max_contrast = 80.0
        return min(contrast / max_contrast, 1.0)

    def _calculate_noise_score(self, frame: np.ndarray) -> float:
        """ノイズスコアを計算（低いほど良い）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ガウシアンフィルタを適用してノイズを推定
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))

        # 正規化（低いノイズほど高スコア）
        max_noise = 20.0
        return max(1.0 - (noise / max_noise), 0.0)

    def _calculate_overall_score(self, scores: dict[str, float]) -> float:
        """総合スコアを計算"""
        weights = {"sharpness": 0.3, "brightness": 0.2, "contrast": 0.2, "noise": 0.3}

        overall = 0.0
        for metric, weight in weights.items():
            if metric in scores:
                overall += scores[metric] * weight

        return overall


class FrameExtractor(LoggerMixin):
    """教師データ作成用フレーム抽出クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.video_processor = VideoProcessor(config_manager)
        self.quality_analyzer = FrameQualityAnalyzer()

        # 教師データ作成用設定
        training_config = self.config_manager.get_config().get("training", {})
        extraction_config = training_config.get("frame_extraction", {})

        self.min_quality_score = extraction_config.get("min_quality_score", 0.6)
        self.max_frames_per_video = extraction_config.get("max_frames_per_video", 1000)
        self.frame_interval_seconds = extraction_config.get("frame_interval_seconds", 2.0)
        self.diversity_threshold = extraction_config.get("diversity_threshold", 0.3)

        # 出力設定
        self.output_dir = Path(training_config.get("output_dir", "data/training/extracted_frames"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("FrameExtractor初期化完了")

    def extract_training_frames(
        self, video_path: str, annotation_data: AnnotationData | None = None
    ) -> list[FrameAnnotation]:
        """
        教師データ作成用フレームを抽出

        Args:
            video_path: 動画ファイルのパス
            annotation_data: 既存のアノテーションデータ（オプション）

        Returns:
            抽出されたフレームのアノテーション情報リスト
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        self.logger.info(f"教師データ用フレーム抽出開始: {video_path}")

        # 動画情報を取得
        video_info = self.video_processor.get_video_info(str(video_path))

        # 動画用出力ディレクトリを作成
        video_name = video_path.stem
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(exist_ok=True)

        # フレーム抽出
        extracted_frames = self._extract_frames_with_quality_filter(
            str(video_path), str(video_output_dir), video_info
        )

        # 多様性フィルタリング
        diverse_frames = self._apply_diversity_filter(extracted_frames)

        # フレーム数制限
        if len(diverse_frames) > self.max_frames_per_video:
            diverse_frames = self._select_best_frames(diverse_frames, self.max_frames_per_video)

        # FrameAnnotationオブジェクトを作成
        frame_annotations = []
        for frame_info in diverse_frames:
            frame_annotation = FrameAnnotation(
                frame_id=frame_info["frame_id"],
                image_path=frame_info["image_path"],
                image_width=frame_info["width"],
                image_height=frame_info["height"],
                timestamp=frame_info["timestamp"],
                tiles=[],  # 初期状態では空
                quality_score=frame_info["quality_score"],
                is_valid=True,
                scene_type="game",  # デフォルト
                game_phase="unknown",
                annotator="frame_extractor",
                notes=f"Quality scores: {frame_info['quality_details']}",
            )
            frame_annotations.append(frame_annotation)

        self.logger.info(f"フレーム抽出完了: {len(frame_annotations)}フレーム")
        return frame_annotations

    def _extract_frames_with_quality_filter(
        self, video_path: str, output_dir: str, video_info: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """品質フィルタリング付きフレーム抽出"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        try:
            fps = video_info.get("fps", 30.0)
            frame_interval = int(fps * self.frame_interval_seconds)

            extracted_frames = []
            frame_count = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    if frame_count == 0:
                        self.logger.error(
                            "最初のフレームも読み込めません。動画ファイルが破損している可能性があります。"
                        )
                    break

                # フレームの妥当性チェック
                if frame is None or frame.size == 0:
                    self.logger.warning(f"フレーム {frame_count} が無効です。スキップします。")
                    frame_count += 1
                    continue

                # 指定間隔でフレームを処理
                if frame_count % frame_interval == 0:
                    # 品質分析
                    quality_scores = self.quality_analyzer.analyze_frame_quality(frame)

                    # 品質閾値チェック
                    if quality_scores["overall"] >= self.min_quality_score:
                        # フレームを保存
                        timestamp = frame_count / fps
                        frame_id = f"{Path(video_path).stem}_{extracted_count:06d}"
                        filename = f"{frame_id}.jpg"
                        image_path = Path(output_dir) / filename

                        # 前処理してから保存
                        processed_frame = self.video_processor.preprocess_frame(frame)
                        cv2.imwrite(
                            str(image_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                        )

                        frame_info = {
                            "frame_id": frame_id,
                            "image_path": str(image_path),
                            "timestamp": timestamp,
                            "width": processed_frame.shape[1],
                            "height": processed_frame.shape[0],
                            "quality_score": quality_scores["overall"],
                            "quality_details": quality_scores,
                            "original_frame_index": frame_count,
                        }

                        extracted_frames.append(frame_info)
                        extracted_count += 1

                frame_count += 1

                # 進捗表示
                if frame_count % 1000 == 0:
                    self.logger.info(
                        f"フレーム処理進捗: {frame_count}フレーム処理, "
                        f"{extracted_count}フレーム抽出"
                    )

            return extracted_frames

        finally:
            cap.release()

    def _apply_diversity_filter(self, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """多様性フィルタリングを適用"""
        if len(frames) <= 1:
            return frames

        diverse_frames = [frames[0]]  # 最初のフレームは必ず含める

        for frame in frames[1:]:
            # 既に選択されたフレームとの類似度をチェック
            is_diverse = True
            current_image = cv2.imread(frame["image_path"])

            for selected_frame in diverse_frames[-5:]:  # 直近5フレームと比較
                selected_image = cv2.imread(selected_frame["image_path"])

                if self._calculate_frame_similarity(current_image, selected_image) > (
                    1.0 - self.diversity_threshold
                ):
                    is_diverse = False
                    break

            if is_diverse:
                diverse_frames.append(frame)

        self.logger.info(f"多様性フィルタリング: {len(frames)} -> {len(diverse_frames)}フレーム")
        return diverse_frames

    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """フレーム間の類似度を計算"""
        if frame1 is None or frame2 is None:
            return 0.0

        # フレームのサイズと型を確認
        if frame1.shape != frame2.shape:
            return 0.0

        # uint8型に変換（必要な場合）
        if frame1.dtype != np.uint8:
            frame1 = frame1.astype(np.uint8)
        if frame2.dtype != np.uint8:
            frame2 = frame2.astype(np.uint8)

        try:
            # 同じ画像の場合は1.0を返す
            if np.array_equal(frame1, frame2):
                return 1.0

            # 簡単な方法：正規化した画素値の差の平均を計算
            diff = np.abs(frame1.astype(float) - frame2.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            self.logger.error(f"フレーム類似度計算でエラー: {e}")
            return 0.0

    def _select_best_frames(
        self, frames: list[dict[str, Any]], max_count: int
    ) -> list[dict[str, Any]]:
        """品質スコアに基づいて最良のフレームを選択"""
        # 品質スコアでソート
        sorted_frames = sorted(frames, key=lambda x: x["quality_score"], reverse=True)

        # 時間的に分散させるため、時間軸でグループ化
        time_groups = {}
        for frame in sorted_frames:
            time_group = int(frame["timestamp"] // 60)  # 1分ごとにグループ化
            if time_group not in time_groups:
                time_groups[time_group] = []
            time_groups[time_group].append(frame)

        # 各グループから均等に選択
        selected_frames = []
        frames_per_group = max(1, max_count // len(time_groups))

        for group_frames in time_groups.values():
            selected_frames.extend(group_frames[:frames_per_group])

        # 不足分を品質順で補完
        if len(selected_frames) < max_count:
            remaining_frames = [f for f in sorted_frames if f not in selected_frames]
            selected_frames.extend(remaining_frames[: max_count - len(selected_frames)])

        # 時間順でソート
        selected_frames.sort(key=lambda x: x["timestamp"])

        self.logger.info(f"フレーム選択: {len(frames)} -> {len(selected_frames)}フレーム")
        return selected_frames[:max_count]

    def extract_frames_around_timestamp(
        self, video_path: str, timestamp: float, window_seconds: float = 5.0, max_frames: int = 10
    ) -> list[FrameAnnotation]:
        """
        指定時刻周辺のフレームを抽出

        Args:
            video_path: 動画ファイルのパス
            timestamp: 中心時刻（秒）
            window_seconds: 抽出範囲（秒）
            max_frames: 最大フレーム数

        Returns:
            抽出されたフレームのアノテーション情報リスト
        """
        video_path = Path(video_path)
        video_info = self.video_processor.get_video_info(str(video_path))

        start_time = max(0, timestamp - window_seconds / 2)
        end_time = min(video_info.get("duration", 0), timestamp + window_seconds / 2)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")

        try:
            fps = video_info.get("fps", 30.0)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # 開始フレームにシーク
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            extracted_frames = []
            frame_interval = max(1, (end_frame - start_frame) // max_frames)

            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (current_frame - start_frame) % frame_interval == 0:
                    # 品質分析
                    quality_scores = self.quality_analyzer.analyze_frame_quality(frame)

                    # フレームを保存
                    current_timestamp = current_frame / fps
                    frame_id = f"{video_path.stem}_ts_{current_timestamp:.2f}"

                    video_output_dir = self.output_dir / video_path.stem
                    video_output_dir.mkdir(exist_ok=True)

                    filename = f"{frame_id}.jpg"
                    image_path = video_output_dir / filename

                    processed_frame = self.video_processor.preprocess_frame(frame)
                    cv2.imwrite(str(image_path), processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    frame_annotation = FrameAnnotation(
                        frame_id=frame_id,
                        image_path=str(image_path),
                        image_width=processed_frame.shape[1],
                        image_height=processed_frame.shape[0],
                        timestamp=current_timestamp,
                        tiles=[],
                        quality_score=quality_scores["overall"],
                        is_valid=True,
                        scene_type="game",
                        game_phase="unknown",
                        annotator="frame_extractor",
                        notes=f"Extracted around timestamp {timestamp:.2f}s",
                    )

                    extracted_frames.append(frame_annotation)

                current_frame += 1

            return extracted_frames

        finally:
            cap.release()

    def analyze_video_scenes(self, video_path: str) -> list[dict[str, Any]]:
        """
        動画のシーン分析を行い、教師データに適したシーンを特定

        Args:
            video_path: 動画ファイルのパス

        Returns:
            シーン情報のリスト
        """
        # シーン変更を検出
        scene_changes = self.video_processor.detect_scene_changes(video_path)

        scenes = []
        prev_time = 0.0

        for i, change_time in enumerate(scene_changes + [float("inf")]):
            scene_duration = change_time - prev_time

            # 短すぎるシーンは除外
            if scene_duration >= 5.0:  # 5秒以上
                scene_info = {
                    "scene_id": f"scene_{i:03d}",
                    "start_time": prev_time,
                    "end_time": change_time if change_time != float("inf") else None,
                    "duration": scene_duration if change_time != float("inf") else None,
                    "is_suitable_for_training": self._is_scene_suitable_for_training(
                        video_path, prev_time, min(change_time, prev_time + 30.0)
                    ),
                }
                scenes.append(scene_info)

            prev_time = change_time

        return scenes

    def _is_scene_suitable_for_training(
        self, video_path: str, start_time: float, end_time: float
    ) -> bool:
        """シーンが教師データ作成に適しているかを判定"""
        # サンプルフレームを抽出して分析
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_times = [start_time + i * 2.0 for i in range(int((end_time - start_time) // 2))]

            quality_scores = []
            for sample_time in sample_times[:5]:  # 最大5サンプル
                frame_number = int(sample_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()
                if ret:
                    scores = self.quality_analyzer.analyze_frame_quality(frame)
                    quality_scores.append(scores["overall"])

            # 平均品質スコアが閾値以上なら適している
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                return avg_quality >= self.min_quality_score

            return False

        finally:
            cap.release()

    def get_extraction_statistics(self) -> dict[str, Any]:
        """抽出統計情報を取得"""
        stats = {
            "output_directory": str(self.output_dir),
            "min_quality_score": self.min_quality_score,
            "max_frames_per_video": self.max_frames_per_video,
            "frame_interval_seconds": self.frame_interval_seconds,
            "diversity_threshold": self.diversity_threshold,
        }

        # 出力ディレクトリの統計
        if self.output_dir.exists():
            video_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
            total_frames = 0

            for video_dir in video_dirs:
                frame_files = list(video_dir.glob("*.jpg"))
                total_frames += len(frame_files)

            stats.update(
                {
                    "processed_videos": len(video_dirs),
                    "total_extracted_frames": total_frames,
                    "avg_frames_per_video": total_frames / len(video_dirs) if video_dirs else 0,
                }
            )

        return stats
