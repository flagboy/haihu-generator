"""
拡張半自動ラベリングシステム

SceneDetector、ScoreReader、PlayerDetectorを統合した
高精度な教師データ作成支援システム
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..detection import (
    PlayerDetectionResult,
    PlayerDetector,
    SceneDetectionResult,
    SceneDetector,
    SceneType,
    ScoreReader,
    ScoreReadingResult,
)
from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from .annotation_data import FrameAnnotation
from .semi_auto_labeler import PredictionResult, SemiAutoLabeler


class EnhancedPredictionResult(PredictionResult):
    """拡張予測結果クラス"""

    def __init__(
        self,
        frame_annotation: FrameAnnotation,
        detection_results: list,
        classification_results: list,
        scene_result: SceneDetectionResult | None = None,
        score_result: ScoreReadingResult | None = None,
        player_result: PlayerDetectionResult | None = None,
    ):
        """
        初期化

        Args:
            frame_annotation: フレームアノテーション
            detection_results: 検出結果
            classification_results: 分類結果
            scene_result: シーン検出結果
            score_result: 点数読み取り結果
            player_result: プレイヤー検出結果
        """
        super().__init__(frame_annotation, detection_results, classification_results)
        self.scene_result = scene_result
        self.score_result = score_result
        self.player_result = player_result

    def get_scene_context(self) -> dict[str, Any]:
        """シーンコンテキストを取得"""
        context = {}

        if self.scene_result:
            context["scene_type"] = self.scene_result.scene_type.value
            context["scene_confidence"] = self.scene_result.confidence
            context["scene_metadata"] = self.scene_result.metadata

        if self.player_result:
            context["player_positions"] = {
                player.position.value: {
                    "name": player.name,
                    "is_dealer": player.is_dealer,
                    "is_active": player.is_active,
                }
                for player in self.player_result.players
            }
            context["active_player"] = (
                self.player_result.active_position.value
                if self.player_result.active_position
                else None
            )
            context["dealer_position"] = self.player_result.dealer_position.value

        if self.score_result:
            context["scores"] = {
                score.player_position: score.score for score in self.score_result.scores
            }

        return context


class EnhancedSemiAutoLabeler(SemiAutoLabeler, LoggerMixin):
    """拡張半自動ラベリングクラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        super().__init__(config_manager)

        # 新しい検出器を初期化
        self.scene_detector = SceneDetector(self.config_manager.get_config().get("scene_detection"))
        self.score_reader = ScoreReader(self.config_manager.get_config().get("score_reading"))
        self.player_detector = PlayerDetector(
            self.config_manager.get_config().get("player_detection")
        )

        # フィルタリング設定
        labeling_config = (
            self.config_manager.get_config().get("training", {}).get("enhanced_labeling", {})
        )
        self.filter_non_game_scenes = labeling_config.get("filter_non_game_scenes", True)
        self.min_scene_confidence = labeling_config.get("min_scene_confidence", 0.7)
        self.use_context_for_classification = labeling_config.get(
            "use_context_for_classification", True
        )

        # 統計情報
        self.scene_stats = {
            SceneType.GAME_PLAY: 0,
            SceneType.ROUND_START: 0,
            SceneType.ROUND_END: 0,
            SceneType.MENU: 0,
            SceneType.UNKNOWN: 0,
        }

        self.logger.info("EnhancedSemiAutoLabeler初期化完了")

    def predict_frame_annotations(
        self, frame_annotation: FrameAnnotation
    ) -> EnhancedPredictionResult:
        """
        拡張フレーム予測

        Args:
            frame_annotation: 入力フレームアノテーション

        Returns:
            拡張予測結果
        """
        image_path = frame_annotation.image_path
        if not Path(image_path).exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        self.logger.info(f"拡張フレーム予測開始: {frame_annotation.frame_id}")

        # 画像を読み込み
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        # シーン検出
        scene_result = self.scene_detector.detect_scene(
            image, frame_annotation.frame_id, frame_annotation.timestamp
        )
        self.scene_stats[scene_result.scene_type] += 1

        # 非ゲームシーンのフィルタリング
        if self.filter_non_game_scenes and self._should_filter_scene(scene_result):
            self.logger.info(
                f"非ゲームシーンをスキップ: {scene_result.scene_type.value} "
                f"(confidence: {scene_result.confidence:.3f})"
            )
            # 空の結果を返す
            return self._create_empty_result(frame_annotation, scene_result)

        # プレイヤー検出
        player_result = self.player_detector.detect_players(
            image, frame_annotation.frame_id, frame_annotation.timestamp
        )

        # 点数読み取り
        score_result = self.score_reader.read_scores(
            image, frame_annotation.frame_id, frame_annotation.timestamp
        )

        # シーンコンテキストを作成
        scene_context = self._create_scene_context(scene_result, player_result, score_result)

        # 牌検出（コンテキスト情報を活用）
        detection_results = self.tile_detector.detect_tiles(image)

        # エリア分類（コンテキスト情報を渡す）
        if self.auto_area_classification and self.use_context_for_classification:
            area_classified_detections = self.tile_detector.classify_tile_areas(
                image, detection_results, scene_context
            )
        else:
            area_classified_detections = self.tile_detector.classify_tile_areas(
                image, detection_results
            )

        # 残りの処理は親クラスと同様
        classification_results = []
        predicted_tiles = []

        for area_type, detections in area_classified_detections.items():
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                tile_image = image[y1:y2, x1:x2]

                if tile_image.size > 0:
                    classification_result = self.tile_classifier.classify_tile(tile_image)
                    classification_results.append(classification_result)

                    if classification_result.confidence >= self.confidence_threshold:
                        is_occluded, occlusion_ratio = (
                            self._detect_occlusion(tile_image)
                            if self.enable_occlusion_detection
                            else (False, 0.0)
                        )

                        from .annotation_data import BoundingBox, TileAnnotation

                        bbox = BoundingBox(x1, y1, x2, y2)
                        tile_annotation = TileAnnotation(
                            tile_id=classification_result.tile_name,
                            bbox=bbox,
                            confidence=min(detection.confidence, classification_result.confidence),
                            area_type=area_type,
                            is_face_up=True,
                            is_occluded=is_occluded,
                            occlusion_ratio=occlusion_ratio,
                            annotator="enhanced_semi_auto_labeler",
                            notes=(
                                f"Scene: {scene_result.scene_type.value}, "
                                f"Detection: {detection.confidence:.3f}, "
                                f"Classification: {classification_result.confidence:.3f}"
                            ),
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
            quality_score=self._calculate_quality_score(
                scene_result, score_result, player_result, len(predicted_tiles)
            ),
            is_valid=True,
            scene_type=scene_result.scene_type.value,
            game_phase=self._determine_game_phase(scene_result),
            annotated_at=datetime.now(),
            annotator="enhanced_semi_auto_labeler",
            notes=(
                f"Scene: {scene_result.scene_type.value} ({scene_result.confidence:.3f}), "
                f"Tiles: {len(predicted_tiles)}, "
                f"Players detected: {len(player_result.players) if player_result else 0}"
            ),
        )

        return EnhancedPredictionResult(
            updated_frame,
            detection_results,
            classification_results,
            scene_result,
            score_result,
            player_result,
        )

    def filter_game_frames(self, frame_annotations: list[FrameAnnotation]) -> list[FrameAnnotation]:
        """
        ゲームフレームのみをフィルタリング

        Args:
            frame_annotations: フレームアノテーションのリスト

        Returns:
            ゲームフレームのみのリスト
        """
        filtered_frames = []

        for frame in frame_annotations:
            try:
                import cv2

                image = cv2.imread(frame.image_path)
                if image is None:
                    continue

                scene_result = self.scene_detector.detect_scene(
                    image, frame.frame_id, frame.timestamp
                )

                if not self._should_filter_scene(scene_result):
                    filtered_frames.append(frame)

            except Exception as e:
                self.logger.error(f"フレームフィルタリングエラー {frame.frame_id}: {e}")

        self.logger.info(
            f"フレームフィルタリング完了: {len(filtered_frames)}/{len(frame_annotations)} "
            f"({len(filtered_frames) / len(frame_annotations) * 100:.1f}%)"
        )

        return filtered_frames

    def _should_filter_scene(self, scene_result: SceneDetectionResult) -> bool:
        """シーンをフィルタリングすべきか判定"""
        # メニューや不明なシーンはフィルタリング
        if scene_result.scene_type in [SceneType.MENU, SceneType.UNKNOWN]:
            return True

        # 信頼度が低い場合もフィルタリング
        return scene_result.confidence < self.min_scene_confidence

    def _create_scene_context(
        self,
        scene_result: SceneDetectionResult,
        player_result: PlayerDetectionResult | None,
        score_result: ScoreReadingResult | None,
    ) -> dict[str, Any]:
        """シーンコンテキストを作成"""
        context = {
            "scene_type": scene_result.scene_type.value,
            "scene_confidence": scene_result.confidence,
        }

        if player_result:
            context["player_positions"] = {
                player.position.value: {
                    "bbox": player.bbox,
                    "is_dealer": player.is_dealer,
                    "is_active": player.is_active,
                }
                for player in player_result.players
            }

        if score_result:
            context["scores"] = {
                score.player_position: score.score for score in score_result.scores
            }

        return context

    def _create_empty_result(
        self, frame_annotation: FrameAnnotation, scene_result: SceneDetectionResult
    ) -> EnhancedPredictionResult:
        """空の結果を作成"""
        empty_frame = FrameAnnotation(
            frame_id=frame_annotation.frame_id,
            image_path=frame_annotation.image_path,
            image_width=frame_annotation.image_width,
            image_height=frame_annotation.image_height,
            timestamp=frame_annotation.timestamp,
            tiles=[],
            quality_score=0.0,
            is_valid=False,
            scene_type=scene_result.scene_type.value,
            game_phase="none",
            annotated_at=datetime.now(),
            annotator="enhanced_semi_auto_labeler",
            notes=f"Filtered scene: {scene_result.scene_type.value}",
        )

        return EnhancedPredictionResult(empty_frame, [], [], scene_result, None, None)

    def _calculate_quality_score(
        self,
        scene_result: SceneDetectionResult,
        score_result: ScoreReadingResult | None,
        player_result: PlayerDetectionResult | None,
        tile_count: int,
    ) -> float:
        """品質スコアを計算"""
        scores = []

        # シーン検出の信頼度
        scores.append(scene_result.confidence)

        # 点数読み取りの信頼度
        if score_result:
            scores.append(score_result.total_confidence)

        # プレイヤー検出の信頼度
        if player_result and player_result.players:
            player_confidences = [p.confidence for p in player_result.players]
            scores.append(np.mean(player_confidences))

        # 牌検出数による調整
        tile_score = min(1.0, tile_count / 14.0)  # 14枚を基準
        scores.append(tile_score)

        return float(np.mean(scores))

    def _determine_game_phase(self, scene_result: SceneDetectionResult) -> str:
        """ゲームフェーズを判定"""
        phase_map = {
            SceneType.GAME_START: "start",
            SceneType.ROUND_START: "round_start",
            SceneType.GAME_PLAY: "playing",
            SceneType.ROUND_END: "round_end",
            SceneType.GAME_END: "end",
            SceneType.DORA_INDICATOR: "dora",
            SceneType.RIICHI: "riichi",
            SceneType.TSUMO: "tsumo",
            SceneType.RON: "ron",
            SceneType.DRAW: "draw",
        }

        return phase_map.get(scene_result.scene_type, "unknown")

    def get_enhanced_statistics(self) -> dict[str, Any]:
        """拡張統計情報を取得"""
        base_stats = self.get_labeling_statistics()

        # シーン統計を追加
        base_stats["scene_statistics"] = {
            scene_type.value: count for scene_type, count in self.scene_stats.items()
        }

        # フィルタリング率を計算
        total_scenes = sum(self.scene_stats.values())
        if total_scenes > 0:
            game_scenes = (
                self.scene_stats[SceneType.GAME_PLAY]
                + self.scene_stats[SceneType.ROUND_START]
                + self.scene_stats[SceneType.ROUND_END]
            )
            base_stats["game_scene_ratio"] = game_scenes / total_scenes
            base_stats["filtered_scene_ratio"] = 1 - (game_scenes / total_scenes)

        return base_stats
