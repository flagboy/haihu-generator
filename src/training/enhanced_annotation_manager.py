"""
拡張版アノテーション管理システム

動画から包括的な情報を抽出し、効率的にアノテーションを行うための統合システム
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..classification.tile_classifier import TileClassifier
from ..detection.player_detector import PlayerDetector
from ..detection.scene_detector import SceneDetector
from ..detection.score_reader import ScoreReader
from ..detection.tile_detector import TileDetector
from ..tracking.action_detector import ActionDetector
from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from .annotation_data import BoundingBox
from .enhanced_annotation_structure import (
    ActionAnnotation,
    EnhancedFrameAnnotation,
    EnhancedTileAnnotation,
    EnhancedVideoAnnotation,
    GameStateAnnotation,
    PlayerInfoAnnotation,
    PlayerPosition,
    SceneAnnotation,
    UIElementsAnnotation,
)
from .enhanced_dataset_manager import EnhancedDatasetManager
from .frame_similarity_detector import FrameSimilarityDetector, SimilarityResult
from .semi_auto_labeler import SemiAutoLabeler


class EnhancedAnnotationManager(LoggerMixin):
    """拡張版アノテーション管理システム"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """初期化"""
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # 各種検出器の初期化
        self.scene_detector = SceneDetector(self.config.get("scene_detection", {}))
        self.score_reader = ScoreReader(self.config.get("score_reading", {}))
        self.player_detector = PlayerDetector(self.config.get("player_detection", {}))
        self.tile_detector = TileDetector(self.config_manager)
        self.tile_classifier = TileClassifier(self.config_manager)
        self.action_detector = ActionDetector(self.config.get("action_detection", {}))

        # 半自動ラベラー
        self.semi_auto_labeler = SemiAutoLabeler(self.config_manager)

        # データセット管理
        self.dataset_manager = EnhancedDatasetManager(self.config_manager)

        # フレーム類似度検出
        self.similarity_detector = FrameSimilarityDetector(
            self.config.get("similarity_detection", {})
        )

        # キャッシュ
        self.previous_frame: np.ndarray | None = None
        self.previous_annotation: EnhancedFrameAnnotation | None = None
        self.annotated_frames_cache: list[dict[str, Any]] = []  # 最近アノテーションしたフレーム

        # 設定
        self.enable_auto_detection = self.config.get("annotation", {}).get(
            "enable_auto_detection", True
        )
        self.confidence_threshold = self.config.get("annotation", {}).get(
            "confidence_threshold", 0.7
        )
        self.mark_low_confidence = self.config.get("annotation", {}).get(
            "mark_low_confidence", True
        )
        self.skip_similar_frames = self.config.get("annotation", {}).get(
            "skip_similar_frames", True
        )
        self.similarity_cache_size = self.config.get("annotation", {}).get(
            "similarity_cache_size", 100
        )

        self.logger.info("EnhancedAnnotationManager初期化完了")

    def check_frame_similarity(
        self, frame: np.ndarray, frame_path: str | None = None
    ) -> SimilarityResult:
        """
        フレームの類似度をチェック

        Args:
            frame: チェック対象のフレーム
            frame_path: フレームのパス（キャッシュ用）

        Returns:
            類似度検出結果
        """
        if not self.skip_similar_frames or not self.annotated_frames_cache:
            return SimilarityResult(
                is_similar=False,
                similarity_score=0.0,
                matched_frame_id=None,
                similarity_type="different",
                metadata={},
            )

        # 参照フレームを準備
        reference_frames = []
        for cached_frame in self.annotated_frames_cache[-self.similarity_cache_size :]:
            reference_frames.append(
                {
                    "id": cached_frame["frame_id"],
                    "path": cached_frame["image_path"],
                    "hash": cached_frame.get("hash"),
                }
            )

        # 類似度チェック
        similarity_result = self.similarity_detector.check_similarity(
            frame, reference_frames, quick_check=True
        )

        if similarity_result.is_similar:
            self.logger.info(
                f"類似フレーム検出: {similarity_result.similarity_type} "
                f"(スコア: {similarity_result.similarity_score:.3f})"
            )

        return similarity_result

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        image_path: str,
        video_id: str,
        skip_if_similar: bool = True,
    ) -> EnhancedFrameAnnotation | None:
        """
        フレームの包括的アノテーション

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ
            image_path: 画像パス
            video_id: 動画ID
            skip_if_similar: 類似フレームをスキップするか

        Returns:
            拡張版フレームアノテーション（類似フレームの場合はNone）
        """
        start_time = time.time()
        frame_id = f"{video_id}_frame_{frame_number}"

        # 類似度チェック
        if skip_if_similar and self.skip_similar_frames:
            similarity_result = self.check_frame_similarity(frame, image_path)

            if similarity_result.is_similar and similarity_result.similarity_type in [
                "identical",
                "very_similar",
            ]:
                self.logger.info(f"フレーム {frame_number} は既存フレームと類似のためスキップ")
                return None

        # 基本アノテーション作成
        annotation = EnhancedFrameAnnotation(
            frame_id=frame_id,
            image_path=image_path,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            timestamp=timestamp,
            frame_number=frame_number,
            auto_detected=self.enable_auto_detection,
        )

        if self.enable_auto_detection:
            # 1. シーン検出
            scene_result = self.scene_detector.detect_scene(frame, frame_number, timestamp)
            annotation.scene_annotation = SceneAnnotation(
                scene_type=scene_result.scene_type.value,
                confidence=scene_result.confidence,
                is_transition=scene_result.metadata.get("is_transition", False),
                metadata=scene_result.metadata,
            )

            # 重要シーンをマーク
            if scene_result.is_game_boundary():
                annotation.is_key_frame = True

            # 2. プレイヤー情報検出
            player_detections = self.player_detector.detect_players(frame)
            current_turn = self.player_detector.detect_current_turn(frame)

            # 3. ゲーム状態読み取り
            scores = self.score_reader.read_scores(frame)
            round_info = self._extract_round_info(frame)
            dora_indicators = self._extract_dora_indicators(frame, scene_result)

            # 4. 牌検出と分類
            tile_detections = self.tile_detector.detect_tiles(frame)

            # 5. 各検出結果を統合
            annotation = self._integrate_detections(
                annotation,
                player_detections,
                current_turn,
                scores,
                round_info,
                dora_indicators,
                tile_detections,
                frame,
            )

            # 6. アクション検出
            if self.previous_frame is not None and self.previous_annotation is not None:
                actions = self._detect_actions(frame, annotation)
                annotation.detected_actions = actions

            # 7. 信頼度チェック
            if self.mark_low_confidence:
                annotation = self._check_confidence(annotation)

        # 処理時間を記録
        processing_time = time.time() - start_time
        annotation.metadata["processing_time"] = processing_time

        # キャッシュ更新
        self.previous_frame = frame.copy()
        self.previous_annotation = annotation

        # アノテーション済みフレームキャッシュに追加
        self._update_annotated_frames_cache(annotation, frame)

        self.logger.debug(f"フレーム {frame_number} のアノテーション完了: {processing_time:.3f}秒")

        return annotation

    def _update_annotated_frames_cache(
        self, annotation: EnhancedFrameAnnotation, frame: np.ndarray
    ):
        """アノテーション済みフレームキャッシュを更新"""
        # フレームハッシュを計算
        frame_hash = self.similarity_detector._compute_hash(frame)

        cache_entry = {
            "frame_id": annotation.frame_id,
            "image_path": annotation.image_path,
            "timestamp": annotation.timestamp,
            "hash": frame_hash,
            "tile_count": len(annotation.tiles),
            "scene_type": annotation.scene_annotation.scene_type
            if annotation.scene_annotation
            else None,
        }

        self.annotated_frames_cache.append(cache_entry)

        # キャッシュサイズ制限
        if len(self.annotated_frames_cache) > self.similarity_cache_size * 2:
            self.annotated_frames_cache = self.annotated_frames_cache[-self.similarity_cache_size :]

    def _integrate_detections(
        self,
        annotation: EnhancedFrameAnnotation,
        player_detections: dict,
        current_turn: dict,
        scores: dict,
        round_info: str,
        dora_indicators: list[str],
        tile_detections: list,
        frame: np.ndarray,
    ) -> EnhancedFrameAnnotation:
        """各種検出結果を統合"""

        # プレイヤー情報を構築
        positions = {}
        for position, detection in player_detections.items():
            positions[position] = PlayerPosition(
                position=position,
                player_area=BoundingBox(
                    detection["area"][0],
                    detection["area"][1],
                    detection["area"][2],
                    detection["area"][3],
                ),
                hand_area=BoundingBox(
                    detection["hand_area"][0],
                    detection["hand_area"][1],
                    detection["hand_area"][2],
                    detection["hand_area"][3],
                ),
                discard_area=BoundingBox(
                    detection["discard_area"][0],
                    detection["discard_area"][1],
                    detection["discard_area"][2],
                    detection["discard_area"][3],
                ),
                call_area=BoundingBox(
                    detection["call_area"][0],
                    detection["call_area"][1],
                    detection["call_area"][2],
                    detection["call_area"][3],
                )
                if "call_area" in detection
                else None,
                is_active=position == current_turn.get("position"),
                is_dealer=detection.get("is_dealer", False),
            )

        annotation.player_info = PlayerInfoAnnotation(
            positions=positions,
            current_turn=current_turn.get("position"),
            scores=scores,
            riichi_states={},  # TODO: リーチ状態の検出
            temp_points={},  # TODO: 一時点数の検出
        )

        # ゲーム状態
        annotation.game_state = GameStateAnnotation(
            round_info=round_info,
            dealer_position=self._find_dealer_position(player_detections),
            dora_indicators=dora_indicators,
            ura_dora_indicators=[],  # TODO: 裏ドラ検出
            remaining_tiles=70,  # TODO: 残り牌数計算
            riichi_sticks=0,  # TODO: 供託検出
            honba=0,  # TODO: 本場数検出
        )

        # 牌アノテーション
        enhanced_tiles = []
        for detection in tile_detections:
            # 牌画像を切り出して分類
            x1, y1, x2, y2 = detection.bbox
            tile_image = frame[y1:y2, x1:x2]

            if tile_image.size > 0:
                classification = self.tile_classifier.classify_tile(tile_image)

                # どのプレイヤーの牌か判定
                player_position = self._assign_tile_to_player(detection.bbox, positions)

                # エリアタイプを判定
                area_type = self._determine_area_type(
                    detection.bbox, positions.get(player_position) if player_position else None
                )

                enhanced_tile = EnhancedTileAnnotation(
                    tile_id=classification.tile_name,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    confidence=min(detection.confidence, classification.confidence),
                    area_type=area_type,
                    is_face_up=True,  # TODO: 裏向き検出
                    is_occluded=False,  # TODO: 遮蔽検出
                    occlusion_ratio=0.0,
                    player_position=player_position,
                    is_dora=classification.tile_name in dora_indicators,
                    is_red_dora=self._is_red_dora(classification.tile_name),
                    turn_number=None,  # 後で計算
                    action_context=None,  # アクション検出時に設定
                    annotator="auto",
                    notes=f"Auto-detected with confidence {classification.confidence:.3f}",
                )

                enhanced_tiles.append(enhanced_tile)

        annotation.tiles = enhanced_tiles

        # UI要素（TODO: 実装）
        annotation.ui_elements = UIElementsAnnotation(elements=[])

        return annotation

    def _detect_actions(
        self, current_frame: np.ndarray, current_annotation: EnhancedFrameAnnotation
    ) -> list[ActionAnnotation]:
        """アクション検出"""
        if not self.previous_annotation:
            return []

        actions = []

        # 各プレイヤーについてアクションを検出
        if current_annotation.player_info and self.previous_annotation.player_info:
            for position in ["東", "南", "西", "北"]:
                # 現在と前フレームの手牌を比較
                curr_hand = [
                    t
                    for t in current_annotation.tiles
                    if t.player_position == position and t.area_type == "hand"
                ]
                prev_hand = [
                    t
                    for t in self.previous_annotation.tiles
                    if t.player_position == position and t.area_type == "hand"
                ]

                # 捨て牌を比較
                curr_discards = [
                    t
                    for t in current_annotation.tiles
                    if t.player_position == position and t.area_type == "discard"
                ]
                prev_discards = [
                    t
                    for t in self.previous_annotation.tiles
                    if t.player_position == position and t.area_type == "discard"
                ]

                # 手牌が増えた → ツモ
                if len(curr_hand) > len(prev_hand):
                    new_tiles = self._find_new_tiles(curr_hand, prev_hand)
                    if new_tiles:
                        actions.append(
                            ActionAnnotation(
                                action_type="draw",
                                player_position=position,
                                tiles=[t.tile_id for t in new_tiles],
                                confidence=min(t.confidence for t in new_tiles),
                                timestamp=current_annotation.timestamp,
                            )
                        )

                # 捨て牌が増えた → 打牌
                if len(curr_discards) > len(prev_discards):
                    new_discards = self._find_new_tiles(curr_discards, prev_discards)
                    if new_discards:
                        actions.append(
                            ActionAnnotation(
                                action_type="discard",
                                player_position=position,
                                tiles=[t.tile_id for t in new_discards],
                                confidence=min(t.confidence for t in new_discards),
                                timestamp=current_annotation.timestamp,
                            )
                        )

        return actions

    def _extract_round_info(self, frame: np.ndarray) -> str:
        """局情報を抽出（TODO: 実装）"""
        # OCRまたはテンプレートマッチングで実装
        return "東1局 0本場"

    def _extract_dora_indicators(self, frame: np.ndarray, scene_result) -> list[str]:
        """ドラ表示牌を抽出（TODO: 実装）"""
        if scene_result.scene_type.value == "dora_indicator":
            # ドラ表示シーンから抽出
            return []
        return []

    def _find_dealer_position(self, player_detections: dict) -> str:
        """親の位置を特定"""
        for position, detection in player_detections.items():
            if detection.get("is_dealer", False):
                return position
        return "東"  # デフォルト

    def _assign_tile_to_player(
        self, tile_bbox: tuple, positions: dict[str, PlayerPosition]
    ) -> str | None:
        """牌をプレイヤーに割り当て"""
        tile_center_x = (tile_bbox[0] + tile_bbox[2]) / 2
        tile_center_y = (tile_bbox[1] + tile_bbox[3]) / 2

        for position, player_pos in positions.items():
            # プレイヤーエリア全体で判定
            if (
                player_pos.player_area.x1 <= tile_center_x <= player_pos.player_area.x2
                and player_pos.player_area.y1 <= tile_center_y <= player_pos.player_area.y2
            ):
                return position

        return None

    def _determine_area_type(self, tile_bbox: tuple, player_position: PlayerPosition | None) -> str:
        """牌のエリアタイプを判定"""
        if not player_position:
            return "unknown"

        tile_center_x = (tile_bbox[0] + tile_bbox[2]) / 2
        tile_center_y = (tile_bbox[1] + tile_bbox[3]) / 2

        # 手牌エリア
        if (
            player_position.hand_area.x1 <= tile_center_x <= player_position.hand_area.x2
            and player_position.hand_area.y1 <= tile_center_y <= player_position.hand_area.y2
        ):
            return "hand"

        # 捨て牌エリア
        if (
            player_position.discard_area.x1 <= tile_center_x <= player_position.discard_area.x2
            and player_position.discard_area.y1 <= tile_center_y <= player_position.discard_area.y2
        ):
            return "discard"

        # 鳴き牌エリア
        if player_position.call_area and (
            player_position.call_area.x1 <= tile_center_x <= player_position.call_area.x2
            and player_position.call_area.y1 <= tile_center_y <= player_position.call_area.y2
        ):
            return "call"

        return "unknown"

    def _is_red_dora(self, tile_id: str) -> bool:
        """赤ドラ判定"""
        return tile_id in ["5mr", "5pr", "5sr"]

    def _find_new_tiles(
        self,
        current_tiles: list[EnhancedTileAnnotation],
        previous_tiles: list[EnhancedTileAnnotation],
    ) -> list[EnhancedTileAnnotation]:
        """新しく追加された牌を検出"""
        new_tiles = []

        # 簡易的な実装（位置ベース）
        prev_positions = {(t.bbox.center, t.tile_id) for t in previous_tiles}

        for tile in current_tiles:
            tile_key = (tile.bbox.center, tile.tile_id)
            if tile_key not in prev_positions:
                new_tiles.append(tile)

        return new_tiles

    def _check_confidence(self, annotation: EnhancedFrameAnnotation) -> EnhancedFrameAnnotation:
        """信頼度チェック"""
        low_confidence_tiles = 0
        total_confidence = 0.0

        for tile in annotation.tiles:
            total_confidence += tile.confidence
            if tile.confidence < self.confidence_threshold:
                low_confidence_tiles += 1

        # 低信頼度の牌が多い場合はレビュー必要
        if annotation.tiles:
            avg_confidence = total_confidence / len(annotation.tiles)
            if (
                avg_confidence < self.confidence_threshold
                or low_confidence_tiles > len(annotation.tiles) * 0.2
            ):
                annotation.needs_review = True
                annotation.notes += f" Low confidence: {avg_confidence:.3f}"

        return annotation

    def annotate_video(
        self,
        video_path: str,
        output_dir: str | None = None,
        save_interval: int = 100,
        skip_similar: bool = True,
        similarity_threshold: float | None = None,
    ) -> EnhancedVideoAnnotation:
        """
        動画全体をアノテーション

        Args:
            video_path: 動画パス
            output_dir: 出力ディレクトリ
            save_interval: 保存間隔（フレーム数）
            skip_similar: 類似フレームをスキップするか
            similarity_threshold: 類似度閾値（Noneの場合はデフォルト値）

        Returns:
            拡張版動画アノテーション
        """
        # 一時的に閾値を変更
        original_threshold = None
        if similarity_threshold is not None:
            original_threshold = self.similarity_detector.similar_threshold
            self.similarity_detector.similar_threshold = similarity_threshold
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"動画が見つかりません: {video_path}")

        # 出力ディレクトリ準備
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # 動画情報取得
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # 動画アノテーション作成
        video_id = video_path.stem
        video_annotation = EnhancedVideoAnnotation(
            video_id=video_id,
            video_path=str(video_path),
            video_name=video_path.name,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )

        self.logger.info(f"動画アノテーション開始: {video_path.name} ({total_frames}フレーム)")

        frame_number = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_number / fps

                # フレーム画像を保存
                if output_dir:
                    image_path = output_dir / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(image_path), frame)
                else:
                    image_path = f"frame_{frame_number:06d}.jpg"

                # アノテーション実行
                frame_annotation = self.annotate_frame(
                    frame,
                    frame_number,
                    timestamp,
                    str(image_path),
                    video_id,
                    skip_if_similar=skip_similar,
                )

                # スキップされたフレームの処理
                if frame_annotation is None:
                    self.logger.debug(f"フレーム {frame_number} はスキップされました")
                    # スキップ情報を記録
                    skip_annotation = EnhancedFrameAnnotation(
                        frame_id=f"{video_id}_frame_{frame_number}",
                        image_path=str(image_path),
                        image_width=frame.shape[1],
                        image_height=frame.shape[0],
                        timestamp=timestamp,
                        frame_number=frame_number,
                        auto_detected=False,
                        metadata={"skipped": True, "reason": "similar_to_previous"},
                    )
                    video_annotation.frames.append(skip_annotation)
                else:
                    video_annotation.frames.append(frame_annotation)

                # 定期的に保存
                if output_dir and (frame_number + 1) % save_interval == 0:
                    self.dataset_manager.save_enhanced_annotation(video_annotation)
                    annotated_count = sum(
                        1 for f in video_annotation.frames if not f.metadata.get("skipped", False)
                    )
                    skipped_count = frame_number + 1 - annotated_count
                    self.logger.info(
                        f"進捗: {frame_number + 1}/{total_frames}フレーム "
                        f"(アノテーション: {annotated_count}, スキップ: {skipped_count})"
                    )

                frame_number += 1

        finally:
            cap.release()
            # 閾値を元に戻す
            if original_threshold is not None:
                self.similarity_detector.similar_threshold = original_threshold

        # 最終保存
        if output_dir:
            self.dataset_manager.save_enhanced_annotation(video_annotation)

        # 統計情報
        total_frames_processed = len(video_annotation.frames)
        annotated_count = sum(
            1 for f in video_annotation.frames if not f.metadata.get("skipped", False)
        )
        skipped_count = total_frames_processed - annotated_count

        self.logger.info(
            f"動画アノテーション完了: 総フレーム数: {total_frames_processed}, "
            f"アノテーション: {annotated_count}, スキップ: {skipped_count} "
            f"(スキップ率: {skipped_count / total_frames_processed * 100:.1f}%)"
        )

        return video_annotation

    def create_correction_session(self, video_id: str, annotator_name: str) -> dict[str, Any]:
        """
        修正セッションを作成

        Args:
            video_id: 動画ID
            annotator_name: アノテーター名

        Returns:
            セッション情報
        """
        # レビューが必要なフレームを取得
        frames_to_review = self.dataset_manager.get_frames_needing_review(video_id)

        session = {
            "session_id": f"{video_id}_{int(time.time())}",
            "video_id": video_id,
            "annotator": annotator_name,
            "start_time": datetime.now().isoformat(),
            "total_frames": len(frames_to_review),
            "completed_frames": 0,
            "frames": frames_to_review,
        }

        self.logger.info(f"修正セッション作成: {len(frames_to_review)}フレーム")

        return session
