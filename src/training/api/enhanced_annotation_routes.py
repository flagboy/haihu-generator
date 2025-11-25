"""
拡張版アノテーションAPI

効率的な教師データ作成のためのWebAPI
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file

from ...utils.config import ConfigManager
from ...utils.logger import get_logger
from ..enhanced_annotation_manager import EnhancedAnnotationManager
from ..enhanced_annotation_structure import (
    EnhancedTileAnnotation,
    GameStateAnnotation,
)
from ..enhanced_dataset_manager import EnhancedDatasetManager
from ..smart_frame_selector import SmartFrameSelector

# Blueprint作成
enhanced_annotation_bp = Blueprint("enhanced_annotation", __name__)
logger = get_logger(__name__)

# グローバル変数
annotation_manager: EnhancedAnnotationManager | None = None
dataset_manager: EnhancedDatasetManager | None = None
frame_selector: SmartFrameSelector | None = None
current_session: dict[str, Any] = {}


def init_managers():
    """マネージャーの初期化"""
    global annotation_manager, dataset_manager, frame_selector

    if not annotation_manager:
        config_manager = ConfigManager()
        annotation_manager = EnhancedAnnotationManager(config_manager)
        dataset_manager = EnhancedDatasetManager(config_manager)
        frame_selector = SmartFrameSelector(config_manager.get_config().get("frame_selection", {}))
        logger.info("拡張版アノテーションマネージャー初期化完了")


@enhanced_annotation_bp.route("/api/annotation/session", methods=["POST"])
def create_session():
    """アノテーションセッションの作成"""
    try:
        init_managers()

        data = request.json
        video_id = data.get("video_id")
        annotator_name = data.get("annotator", "unknown")

        if not video_id:
            return jsonify({"error": "video_id is required"}), 400

        # セッション作成
        session = annotation_manager.create_correction_session(video_id, annotator_name)
        current_session.update(session)

        return jsonify({"success": True, "session": session})

    except Exception as e:
        logger.error(f"セッション作成エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/frame/<int:frame_number>")
def get_frame(frame_number: int):
    """フレーム情報の取得"""
    try:
        if not current_session:
            return jsonify({"error": "No active session"}), 400

        video_id = current_session.get("video_id")
        if not video_id:
            return jsonify({"error": "Invalid session"}), 400

        # アノテーションを読み込み
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if not video_annotation or frame_number >= len(video_annotation.frames):
            return jsonify({"error": "Frame not found"}), 404

        frame = video_annotation.frames[frame_number]

        # スキップされたフレームかチェック
        is_skipped = frame.metadata.get("skipped", False)
        skip_reason = frame.metadata.get("reason", "")

        # レスポンス作成
        response = {
            "frameNumber": frame_number,
            "imagePath": f"/api/annotation/image/{frame.frame_id}",
            "sceneType": frame.scene_annotation.scene_type if frame.scene_annotation else None,
            "roundInfo": frame.game_state.round_info if frame.game_state else None,
            "dealerPosition": frame.game_state.dealer_position if frame.game_state else None,
            "annotations": [],
            "isSkipped": is_skipped,
            "skipReason": skip_reason,
        }

        # アノテーション変換
        for tile in frame.tiles:
            response["annotations"].append(
                {
                    "bbox": {
                        "x1": tile.bbox.x1,
                        "y1": tile.bbox.y1,
                        "x2": tile.bbox.x2,
                        "y2": tile.bbox.y2,
                    },
                    "type": tile.tile_id,
                    "confidence": tile.confidence,
                    "area": tile.area_type,
                    "player": tile.player_position,
                    "notes": tile.notes,
                }
            )

        return jsonify(response)

    except Exception as e:
        logger.error(f"フレーム取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/image/<frame_id>")
def get_image(frame_id: str):
    """フレーム画像の取得"""
    try:
        # TODO: 実際の画像パスを解決
        # 仮実装
        image_path = Path(f"data/frames/{frame_id}.jpg")

        if not image_path.exists():
            # ダミー画像を生成
            dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(
                dummy_image,
                f"Frame: {frame_id}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )

            temp_path = Path(f"/tmp/{frame_id}.jpg")
            cv2.imwrite(str(temp_path), dummy_image)
            return send_file(str(temp_path), mimetype="image/jpeg")

        return send_file(str(image_path), mimetype="image/jpeg")

    except Exception as e:
        logger.error(f"画像取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/auto-detect/<int:frame_number>", methods=["POST"])
def auto_detect(frame_number: int):
    """自動検出の実行"""
    try:
        init_managers()

        if not current_session:
            return jsonify({"error": "No active session"}), 400

        video_id = current_session.get("video_id")

        # フレーム画像を取得（仮実装）
        # TODO: 実際の画像取得処理
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 自動アノテーション実行
        timestamp = frame_number / 30.0  # 30fps想定
        image_path = f"frame_{frame_number:06d}.jpg"

        annotation = annotation_manager.annotate_frame(
            frame, frame_number, timestamp, image_path, video_id
        )

        # レスポンス作成
        detected_annotations = []
        for tile in annotation.tiles:
            detected_annotations.append(
                {
                    "bbox": {
                        "x1": tile.bbox.x1,
                        "y1": tile.bbox.y1,
                        "x2": tile.bbox.x2,
                        "y2": tile.bbox.y2,
                    },
                    "type": tile.tile_id,
                    "confidence": tile.confidence,
                    "area": tile.area_type,
                    "player": tile.player_position,
                }
            )

        return jsonify(
            {
                "success": True,
                "annotations": detected_annotations,
                "detectedActions": [
                    {
                        "type": action.action_type,
                        "player": action.player_position,
                        "tiles": action.tiles,
                        "confidence": action.confidence,
                    }
                    for action in annotation.detected_actions
                ],
            }
        )

    except Exception as e:
        logger.error(f"自動検出エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/save", methods=["POST"])
def save_annotations():
    """アノテーションの保存"""
    try:
        init_managers()

        data = request.json
        video_id = data.get("videoId")
        frame_number = data.get("frameNumber")
        annotations = data.get("annotations", [])
        game_state = data.get("gameState", {})

        if not video_id or frame_number is None:
            return jsonify({"error": "Invalid request data"}), 400

        # 既存のアノテーションを読み込み
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if not video_annotation:
            return jsonify({"error": "Video annotation not found"}), 404

        # フレームアノテーションを更新
        if frame_number < len(video_annotation.frames):
            frame = video_annotation.frames[frame_number]

            # ゲーム状態更新
            if game_state:
                frame.game_state = GameStateAnnotation(
                    round_info=f"{game_state.get('round', '東1局')} {game_state.get('honba', 0)}本場",
                    dealer_position=game_state.get("dealer", "東"),
                    remaining_tiles=game_state.get("remainingTiles", 70),
                )

            # 牌アノテーション更新
            frame.tiles = []
            for anno in annotations:
                bbox_data = anno["bbox"]
                from ..annotation_data import BoundingBox

                tile = EnhancedTileAnnotation(
                    tile_id=anno["type"],
                    bbox=BoundingBox(
                        bbox_data["x1"], bbox_data["y1"], bbox_data["x2"], bbox_data["y2"]
                    ),
                    confidence=anno.get("confidence", 1.0),
                    area_type=anno.get("area", "unknown"),
                    player_position=anno.get("player"),
                    annotator=current_session.get("annotator", "unknown"),
                    notes=anno.get("notes", ""),
                )
                frame.tiles.append(tile)

            # 更新時刻
            frame.annotated_at = datetime.now()
            frame.needs_review = False

            # 保存
            success = dataset_manager.save_enhanced_annotation(video_annotation)

            if success:
                return jsonify(
                    {
                        "success": True,
                        "savedCount": len(frame.tiles),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                return jsonify({"error": "Failed to save"}), 500

        return jsonify({"error": "Frame out of range"}), 400

    except Exception as e:
        logger.error(f"保存エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/batch-process", methods=["POST"])
def batch_process():
    """バッチ処理の実行"""
    try:
        init_managers()

        data = request.json
        # video_id = data.get("video_id")  # TODO: 実装時に使用
        start_frame = data.get("start_frame", 0)
        end_frame = data.get("end_frame", 100)
        # options = data.get("options", {})

        # 処理オプション（TODO: 実装時に使用）
        # enable_auto_detect = options.get("auto_detect", True)
        # copy_stable_regions = options.get("copy_stable", True)
        # confidence_threshold = options.get("confidence_threshold", 0.7)

        results = {"processed": 0, "detected": 0, "errors": 0}

        # TODO: 実際のバッチ処理実装
        # 仮の処理結果
        results["processed"] = end_frame - start_frame
        results["detected"] = results["processed"] * 20  # 1フレーム20個想定

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error(f"バッチ処理エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/export/<video_id>")
def export_annotations(video_id: str):
    """アノテーションのエクスポート"""
    try:
        init_managers()

        # アノテーション読み込み
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if not video_annotation:
            return jsonify({"error": "Video annotation not found"}), 404

        # エクスポート形式
        export_format = request.args.get("format", "json")

        if export_format == "json":
            # JSON形式でエクスポート
            export_data = {
                "video_id": video_annotation.video_id,
                "video_name": video_annotation.video_name,
                "frame_count": len(video_annotation.frames),
                "annotation_version": video_annotation.annotation_version,
                "frames": [],
            }

            for frame in video_annotation.frames:
                frame_data = {
                    "frame_number": frame.frame_number,
                    "timestamp": frame.timestamp,
                    "scene_type": frame.scene_annotation.scene_type
                    if frame.scene_annotation
                    else None,
                    "tiles": [
                        {
                            "tile_id": tile.tile_id,
                            "bbox": [tile.bbox.x1, tile.bbox.y1, tile.bbox.x2, tile.bbox.y2],
                            "confidence": tile.confidence,
                            "area_type": tile.area_type,
                            "player": tile.player_position,
                        }
                        for tile in frame.tiles
                    ],
                    "actions": [
                        {
                            "type": action.action_type,
                            "player": action.player_position,
                            "tiles": action.tiles,
                        }
                        for action in frame.detected_actions
                    ],
                }
                export_data["frames"].append(frame_data)

            return jsonify(export_data)

        elif export_format == "yolo":
            # YOLO形式でエクスポート（TODO: 実装）
            return jsonify({"error": "YOLO format export not implemented yet"}), 501

        else:
            return jsonify({"error": "Invalid export format"}), 400

    except Exception as e:
        logger.error(f"エクスポートエラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/statistics/<video_id>")
def get_statistics(video_id: str):
    """アノテーション統計の取得"""
    try:
        init_managers()

        # 進捗情報取得
        progress = dataset_manager.get_annotation_progress(video_id)

        # 詳細統計
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if video_annotation:
            tile_distribution = {}
            action_distribution = {}
            scene_distribution = {}
            skipped_frames = 0

            for frame in video_annotation.frames:
                # スキップされたフレームをカウント
                if frame.metadata.get("skipped", False):
                    skipped_frames += 1
                    continue

                # シーン分布
                if frame.scene_annotation:
                    scene_type = frame.scene_annotation.scene_type
                    scene_distribution[scene_type] = scene_distribution.get(scene_type, 0) + 1

                # 牌分布
                for tile in frame.tiles:
                    tile_distribution[tile.tile_id] = tile_distribution.get(tile.tile_id, 0) + 1

                # アクション分布
                for action in frame.detected_actions:
                    action_distribution[action.action_type] = (
                        action_distribution.get(action.action_type, 0) + 1
                    )

            progress["tile_distribution"] = tile_distribution
            progress["action_distribution"] = action_distribution
            progress["scene_distribution"] = scene_distribution
            progress["skipped_frames"] = skipped_frames
            progress["skip_rate"] = (
                skipped_frames / len(video_annotation.frames) if video_annotation.frames else 0
            )

        return jsonify(progress)

    except Exception as e:
        logger.error(f"統計取得エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/check-similarity", methods=["POST"])
def check_similarity():
    """フレーム類似度チェック"""
    try:
        init_managers()

        data = request.json
        frame_number = data.get("frame_number")
        video_id = data.get("video_id", current_session.get("video_id"))

        if frame_number is None or not video_id:
            return jsonify({"error": "Invalid parameters"}), 400

        # フレーム画像を取得（仮実装）
        # TODO: 実際の画像取得処理
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # 類似度チェック
        similarity_result = annotation_manager.check_frame_similarity(frame)

        return jsonify(
            {
                "is_similar": similarity_result.is_similar,
                "similarity_score": similarity_result.similarity_score,
                "similarity_type": similarity_result.similarity_type,
                "matched_frame_id": similarity_result.matched_frame_id,
                "should_skip": similarity_result.similarity_type in ["identical", "very_similar"],
            }
        )

    except Exception as e:
        logger.error(f"類似度チェックエラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/smart-select", methods=["POST"])
def smart_frame_selection():
    """スマートフレーム選択"""
    try:
        init_managers()

        data = request.json
        video_id = data.get("video_id")
        target_frames = data.get("target_frames")
        selection_mode = data.get("mode", "adaptive")  # 'adaptive', 'uniform', 'important'

        if not video_id:
            return jsonify({"error": "video_id is required"}), 400

        # 動画情報取得
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if not video_annotation:
            return jsonify({"error": "Video not found"}), 404

        selected_frames = []

        if selection_mode == "uniform":
            # 均等間隔で選択
            total_frames = len(video_annotation.frames)
            if target_frames and target_frames < total_frames:
                interval = total_frames / target_frames
                selected_frames = [int(i * interval) for i in range(target_frames)]
            else:
                selected_frames = list(range(total_frames))

        elif selection_mode == "important":
            # 重要なフレームを優先
            importance_scores = []
            for i, frame in enumerate(video_annotation.frames):
                score = 0.0
                if frame.is_key_frame:
                    score += 1.0
                if frame.scene_annotation and frame.scene_annotation.scene_type in [
                    "round_start",
                    "round_end",
                    "riichi",
                    "tsumo",
                    "ron",
                ]:
                    score += 0.5
                if frame.detected_actions:
                    score += 0.3
                importance_scores.append((i, score))

            # スコアでソート
            importance_scores.sort(key=lambda x: x[1], reverse=True)

            # 上位N個を選択
            if target_frames:
                selected_frames = [idx for idx, _ in importance_scores[:target_frames]]
            else:
                selected_frames = [idx for idx, score in importance_scores if score > 0]

        else:  # adaptive
            # 適応的選択（類似度と重要度を考慮）
            previous_frame = None
            for i, frame in enumerate(video_annotation.frames):
                # スキップ済みフレームは除外
                if frame.metadata.get("skipped", False):
                    continue

                # 重要フレームは必ず選択
                if frame.is_key_frame:
                    selected_frames.append(i)
                    previous_frame = i
                    continue

                # 前のフレームとの間隔をチェック
                if previous_frame is None or (i - previous_frame) >= 30:
                    selected_frames.append(i)
                    previous_frame = i

        return jsonify(
            {
                "success": True,
                "selected_frames": sorted(selected_frames),
                "total_selected": len(selected_frames),
                "total_frames": len(video_annotation.frames),
                "selection_rate": len(selected_frames) / len(video_annotation.frames)
                if video_annotation.frames
                else 0,
            }
        )

    except Exception as e:
        logger.error(f"スマート選択エラー: {e}")
        return jsonify({"error": str(e)}), 500


@enhanced_annotation_bp.route("/api/annotation/skip-frame/<int:frame_number>", methods=["POST"])
def skip_frame(frame_number: int):
    """フレームをスキップとしてマーク"""
    try:
        init_managers()

        data = request.json
        video_id = data.get("video_id", current_session.get("video_id"))
        reason = data.get("reason", "manual_skip")

        if not video_id:
            return jsonify({"error": "video_id is required"}), 400

        # アノテーション読み込み
        video_annotation = dataset_manager.load_enhanced_annotation(video_id)
        if not video_annotation or frame_number >= len(video_annotation.frames):
            return jsonify({"error": "Frame not found"}), 404

        # フレームをスキップとしてマーク
        frame = video_annotation.frames[frame_number]
        frame.metadata["skipped"] = True
        frame.metadata["reason"] = reason
        frame.metadata["skip_timestamp"] = datetime.now().isoformat()

        # 保存
        success = dataset_manager.save_enhanced_annotation(video_annotation)

        if success:
            return jsonify({"success": True, "message": f"Frame {frame_number} marked as skipped"})
        else:
            return jsonify({"error": "Failed to save"}), 500

    except Exception as e:
        logger.error(f"スキップ設定エラー: {e}")
        return jsonify({"error": str(e)}), 500
