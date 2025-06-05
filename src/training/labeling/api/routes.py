"""
RESTful APIルーティング
"""

import tempfile
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file
from flask_restx import Api, Namespace, Resource, fields
from loguru import logger

from ..core.hand_area_detector import UnifiedHandAreaDetector
from ..core.labeling_session import LabelingSession
from ..core.tile_splitter import TileSplitter
from ..core.video_processor import EnhancedVideoProcessor

# Blueprintの作成
labeling_bp = Blueprint("labeling", __name__, url_prefix="/api/labeling")

# Flask-RESTXのAPI設定
api = Api(
    labeling_bp,
    version="1.0",
    title="手牌ラベリングAPI",
    description="麻雀牌の学習データ作成のためのAPI",
)

# 名前空間の定義
sessions_ns = Namespace("sessions", description="セッション管理")
frames_ns = Namespace("frames", description="フレーム管理")
hand_areas_ns = Namespace("hand-areas", description="手牌領域管理")
annotations_ns = Namespace("annotations", description="アノテーション管理")

# APIに名前空間を追加
api.add_namespace(sessions_ns)
api.add_namespace(frames_ns)
api.add_namespace(hand_areas_ns)
api.add_namespace(annotations_ns)

# モデル定義
session_model = api.model(
    "Session",
    {
        "session_id": fields.String(required=True, description="セッションID"),
        "video_path": fields.String(required=True, description="動画ファイルパス"),
        "created_at": fields.DateTime(description="作成日時"),
        "updated_at": fields.DateTime(description="更新日時"),
    },
)

hand_area_model = api.model(
    "HandArea",
    {
        "x": fields.Float(required=True, description="X座標（比率）"),
        "y": fields.Float(required=True, description="Y座標（比率）"),
        "w": fields.Float(required=True, description="幅（比率）"),
        "h": fields.Float(required=True, description="高さ（比率）"),
    },
)

tile_annotation_model = api.model(
    "TileAnnotation",
    {
        "index": fields.Integer(required=True, description="牌のインデックス"),
        "label": fields.String(required=True, description="牌のラベル（例: 1m, 2p, 3s）"),
        "x": fields.Integer(required=True, description="X座標"),
        "y": fields.Integer(required=True, description="Y座標"),
        "w": fields.Integer(required=True, description="幅"),
        "h": fields.Integer(required=True, description="高さ"),
        "confidence": fields.Float(description="信頼度スコア"),
    },
)

# グローバル変数（セッション管理）
active_sessions: dict[str, dict[str, Any]] = {}


@sessions_ns.route("/")
class SessionList(Resource):
    @sessions_ns.doc("list_sessions")
    def get(self):
        """セッション一覧を取得"""
        try:
            sessions = LabelingSession.list_sessions()
            return jsonify(sessions)
        except Exception as e:
            logger.error(f"セッション一覧取得エラー: {e}")
            return {"error": str(e)}, 500

    @sessions_ns.doc("create_session")
    @sessions_ns.expect(session_model)
    def post(self):
        """新規セッションを作成"""
        try:
            data = request.json
            video_path = data.get("video_path")

            if not video_path or not Path(video_path).exists():
                return {"error": "有効な動画パスが必要です"}, 400

            # セッションを作成
            session = LabelingSession()

            # 動画処理を初期化
            video_processor = EnhancedVideoProcessor(video_path)

            # セッションに動画情報を設定
            video_info = {
                "path": video_path,
                "fps": video_processor.fps,
                "frame_count": video_processor.frame_count,
                "width": video_processor.width,
                "height": video_processor.height,
                "duration": video_processor.duration,
            }
            session.set_video_info(video_info)

            # アクティブセッションに追加
            active_sessions[session.session_id] = {
                "session": session,
                "video_processor": video_processor,
                "hand_detector": UnifiedHandAreaDetector(),
                "tile_splitter": TileSplitter(),
            }

            return {"session_id": session.session_id, "video_info": video_info}, 201

        except Exception as e:
            logger.error(f"セッション作成エラー: {e}")
            return {"error": str(e)}, 500


@sessions_ns.route("/<string:session_id>")
@sessions_ns.param("session_id", "セッションID")
class SessionDetail(Resource):
    @sessions_ns.doc("get_session")
    def get(self, session_id):
        """セッション詳細を取得"""
        try:
            if session_id not in active_sessions:
                # アクティブでない場合は読み込みを試みる
                session = LabelingSession(session_id=session_id)
                return session.get_session_summary()
            else:
                session = active_sessions[session_id]["session"]
                return session.get_session_summary()
        except Exception as e:
            logger.error(f"セッション取得エラー: {e}")
            return {"error": str(e)}, 404

    @sessions_ns.doc("delete_session")
    def delete(self, session_id):
        """セッションを削除"""
        try:
            if session_id in active_sessions:
                del active_sessions[session_id]
            return {"message": "セッションを削除しました"}, 200
        except Exception as e:
            logger.error(f"セッション削除エラー: {e}")
            return {"error": str(e)}, 500


@frames_ns.route("/<string:session_id>/extract")
@frames_ns.param("session_id", "セッションID")
class FrameExtraction(Resource):
    @frames_ns.doc("extract_frames")
    def post(self, session_id):
        """フレームを抽出"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            data = request.json
            interval = data.get("interval", 1.0)
            start_time = data.get("start_time", 0.0)
            end_time = data.get("end_time", None)

            video_processor = active_sessions[session_id]["video_processor"]

            # フレーム抽出
            extracted_frames = video_processor.extract_frames(
                interval=interval, start_time=start_time, end_time=end_time
            )

            return {"extracted_count": len(extracted_frames), "frames": extracted_frames}, 200

        except Exception as e:
            logger.error(f"フレーム抽出エラー: {e}")
            return {"error": str(e)}, 500


@frames_ns.route("/<string:session_id>/<int:frame_number>")
@frames_ns.param("session_id", "セッションID")
@frames_ns.param("frame_number", "フレーム番号")
class FrameDetail(Resource):
    @frames_ns.doc("get_frame")
    def get(self, session_id, frame_number):
        """指定フレームの画像を取得"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            video_processor = active_sessions[session_id]["video_processor"]
            frame = video_processor.get_frame(frame_number)

            if frame is None:
                return {"error": "フレームが見つかりません"}, 404

            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                import cv2

                cv2.imwrite(tmp.name, frame)
                return send_file(tmp.name, mimetype="image/jpeg")

        except Exception as e:
            logger.error(f"フレーム取得エラー: {e}")
            return {"error": str(e)}, 500


@frames_ns.route("/<string:session_id>/<int:frame_number>/tiles")
@frames_ns.param("session_id", "セッションID")
@frames_ns.param("frame_number", "フレーム番号")
class FrameTiles(Resource):
    @frames_ns.doc("split_tiles")
    def post(self, session_id, frame_number):
        """フレームから牌を分割"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            data = request.json
            player = data.get("player", "bottom")

            ctx = active_sessions[session_id]
            video_processor = ctx["video_processor"]
            hand_detector = ctx["hand_detector"]
            tile_splitter = ctx["tile_splitter"]

            # フレームを取得
            frame = video_processor.get_frame(frame_number)
            if frame is None:
                return {"error": "フレームが見つかりません"}, 404

            # 手牌領域を抽出
            hand_region = hand_detector.extract_hand_region(frame, player)
            if hand_region is None:
                return {"error": "手牌領域が設定されていません"}, 400

            # 牌を分割
            tiles = tile_splitter.split_tiles(hand_region)

            # 位置情報を取得
            positions = tile_splitter.get_tile_positions(tiles, hand_region.shape[1])

            return {"player": player, "tile_count": len(tiles), "positions": positions}, 200

        except Exception as e:
            logger.error(f"牌分割エラー: {e}")
            return {"error": str(e)}, 500


@hand_areas_ns.route("/<string:session_id>")
@hand_areas_ns.param("session_id", "セッションID")
class HandAreas(Resource):
    @hand_areas_ns.doc("get_hand_areas")
    def get(self, session_id):
        """手牌領域設定を取得"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            hand_detector = active_sessions[session_id]["hand_detector"]
            return hand_detector.regions, 200

        except Exception as e:
            logger.error(f"手牌領域取得エラー: {e}")
            return {"error": str(e)}, 500

    @hand_areas_ns.doc("set_hand_areas")
    @hand_areas_ns.expect({"regions": fields.Nested(hand_area_model)})
    def put(self, session_id):
        """手牌領域を設定"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            data = request.json
            regions = data.get("regions", {})

            ctx = active_sessions[session_id]
            hand_detector = ctx["hand_detector"]
            session = ctx["session"]

            # 各プレイヤーの領域を設定
            for player, area in regions.items():
                hand_detector.set_manual_area(player, area)

            # セッションにも保存
            session.set_hand_regions(regions)

            return {"message": "手牌領域を設定しました"}, 200

        except Exception as e:
            logger.error(f"手牌領域設定エラー: {e}")
            return {"error": str(e)}, 500


@annotations_ns.route("/<string:session_id>")
@annotations_ns.param("session_id", "セッションID")
class AnnotationList(Resource):
    @annotations_ns.doc("add_annotation")
    @annotations_ns.expect(
        {
            "frame_number": fields.Integer,
            "player": fields.String,
            "tiles": fields.List(fields.Nested(tile_annotation_model)),
        }
    )
    def post(self, session_id):
        """アノテーションを追加"""
        try:
            if session_id not in active_sessions:
                return {"error": "セッションが見つかりません"}, 404

            data = request.json
            frame_number = data.get("frame_number")
            player = data.get("player")
            tiles = data.get("tiles", [])

            session = active_sessions[session_id]["session"]
            session.add_annotation(frame_number, player, tiles)

            return {
                "message": "アノテーションを追加しました",
                "frame_number": frame_number,
                "player": player,
                "tile_count": len(tiles),
            }, 201

        except Exception as e:
            logger.error(f"アノテーション追加エラー: {e}")
            return {"error": str(e)}, 500


@annotations_ns.route("/<string:session_id>/export")
@annotations_ns.param("session_id", "セッションID")
class AnnotationExport(Resource):
    @annotations_ns.doc("export_annotations")
    def get(self, session_id):
        """アノテーションをエクスポート"""
        try:
            if session_id not in active_sessions:
                # アクティブでない場合は読み込む
                session = LabelingSession(session_id=session_id)
            else:
                session = active_sessions[session_id]["session"]

            format_type = request.args.get("format", "coco")

            # エクスポート実行
            exported_data = session.export_annotations(format=format_type)

            return exported_data, 200

        except Exception as e:
            logger.error(f"エクスポートエラー: {e}")
            return {"error": str(e)}, 500


# エラーハンドラー
@api.errorhandler
def default_error_handler(e):
    """デフォルトのエラーハンドラー"""
    logger.error(f"APIエラー: {e}")
    return {"error": str(e)}, 500
