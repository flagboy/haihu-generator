"""
拡張版データセット管理システム

動画から取得可能な全情報を管理する包括的なデータベースシステム
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from ..utils.config import ConfigManager
from ..utils.logger import LoggerMixin
from .dataset_manager import DatasetManager
from .enhanced_annotation_structure import (
    ActionAnnotation,
    EnhancedFrameAnnotation,
    EnhancedTileAnnotation,
    EnhancedVideoAnnotation,
    GameStateAnnotation,
    PlayerInfoAnnotation,
    PlayerPosition,
    SceneAnnotation,
    UIElementAnnotation,
    UIElementsAnnotation,
)


class EnhancedDatasetManager(DatasetManager, LoggerMixin):
    """拡張版データセット管理クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """初期化"""
        super().__init__(config_manager)
        self._init_enhanced_database()
        self.logger.info("EnhancedDatasetManager初期化完了")

    def _init_enhanced_database(self):
        """拡張データベースの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # シーン検出テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scene_detections (
                    frame_id TEXT PRIMARY KEY,
                    scene_type TEXT NOT NULL,
                    confidence REAL,
                    is_transition BOOLEAN,
                    metadata TEXT,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # ゲーム状態テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_states (
                    frame_id TEXT PRIMARY KEY,
                    round_info TEXT,
                    dealer_position TEXT,
                    dora_indicators TEXT,  -- JSON配列
                    ura_dora_indicators TEXT,  -- JSON配列
                    remaining_tiles INTEGER,
                    riichi_sticks INTEGER,
                    honba INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # プレイヤー情報テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_info (
                    frame_id TEXT,
                    position TEXT,  -- "東", "南", "西", "北"
                    score INTEGER,
                    is_riichi BOOLEAN,
                    is_current_turn BOOLEAN,
                    is_dealer BOOLEAN,
                    temp_points INTEGER,
                    hand_bbox_x1 INTEGER,
                    hand_bbox_y1 INTEGER,
                    hand_bbox_x2 INTEGER,
                    hand_bbox_y2 INTEGER,
                    discard_bbox_x1 INTEGER,
                    discard_bbox_y1 INTEGER,
                    discard_bbox_x2 INTEGER,
                    discard_bbox_y2 INTEGER,
                    call_bbox_x1 INTEGER,
                    call_bbox_y1 INTEGER,
                    call_bbox_x2 INTEGER,
                    call_bbox_y2 INTEGER,
                    PRIMARY KEY (frame_id, position),
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # アクションテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    frame_id TEXT,
                    action_type TEXT,
                    player_position TEXT,
                    tiles TEXT,  -- JSON配列
                    from_player TEXT,
                    confidence REAL,
                    is_inferred BOOLEAN,
                    timestamp REAL,
                    metadata TEXT,
                    sequence_number INTEGER,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # UI要素テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ui_elements (
                    id TEXT PRIMARY KEY,
                    frame_id TEXT,
                    element_type TEXT,
                    position TEXT,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    text_content TEXT,
                    confidence REAL,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # 拡張牌アノテーションテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_tile_annotations (
                    id TEXT PRIMARY KEY,
                    frame_id TEXT,
                    tile_id TEXT NOT NULL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    confidence REAL,
                    area_type TEXT,
                    is_face_up BOOLEAN,
                    is_occluded BOOLEAN,
                    occlusion_ratio REAL,
                    player_position TEXT,
                    is_dora BOOLEAN,
                    is_red_dora BOOLEAN,
                    turn_number INTEGER,
                    action_context TEXT,
                    annotator TEXT,
                    notes TEXT,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # フレームメタデータテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_metadata (
                    frame_id TEXT PRIMARY KEY,
                    is_key_frame BOOLEAN,
                    auto_detected BOOLEAN,
                    needs_review BOOLEAN,
                    frame_number INTEGER,
                    metadata TEXT,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # インデックス作成
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_scene_type ON scene_detections (scene_type)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_round_info ON game_states (round_info)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_type ON actions (action_type)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_player_position ON player_info (position)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_key_frames ON frame_metadata (is_key_frame)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_needs_review ON frame_metadata (needs_review)"
            )

            conn.commit()

    def save_enhanced_annotation(self, video_annotation: EnhancedVideoAnnotation) -> bool:
        """
        拡張アノテーションデータを保存

        Args:
            video_annotation: 拡張版動画アノテーション

        Returns:
            保存成功かどうか
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 動画情報を保存
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO videos
                    (id, name, path, duration, fps, width, height,
                     created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        video_annotation.video_id,
                        video_annotation.video_name,
                        video_annotation.video_path,
                        video_annotation.duration,
                        video_annotation.fps,
                        video_annotation.width,
                        video_annotation.height,
                        video_annotation.created_at.isoformat()
                        if video_annotation.created_at
                        else None,
                        video_annotation.updated_at.isoformat()
                        if video_annotation.updated_at
                        else None,
                        json.dumps(
                            {
                                "game_type": video_annotation.game_type,
                                "platform": video_annotation.platform,
                                "players": video_annotation.players,
                                "annotation_version": video_annotation.annotation_version,
                                **video_annotation.metadata,
                            }
                        ),
                    ),
                )

                # フレーム情報を保存
                for frame in video_annotation.frames:
                    self._save_enhanced_frame(cursor, video_annotation.video_id, frame)

                conn.commit()
                self.logger.info(f"拡張アノテーションデータを保存: {video_annotation.video_name}")
                return True

        except Exception as e:
            self.logger.error(f"拡張アノテーションデータの保存に失敗: {e}")
            return False

    def _save_enhanced_frame(
        self, cursor: sqlite3.Cursor, video_id: str, frame: EnhancedFrameAnnotation
    ):
        """拡張フレーム情報を保存"""
        # 基本フレーム情報
        cursor.execute(
            """
            INSERT OR REPLACE INTO frames
            (id, video_id, image_path, timestamp, width, height, quality_score,
             is_valid, scene_type, game_phase, annotated_at, annotator, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                frame.frame_id,
                video_id,
                frame.image_path,
                frame.timestamp,
                frame.image_width,
                frame.image_height,
                frame.quality_score,
                frame.is_valid,
                frame.scene_annotation.scene_type if frame.scene_annotation else None,
                None,  # game_phaseは後で削除予定
                frame.annotated_at.isoformat() if frame.annotated_at else None,
                frame.annotator,
                frame.notes,
            ),
        )

        # シーン情報
        if frame.scene_annotation:
            cursor.execute(
                """
                INSERT OR REPLACE INTO scene_detections
                (frame_id, scene_type, confidence, is_transition, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    frame.frame_id,
                    frame.scene_annotation.scene_type,
                    frame.scene_annotation.confidence,
                    frame.scene_annotation.is_transition,
                    json.dumps(frame.scene_annotation.metadata),
                ),
            )

        # ゲーム状態
        if frame.game_state:
            cursor.execute(
                """
                INSERT OR REPLACE INTO game_states
                (frame_id, round_info, dealer_position, dora_indicators,
                 ura_dora_indicators, remaining_tiles, riichi_sticks, honba, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    frame.frame_id,
                    frame.game_state.round_info,
                    frame.game_state.dealer_position,
                    json.dumps(frame.game_state.dora_indicators),
                    json.dumps(frame.game_state.ura_dora_indicators),
                    frame.game_state.remaining_tiles,
                    frame.game_state.riichi_sticks,
                    frame.game_state.honba,
                    json.dumps(frame.game_state.metadata),
                ),
            )

        # プレイヤー情報
        if frame.player_info:
            for position, player_pos in frame.player_info.positions.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO player_info
                    (frame_id, position, score, is_riichi, is_current_turn, is_dealer,
                     temp_points, hand_bbox_x1, hand_bbox_y1, hand_bbox_x2, hand_bbox_y2,
                     discard_bbox_x1, discard_bbox_y1, discard_bbox_x2, discard_bbox_y2,
                     call_bbox_x1, call_bbox_y1, call_bbox_x2, call_bbox_y2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        frame.frame_id,
                        position,
                        frame.player_info.scores.get(position, 25000),
                        frame.player_info.riichi_states.get(position, False),
                        position == frame.player_info.current_turn,
                        player_pos.is_dealer,
                        frame.player_info.temp_points.get(position, 0),
                        player_pos.hand_area.x1,
                        player_pos.hand_area.y1,
                        player_pos.hand_area.x2,
                        player_pos.hand_area.y2,
                        player_pos.discard_area.x1,
                        player_pos.discard_area.y1,
                        player_pos.discard_area.x2,
                        player_pos.discard_area.y2,
                        player_pos.call_area.x1 if player_pos.call_area else None,
                        player_pos.call_area.y1 if player_pos.call_area else None,
                        player_pos.call_area.x2 if player_pos.call_area else None,
                        player_pos.call_area.y2 if player_pos.call_area else None,
                    ),
                )

        # アクション情報
        for i, action in enumerate(frame.detected_actions):
            action_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT OR REPLACE INTO actions
                (id, frame_id, action_type, player_position, tiles, from_player,
                 confidence, is_inferred, timestamp, metadata, sequence_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    action_id,
                    frame.frame_id,
                    action.action_type,
                    action.player_position,
                    json.dumps(action.tiles),
                    action.from_player,
                    action.confidence,
                    action.is_inferred,
                    action.timestamp,
                    json.dumps(action.metadata),
                    i,
                ),
            )

        # UI要素
        if frame.ui_elements:
            for element in frame.ui_elements.elements:
                element_id = str(uuid.uuid4())
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO ui_elements
                    (id, frame_id, element_type, position, bbox_x1, bbox_y1,
                     bbox_x2, bbox_y2, text_content, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        element_id,
                        frame.frame_id,
                        element.element_type,
                        element.position,
                        element.bbox.x1 if element.bbox else None,
                        element.bbox.y1 if element.bbox else None,
                        element.bbox.x2 if element.bbox else None,
                        element.bbox.y2 if element.bbox else None,
                        element.text_content,
                        element.confidence,
                    ),
                )

        # 拡張牌アノテーション
        for tile in frame.tiles:
            tile_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT OR REPLACE INTO enhanced_tile_annotations
                (id, frame_id, tile_id, x1, y1, x2, y2, confidence, area_type,
                 is_face_up, is_occluded, occlusion_ratio, player_position,
                 is_dora, is_red_dora, turn_number, action_context,
                 annotator, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tile_id,
                    frame.frame_id,
                    tile.tile_id,
                    tile.bbox.x1,
                    tile.bbox.y1,
                    tile.bbox.x2,
                    tile.bbox.y2,
                    tile.confidence,
                    tile.area_type,
                    tile.is_face_up,
                    tile.is_occluded,
                    tile.occlusion_ratio,
                    tile.player_position,
                    tile.is_dora,
                    tile.is_red_dora,
                    tile.turn_number,
                    tile.action_context,
                    tile.annotator,
                    tile.notes,
                ),
            )

        # フレームメタデータ
        cursor.execute(
            """
            INSERT OR REPLACE INTO frame_metadata
            (frame_id, is_key_frame, auto_detected, needs_review,
             frame_number, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                frame.frame_id,
                frame.is_key_frame,
                frame.auto_detected,
                frame.needs_review,
                frame.frame_number,
                json.dumps(frame.metadata),
            ),
        )

    def load_enhanced_annotation(self, video_id: str) -> EnhancedVideoAnnotation | None:
        """
        拡張アノテーションデータを読み込み

        Args:
            video_id: 動画ID

        Returns:
            拡張版動画アノテーション
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 動画情報を取得
                cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
                video_row = cursor.fetchone()

                if not video_row:
                    return None

                metadata = json.loads(video_row[9]) if video_row[9] else {}

                video_annotation = EnhancedVideoAnnotation(
                    video_id=video_row[0],
                    video_path=video_row[2],
                    video_name=video_row[1],
                    duration=video_row[3] or 0.0,
                    fps=video_row[4] or 30.0,
                    width=video_row[5] or 1920,
                    height=video_row[6] or 1080,
                    frames=[],
                    game_type=metadata.get("game_type", "四麻"),
                    platform=metadata.get("platform", "unknown"),
                    players=metadata.get("players", []),
                    created_at=datetime.fromisoformat(video_row[7]) if video_row[7] else None,
                    updated_at=datetime.fromisoformat(video_row[8]) if video_row[8] else None,
                    annotation_version=metadata.get("annotation_version", "enhanced_v1"),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["game_type", "platform", "players", "annotation_version"]
                    },
                )

                # フレーム情報を取得
                cursor.execute("SELECT * FROM frames WHERE video_id = ?", (video_id,))
                frame_rows = cursor.fetchall()

                for frame_row in frame_rows:
                    frame = self._load_enhanced_frame(cursor, frame_row)
                    video_annotation.frames.append(frame)

                return video_annotation

        except Exception as e:
            self.logger.error(f"拡張アノテーションデータの読み込みに失敗: {e}")
            return None

    def _load_enhanced_frame(
        self, cursor: sqlite3.Cursor, frame_row: tuple
    ) -> EnhancedFrameAnnotation:
        """拡張フレーム情報を読み込み"""
        frame_id = frame_row[0]

        # シーン情報を取得
        cursor.execute("SELECT * FROM scene_detections WHERE frame_id = ?", (frame_id,))
        scene_row = cursor.fetchone()
        scene_annotation = None
        if scene_row:
            scene_annotation = SceneAnnotation(
                scene_type=scene_row[1],
                confidence=scene_row[2] or 1.0,
                is_transition=bool(scene_row[3]),
                metadata=json.loads(scene_row[4]) if scene_row[4] else {},
            )

        # ゲーム状態を取得
        cursor.execute("SELECT * FROM game_states WHERE frame_id = ?", (frame_id,))
        game_row = cursor.fetchone()
        game_state = None
        if game_row:
            game_state = GameStateAnnotation(
                round_info=game_row[1] or "",
                dealer_position=game_row[2] or "",
                dora_indicators=json.loads(game_row[3]) if game_row[3] else [],
                ura_dora_indicators=json.loads(game_row[4]) if game_row[4] else [],
                remaining_tiles=game_row[5] or 70,
                riichi_sticks=game_row[6] or 0,
                honba=game_row[7] or 0,
                metadata=json.loads(game_row[8]) if game_row[8] else {},
            )

        # プレイヤー情報を取得
        cursor.execute("SELECT * FROM player_info WHERE frame_id = ?", (frame_id,))
        player_rows = cursor.fetchall()
        player_info = None
        if player_rows:
            positions = {}
            scores = {}
            riichi_states = {}
            temp_points = {}
            current_turn = None

            for p_row in player_rows:
                position = p_row[1]
                scores[position] = p_row[2] or 25000
                riichi_states[position] = bool(p_row[3])
                temp_points[position] = p_row[6] or 0

                if p_row[4]:  # is_current_turn
                    current_turn = position

                from .annotation_data import BoundingBox

                positions[position] = PlayerPosition(
                    position=position,
                    player_area=BoundingBox(0, 0, 0, 0),  # TODO: プレイヤー全体エリア
                    hand_area=BoundingBox(p_row[7], p_row[8], p_row[9], p_row[10]),
                    discard_area=BoundingBox(p_row[11], p_row[12], p_row[13], p_row[14]),
                    call_area=BoundingBox(p_row[15], p_row[16], p_row[17], p_row[18])
                    if p_row[15] is not None
                    else None,
                    is_active=bool(p_row[4]),
                    is_dealer=bool(p_row[5]),
                )

            player_info = PlayerInfoAnnotation(
                positions=positions,
                current_turn=current_turn,
                scores=scores,
                riichi_states=riichi_states,
                temp_points=temp_points,
            )

        # アクション情報を取得
        cursor.execute(
            "SELECT * FROM actions WHERE frame_id = ? ORDER BY sequence_number", (frame_id,)
        )
        action_rows = cursor.fetchall()
        actions = []
        for a_row in action_rows:
            actions.append(
                ActionAnnotation(
                    action_type=a_row[2],
                    player_position=a_row[3],
                    tiles=json.loads(a_row[4]) if a_row[4] else [],
                    from_player=a_row[5],
                    confidence=a_row[6] or 1.0,
                    is_inferred=bool(a_row[7]),
                    timestamp=a_row[8] or 0.0,
                    metadata=json.loads(a_row[9]) if a_row[9] else {},
                )
            )

        # UI要素を取得
        cursor.execute("SELECT * FROM ui_elements WHERE frame_id = ?", (frame_id,))
        ui_rows = cursor.fetchall()
        ui_elements = None
        if ui_rows:
            elements = []
            for u_row in ui_rows:
                from .annotation_data import BoundingBox

                bbox = None
                if u_row[4] is not None:
                    bbox = BoundingBox(u_row[4], u_row[5], u_row[6], u_row[7])

                elements.append(
                    UIElementAnnotation(
                        element_type=u_row[2],
                        position=u_row[3],
                        bbox=bbox,
                        text_content=u_row[8],
                        confidence=u_row[9] or 1.0,
                    )
                )
            ui_elements = UIElementsAnnotation(elements=elements)

        # 拡張牌アノテーションを取得
        cursor.execute("SELECT * FROM enhanced_tile_annotations WHERE frame_id = ?", (frame_id,))
        tile_rows = cursor.fetchall()
        tiles = []
        for t_row in tile_rows:
            from .annotation_data import BoundingBox

            tiles.append(
                EnhancedTileAnnotation(
                    tile_id=t_row[2],
                    bbox=BoundingBox(t_row[3], t_row[4], t_row[5], t_row[6]),
                    confidence=t_row[7] or 1.0,
                    area_type=t_row[8] or "unknown",
                    is_face_up=bool(t_row[9]) if t_row[9] is not None else True,
                    is_occluded=bool(t_row[10]) if t_row[10] is not None else False,
                    occlusion_ratio=t_row[11] or 0.0,
                    player_position=t_row[12],
                    is_dora=bool(t_row[13]),
                    is_red_dora=bool(t_row[14]),
                    turn_number=t_row[15],
                    action_context=t_row[16],
                    annotator=t_row[17] or "unknown",
                    notes=t_row[18] or "",
                )
            )

        # フレームメタデータを取得
        cursor.execute("SELECT * FROM frame_metadata WHERE frame_id = ?", (frame_id,))
        meta_row = cursor.fetchone()

        frame = EnhancedFrameAnnotation(
            frame_id=frame_row[0],
            image_path=frame_row[2],
            image_width=frame_row[4] or 1920,
            image_height=frame_row[5] or 1080,
            timestamp=frame_row[3] or 0.0,
            frame_number=meta_row[4] if meta_row else 0,
            scene_annotation=scene_annotation,
            game_state=game_state,
            player_info=player_info,
            tiles=tiles,
            detected_actions=actions,
            ui_elements=ui_elements,
            quality_score=frame_row[6] or 1.0,
            is_valid=bool(frame_row[7]) if frame_row[7] is not None else True,
            is_key_frame=bool(meta_row[1]) if meta_row else False,
            auto_detected=bool(meta_row[2]) if meta_row else False,
            needs_review=bool(meta_row[3]) if meta_row else False,
            annotated_at=datetime.fromisoformat(frame_row[10]) if frame_row[10] else None,
            annotator=frame_row[11] or "unknown",
            notes=frame_row[12] or "",
            metadata=json.loads(meta_row[5]) if meta_row and meta_row[5] else {},
        )

        return frame

    def get_frames_needing_review(self, video_id: str | None = None) -> list[dict]:
        """
        レビューが必要なフレームを取得

        Args:
            video_id: 特定の動画IDでフィルタ（Noneの場合は全動画）

        Returns:
            レビューが必要なフレームのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if video_id:
                    query = """
                        SELECT f.id, f.video_id, f.image_path, f.timestamp,
                               fm.needs_review, fm.auto_detected, f.notes
                        FROM frames f
                        JOIN frame_metadata fm ON f.id = fm.frame_id
                        WHERE f.video_id = ? AND fm.needs_review = 1
                        ORDER BY f.timestamp
                    """
                    cursor.execute(query, (video_id,))
                else:
                    query = """
                        SELECT f.id, f.video_id, f.image_path, f.timestamp,
                               fm.needs_review, fm.auto_detected, f.notes
                        FROM frames f
                        JOIN frame_metadata fm ON f.id = fm.frame_id
                        WHERE fm.needs_review = 1
                        ORDER BY f.video_id, f.timestamp
                    """
                    cursor.execute(query)

                frames = []
                for row in cursor.fetchall():
                    frames.append(
                        {
                            "frame_id": row[0],
                            "video_id": row[1],
                            "image_path": row[2],
                            "timestamp": row[3],
                            "needs_review": bool(row[4]),
                            "auto_detected": bool(row[5]),
                            "notes": row[6] or "",
                        }
                    )

                return frames

        except Exception as e:
            self.logger.error(f"レビュー必要フレームの取得に失敗: {e}")
            return []

    def get_key_frames(self, video_id: str) -> list[dict]:
        """
        重要フレームを取得

        Args:
            video_id: 動画ID

        Returns:
            重要フレームのリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT f.id, f.image_path, f.timestamp,
                           sd.scene_type, gs.round_info
                    FROM frames f
                    JOIN frame_metadata fm ON f.id = fm.frame_id
                    LEFT JOIN scene_detections sd ON f.id = sd.frame_id
                    LEFT JOIN game_states gs ON f.id = gs.frame_id
                    WHERE f.video_id = ? AND fm.is_key_frame = 1
                    ORDER BY f.timestamp
                """
                cursor.execute(query, (video_id,))

                frames = []
                for row in cursor.fetchall():
                    frames.append(
                        {
                            "frame_id": row[0],
                            "image_path": row[1],
                            "timestamp": row[2],
                            "scene_type": row[3],
                            "round_info": row[4],
                        }
                    )

                return frames

        except Exception as e:
            self.logger.error(f"重要フレームの取得に失敗: {e}")
            return []

    def get_annotation_progress(self, video_id: str | None = None) -> dict[str, Any]:
        """
        アノテーション進捗を取得

        Args:
            video_id: 特定の動画IDでフィルタ（Noneの場合は全動画）

        Returns:
            進捗情報
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if video_id:
                    # 特定動画の進捗
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM frames WHERE video_id = ?
                    """,
                        (video_id,),
                    )
                    total_frames = cursor.fetchone()[0]

                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM frames f
                        JOIN frame_metadata fm ON f.id = fm.frame_id
                        WHERE f.video_id = ? AND fm.auto_detected = 1
                    """,
                        (video_id,),
                    )
                    auto_detected = cursor.fetchone()[0]

                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM frames f
                        JOIN frame_metadata fm ON f.id = fm.frame_id
                        WHERE f.video_id = ? AND fm.needs_review = 1
                    """,
                        (video_id,),
                    )
                    needs_review = cursor.fetchone()[0]

                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM enhanced_tile_annotations eta
                        JOIN frames f ON eta.frame_id = f.id
                        WHERE f.video_id = ?
                    """,
                        (video_id,),
                    )
                    total_tiles = cursor.fetchone()[0]

                else:
                    # 全体の進捗
                    cursor.execute("SELECT COUNT(*) FROM frames")
                    total_frames = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT COUNT(*) FROM frame_metadata WHERE auto_detected = 1
                    """)
                    auto_detected = cursor.fetchone()[0]

                    cursor.execute("""
                        SELECT COUNT(*) FROM frame_metadata WHERE needs_review = 1
                    """)
                    needs_review = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM enhanced_tile_annotations")
                    total_tiles = cursor.fetchone()[0]

                return {
                    "total_frames": total_frames,
                    "auto_detected_frames": auto_detected,
                    "needs_review_frames": needs_review,
                    "completed_frames": total_frames - needs_review,
                    "total_tiles_annotated": total_tiles,
                    "completion_rate": (total_frames - needs_review) / total_frames
                    if total_frames > 0
                    else 0.0,
                }

        except Exception as e:
            self.logger.error(f"進捗情報の取得に失敗: {e}")
            return {}
