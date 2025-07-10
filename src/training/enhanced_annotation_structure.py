"""
拡張版アノテーション構造定義

動画から取得可能な全情報を統合した包括的なアノテーション構造
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .annotation_data import BoundingBox, TileAnnotation


@dataclass
class SceneAnnotation:
    """シーン検出アノテーション"""

    scene_type: str  # SceneType enumの値
    confidence: float
    is_transition: bool = False  # シーン遷移中かどうか
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerPosition:
    """プレイヤー位置情報"""

    position: str  # "東", "南", "西", "北"
    player_area: BoundingBox  # プレイヤー全体のエリア
    hand_area: BoundingBox  # 手牌エリア
    discard_area: BoundingBox  # 捨て牌エリア
    call_area: BoundingBox | None = None  # 鳴き牌エリア
    is_active: bool = False  # 現在の手番かどうか
    is_dealer: bool = False  # 親かどうか


@dataclass
class GameStateAnnotation:
    """ゲーム状態アノテーション"""

    round_info: str  # "東1局 0本場"
    dealer_position: str  # "東"
    dora_indicators: list[str] = field(default_factory=list)  # ["5p", "9s"]
    ura_dora_indicators: list[str] = field(default_factory=list)  # 裏ドラ
    remaining_tiles: int = 70  # 残り牌数
    riichi_sticks: int = 0  # 供託リーチ棒
    honba: int = 0  # 本場数
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerInfoAnnotation:
    """プレイヤー情報アノテーション"""

    positions: dict[str, PlayerPosition] = field(default_factory=dict)  # {"東": {...}, "南": {...}}
    current_turn: str | None = None  # "東"
    scores: dict[str, int] = field(default_factory=dict)  # {"東": 25000, "南": 24000, ...}
    riichi_states: dict[str, bool] = field(default_factory=dict)  # {"東": False, "南": True, ...}
    temp_points: dict[str, int] = field(default_factory=dict)  # 一時的な点数（1000点棒など）


@dataclass
class ActionAnnotation:
    """アクションアノテーション"""

    action_type: str  # "draw", "discard", "chi", "pon", "kan", "riichi", "tsumo", "ron"
    player_position: str  # "東", "南", "西", "北"
    tiles: list[str] = field(default_factory=list)  # 関連する牌
    from_player: str | None = None  # 鳴きの場合の相手
    confidence: float = 1.0
    is_inferred: bool = False  # カメラ切り替えによる推測かどうか
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UIElementAnnotation:
    """UI要素アノテーション"""

    element_type: str  # "score_display", "round_info", "dora_display", "timer", etc.
    position: str | None = None  # "東", "南", "西", "北", "center", None
    bbox: BoundingBox | None = None
    text_content: str | None = None  # OCRで読み取ったテキスト
    confidence: float = 1.0


@dataclass
class UIElementsAnnotation:
    """UI要素群アノテーション"""

    elements: list[UIElementAnnotation] = field(default_factory=list)

    def get_elements_by_type(self, element_type: str) -> list[UIElementAnnotation]:
        """指定タイプのUI要素を取得"""
        return [e for e in self.elements if e.element_type == element_type]

    def get_element_by_position(self, position: str) -> list[UIElementAnnotation]:
        """指定位置のUI要素を取得"""
        return [e for e in self.elements if e.position == position]


@dataclass
class EnhancedTileAnnotation(TileAnnotation):
    """拡張版牌アノテーション"""

    player_position: str | None = None  # どのプレイヤーの牌か
    is_dora: bool = False  # ドラかどうか
    is_red_dora: bool = False  # 赤ドラかどうか
    turn_number: int | None = None  # 何巡目に出されたか
    action_context: str | None = None  # "just_drawn", "just_discarded", etc.


@dataclass
class EnhancedFrameAnnotation:
    """拡張版フレームアノテーション"""

    # 基本情報
    frame_id: str
    image_path: str
    image_width: int
    image_height: int
    timestamp: float
    frame_number: int

    # シーン情報
    scene_annotation: SceneAnnotation | None = None

    # ゲーム状態
    game_state: GameStateAnnotation | None = None

    # プレイヤー情報
    player_info: PlayerInfoAnnotation | None = None

    # 牌情報（拡張版）
    tiles: list[EnhancedTileAnnotation] = field(default_factory=list)

    # 検出されたアクション
    detected_actions: list[ActionAnnotation] = field(default_factory=list)

    # UI要素
    ui_elements: UIElementsAnnotation | None = None

    # メタデータ
    quality_score: float = 1.0
    is_valid: bool = True
    is_key_frame: bool = False  # 重要フレーム（局開始、和了など）
    annotated_at: datetime | None = None
    annotator: str = "unknown"
    auto_detected: bool = False  # 自動検出されたかどうか
    needs_review: bool = False  # 人間のレビューが必要か
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後処理"""
        if self.annotated_at is None:
            self.annotated_at = datetime.now()

    @property
    def tile_count(self) -> int:
        """牌の数を取得"""
        return len(self.tiles)

    def get_tiles_by_player(self, player_position: str) -> list[EnhancedTileAnnotation]:
        """指定プレイヤーの牌を取得"""
        return [t for t in self.tiles if t.player_position == player_position]

    def get_tiles_by_area(self, area_type: str) -> list[EnhancedTileAnnotation]:
        """指定エリアの牌を取得"""
        return [t for t in self.tiles if t.area_type == area_type]

    def get_current_player_hand(self) -> list[EnhancedTileAnnotation]:
        """現在のプレイヤーの手牌を取得"""
        if self.player_info and self.player_info.current_turn:
            return [
                t
                for t in self.tiles
                if t.player_position == self.player_info.current_turn and t.area_type == "hand"
            ]
        return []

    def validate(self) -> list[str]:
        """アノテーションの妥当性を検証"""
        errors = []

        # 基本検証
        if not self.frame_id:
            errors.append("frame_id is required")
        if not self.image_path:
            errors.append("image_path is required")

        # 画像サイズ検証
        if self.image_width <= 0 or self.image_height <= 0:
            errors.append("Invalid image dimensions")

        # 牌の重複チェック
        tile_positions = set()
        for tile in self.tiles:
            tile_key = (tile.bbox.center, tile.tile_id)
            if tile_key in tile_positions:
                errors.append(f"Duplicate tile at position {tile.bbox.center}")
            tile_positions.add(tile_key)

        # プレイヤー情報の整合性
        if (
            self.player_info
            and self.player_info.current_turn
            and self.player_info.current_turn
            not in [
                "東",
                "南",
                "西",
                "北",
            ]
        ):
            errors.append(f"Invalid current turn: {self.player_info.current_turn}")

        return errors


@dataclass
class EnhancedVideoAnnotation:
    """拡張版動画アノテーション"""

    # 基本情報
    video_id: str
    video_path: str
    video_name: str
    duration: float
    fps: float
    width: int
    height: int

    # フレーム
    frames: list[EnhancedFrameAnnotation] = field(default_factory=list)

    # ゲーム情報
    game_type: str = "四麻"  # "四麻", "三麻"
    platform: str = "unknown"  # "天鳳", "雀魂", etc.
    players: list[str] = field(default_factory=list)  # プレイヤー名

    # メタデータ
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: str = "2.0"
    annotation_version: str = "enhanced_v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後処理"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    @property
    def frame_count(self) -> int:
        """フレーム数を取得"""
        return len(self.frames)

    @property
    def annotated_frames(self) -> list[EnhancedFrameAnnotation]:
        """アノテーション済みフレームを取得"""
        return [f for f in self.frames if f.tiles or f.detected_actions]

    @property
    def key_frames(self) -> list[EnhancedFrameAnnotation]:
        """重要フレームを取得"""
        return [f for f in self.frames if f.is_key_frame]

    def get_frames_by_scene(self, scene_type: str) -> list[EnhancedFrameAnnotation]:
        """指定シーンタイプのフレームを取得"""
        return [
            f
            for f in self.frames
            if f.scene_annotation and f.scene_annotation.scene_type == scene_type
        ]

    def get_round_frames(self, round_info: str) -> list[EnhancedFrameAnnotation]:
        """指定局のフレームを取得"""
        return [f for f in self.frames if f.game_state and f.game_state.round_info == round_info]

    def get_action_timeline(self) -> list[tuple[float, ActionAnnotation]]:
        """アクションのタイムラインを取得"""
        timeline = []
        for frame in self.frames:
            for action in frame.detected_actions:
                timeline.append((frame.timestamp, action))
        return sorted(timeline, key=lambda x: x[0])
