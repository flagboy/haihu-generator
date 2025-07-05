"""
検出モジュール用の型定義

TypedDictを使用して、各種メタデータの型を明確化
"""

from typing import TypedDict


class SceneMetadata(TypedDict, total=False):
    """シーン検出のメタデータ型"""

    scene_changed: bool
    green_ratio: float
    dark_ratio: float
    top_edge_density: float
    bottom_edge_density: float
    center_edge_density: float
    hand_tiles: int
    discarded_tiles: int
    called_tiles: int
    total_tiles: int


class ScoreMetadata(TypedDict):
    """点数読み取りのメタデータ型"""

    total_scores: int
    valid_scores: int
    ocr_used: bool
    template_matching_used: bool


class PlayerMetadata(TypedDict):
    """プレイヤー検出のメタデータ型"""

    active_player: str | None
    dealer_position: str
    round_wind: str
    turn_indicator_type: str | None


class TileDetectionMetadata(TypedDict):
    """牌検出のメタデータ型"""

    detection_count: int
    hand_tiles_count: int
    discarded_tiles_count: int
    called_tiles_count: int
    avg_confidence: float


class GameContext(TypedDict, total=False):
    """ゲームコンテキスト型"""

    scene_type: str
    scene_confidence: float
    player_positions: dict[str, dict]
    scores: dict[str, int]
    active_player: str | None
    dealer_position: str
    round_wind: str
