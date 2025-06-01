"""
卓状態管理クラス
"""

import copy
import random
from dataclasses import dataclass, field
from enum import Enum

from ..utils.tile_definitions import TileDefinitions


class Wind(Enum):
    """風の種類"""

    EAST = "東"
    SOUTH = "南"
    WEST = "西"
    NORTH = "北"


class GameType(Enum):
    """ゲームタイプ"""

    TONPUU = "東風戦"  # 東風戦
    HANCHAN = "東南戦"  # 東南戦


@dataclass
class TableState:
    """卓の状態情報"""

    round_wind: Wind = Wind.EAST
    round_number: int = 1
    honba: int = 0
    riichi_sticks: int = 0
    game_type: GameType = GameType.HANCHAN
    wall_tiles: list[str] = field(default_factory=list)
    dora_indicators: list[str] = field(default_factory=list)
    ura_dora_indicators: list[str] = field(default_factory=list)
    dead_wall: list[str] = field(default_factory=list)
    remaining_tiles: int = 70
    kan_count: int = 0

    def __post_init__(self):
        """初期化後の処理"""
        if not self.wall_tiles:
            self.wall_tiles = []
        if not self.dora_indicators:
            self.dora_indicators = []
        if not self.ura_dora_indicators:
            self.ura_dora_indicators = []
        if not self.dead_wall:
            self.dead_wall = []


class Table:
    """卓管理クラス"""

    def __init__(self, game_type: GameType = GameType.HANCHAN):
        """
        卓を初期化

        Args:
            game_type: ゲームタイプ（東風戦/東南戦）
        """
        self.tile_definitions = TileDefinitions()
        self.state = TableState(game_type=game_type)
        self._all_tiles = self._create_full_tile_set()
        self._used_tiles: set[str] = set()

    def _create_full_tile_set(self) -> list[str]:
        """完全な牌セット（136枚）を作成"""
        tiles = []
        tile_set = self.tile_definitions.get_standard_tile_set()

        for tile, count in tile_set.items():
            tiles.extend([tile] * count)

        return tiles

    def shuffle_and_deal(self) -> dict[str, list[str]]:
        """
        牌をシャッフルして配牌

        Returns:
            Dict[str, List[str]]: 各プレイヤーの配牌
        """
        # 牌をシャッフル
        shuffled_tiles = self._all_tiles.copy()
        random.shuffle(shuffled_tiles)

        # 王牌（14枚）を分離
        self.state.dead_wall = shuffled_tiles[-14:]
        wall_tiles = shuffled_tiles[:-14]

        # ドラ表示牌を設定
        self.state.dora_indicators = [self.state.dead_wall[5]]  # 最初のドラ表示牌
        self.state.ura_dora_indicators = [self.state.dead_wall[9]]  # 最初の裏ドラ表示牌

        # 配牌（各プレイヤー13枚）
        hands = {
            "east": wall_tiles[0:13],
            "south": wall_tiles[13:26],
            "west": wall_tiles[26:39],
            "north": wall_tiles[39:52],
        }

        # 残りの牌を山牌として設定
        self.state.wall_tiles = wall_tiles[52:]
        self.state.remaining_tiles = len(self.state.wall_tiles)

        return hands

    def draw_tile(self) -> str | None:
        """
        山牌から1枚ツモ

        Returns:
            Optional[str]: ツモった牌（山牌が空の場合はNone）
        """
        if not self.state.wall_tiles:
            return None

        tile = self.state.wall_tiles.pop(0)
        self.state.remaining_tiles = len(self.state.wall_tiles)
        return tile

    def draw_kan_tile(self) -> str | None:
        """
        カン用の牌を王牌から取得

        Returns:
            Optional[str]: カン用の牌（取得できない場合はNone）
        """
        if self.state.kan_count >= 4:
            return None  # カンは4回まで

        # 王牌の末尾から取得
        kan_tile_index = 13 - self.state.kan_count
        if kan_tile_index < 0 or kan_tile_index >= len(self.state.dead_wall):
            return None

        tile = self.state.dead_wall[kan_tile_index]
        self.state.kan_count += 1

        # 新しいドラ表示牌を追加
        new_dora_index = 5 - self.state.kan_count
        if new_dora_index >= 0:
            self.state.dora_indicators.append(self.state.dead_wall[new_dora_index])
            self.state.ura_dora_indicators.append(self.state.dead_wall[new_dora_index + 4])

        return tile

    def get_dora_tiles(self) -> list[str]:
        """
        ドラ牌のリストを取得

        Returns:
            List[str]: ドラ牌のリスト
        """
        dora_tiles = []
        for indicator in self.state.dora_indicators:
            dora_tile = self._get_next_tile(indicator)
            if dora_tile:
                dora_tiles.append(dora_tile)
        return dora_tiles

    def get_ura_dora_tiles(self) -> list[str]:
        """
        裏ドラ牌のリストを取得

        Returns:
            List[str]: 裏ドラ牌のリスト
        """
        ura_dora_tiles = []
        for indicator in self.state.ura_dora_indicators:
            ura_dora_tile = self._get_next_tile(indicator)
            if ura_dora_tile:
                ura_dora_tiles.append(ura_dora_tile)
        return ura_dora_tiles

    def _get_next_tile(self, indicator: str) -> str | None:
        """
        ドラ表示牌から実際のドラ牌を取得

        Args:
            indicator: ドラ表示牌

        Returns:
            Optional[str]: ドラ牌
        """
        if self.tile_definitions.is_number_tile(indicator):
            # 数牌の場合
            number = self.tile_definitions.get_tile_number(indicator)
            suit = self.tile_definitions.get_tile_suit(indicator)
            next_number = 1 if number == 9 else number + 1
            return f"{next_number}{suit}"

        elif self.tile_definitions.is_honor_tile(indicator):
            # 字牌の場合
            wind_cycle = ["東", "南", "西", "北"]
            dragon_cycle = ["白", "發", "中"]

            if indicator in wind_cycle:
                current_index = wind_cycle.index(indicator)
                next_index = (current_index + 1) % 4
                return wind_cycle[next_index]
            elif indicator in dragon_cycle:
                current_index = dragon_cycle.index(indicator)
                next_index = (current_index + 1) % 3
                return dragon_cycle[next_index]

        return None

    def is_wall_empty(self) -> bool:
        """山牌が空かどうかを判定"""
        return len(self.state.wall_tiles) == 0

    def can_continue_game(self) -> bool:
        """ゲームを続行できるかどうかを判定"""
        return self.state.remaining_tiles > 0

    def advance_round(self):
        """次の局に進む"""
        if self.state.round_wind == Wind.EAST:
            if self.state.round_number < 4:
                self.state.round_number += 1
            else:
                if self.state.game_type == GameType.HANCHAN:
                    self.state.round_wind = Wind.SOUTH
                    self.state.round_number = 1
                else:
                    # 東風戦終了
                    pass
        elif self.state.round_wind == Wind.SOUTH:
            if self.state.round_number < 4:
                self.state.round_number += 1
            else:
                # 東南戦終了
                pass

        # 本場とリーチ棒はリセットしない（継続）

    def add_honba(self):
        """本場を追加"""
        self.state.honba += 1

    def add_riichi_stick(self):
        """リーチ棒を追加"""
        self.state.riichi_sticks += 1

    def clear_riichi_sticks(self) -> int:
        """リーチ棒をクリアして枚数を返す"""
        sticks = self.state.riichi_sticks
        self.state.riichi_sticks = 0
        return sticks

    def reset_for_new_round(self):
        """新しい局のためにリセット"""
        self.state.wall_tiles = []
        self.state.dora_indicators = []
        self.state.ura_dora_indicators = []
        self.state.dead_wall = []
        self.state.remaining_tiles = 70
        self.state.kan_count = 0
        self._used_tiles = set()

    def get_current_round_name(self) -> str:
        """現在の局名を取得"""
        wind_names = {Wind.EAST: "東", Wind.SOUTH: "南", Wind.WEST: "西", Wind.NORTH: "北"}
        wind_name = wind_names[self.state.round_wind]
        return f"{wind_name}{self.state.round_number}局"

    def get_state_copy(self) -> TableState:
        """卓状態のコピーを取得"""
        return copy.deepcopy(self.state)

    def restore_state(self, state: TableState):
        """卓状態を復元"""
        self.state = copy.deepcopy(state)

    def __str__(self) -> str:
        """文字列表現"""
        return f"Table({self.get_current_round_name()}, {self.state.honba}本場, 残り{self.state.remaining_tiles}枚)"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"Table(round={self.get_current_round_name()}, honba={self.state.honba}, "
            f"riichi_sticks={self.state.riichi_sticks}, remaining={self.state.remaining_tiles}, "
            f"dora_count={len(self.state.dora_indicators)})"
        )
