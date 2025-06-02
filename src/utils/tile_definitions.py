"""
麻雀牌定義モジュール
"""

from enum import Enum


class TileType(Enum):
    """牌の種類"""

    MANZU = "manzu"  # 萬子
    PINZU = "pinzu"  # 筒子
    SOUZU = "souzu"  # 索子
    JIHAI = "jihai"  # 字牌
    AKADORA = "akadora"  # 赤ドラ


class TileDefinitions:
    """麻雀牌の定義と操作を管理するクラス"""

    # 基本牌定義
    MANZU_TILES = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]
    PINZU_TILES = ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"]
    SOUZU_TILES = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s"]
    JIHAI_TILES = ["東", "南", "西", "北", "白", "發", "中"]
    AKADORA_TILES = ["5mr", "5pr", "5sr"]

    # 天鳳記法の字牌マッピング
    JIHAI_TENHOU = {
        "東": "1z",
        "南": "2z",
        "西": "3z",
        "北": "4z",
        "白": "5z",
        "發": "6z",
        "中": "7z",
    }

    # 天鳳記法の赤ドラマッピング
    AKADORA_TENHOU = {"5mr": "0m", "5pr": "0p", "5sr": "0s"}

    # 字牌の英語表記
    JIHAI_ENGLISH = {"東": "E", "南": "S", "西": "W", "北": "N", "白": "P", "發": "F", "中": "C"}

    def __init__(self):
        """牌定義クラスの初期化"""
        self._all_tiles = self._create_all_tiles()
        self._tile_to_type = self._create_tile_type_mapping()
        self._tile_to_id = self._create_tile_id_mapping()
        self._id_to_tile = {v: k for k, v in self._tile_to_id.items()}

    def _create_all_tiles(self) -> list[str]:
        """全ての牌のリストを作成"""
        return (
            self.MANZU_TILES
            + self.PINZU_TILES
            + self.SOUZU_TILES
            + self.JIHAI_TILES
            + self.AKADORA_TILES
        )

    def _create_tile_type_mapping(self) -> dict[str, TileType]:
        """牌から種類へのマッピングを作成"""
        mapping = {}

        for tile in self.MANZU_TILES:
            mapping[tile] = TileType.MANZU
        for tile in self.PINZU_TILES:
            mapping[tile] = TileType.PINZU
        for tile in self.SOUZU_TILES:
            mapping[tile] = TileType.SOUZU
        for tile in self.JIHAI_TILES:
            mapping[tile] = TileType.JIHAI
        for tile in self.AKADORA_TILES:
            mapping[tile] = TileType.AKADORA

        return mapping

    def _create_tile_id_mapping(self) -> dict[str, int]:
        """牌からIDへのマッピングを作成"""
        mapping = {}
        for tile_id, tile in enumerate(self._all_tiles):
            mapping[tile] = tile_id

        return mapping

    def get_all_tiles(self) -> list[str]:
        """全ての牌を取得"""
        return self._all_tiles.copy()

    def get_tiles_by_type(self, tile_type: TileType) -> list[str]:
        """指定された種類の牌を取得"""
        if tile_type == TileType.MANZU:
            return self.MANZU_TILES.copy()
        elif tile_type == TileType.PINZU:
            return self.PINZU_TILES.copy()
        elif tile_type == TileType.SOUZU:
            return self.SOUZU_TILES.copy()
        elif tile_type == TileType.JIHAI:
            return self.JIHAI_TILES.copy()
        elif tile_type == TileType.AKADORA:
            return self.AKADORA_TILES.copy()
        else:
            return []

    def get_tile_type(self, tile: str) -> TileType:
        """牌の種類を取得"""
        return self._tile_to_type.get(tile)

    def get_tile_id(self, tile: str) -> int:
        """牌のIDを取得"""
        return self._tile_to_id.get(tile, -1)

    def get_tile_by_id(self, tile_id: int) -> str:
        """IDから牌を取得"""
        return self._id_to_tile.get(tile_id, "")

    def is_valid_tile(self, tile: str) -> bool:
        """有効な牌かどうかを判定"""
        return tile in self._all_tiles

    def is_number_tile(self, tile: str) -> bool:
        """数牌かどうかを判定"""
        tile_type = self.get_tile_type(tile)
        return tile_type in [TileType.MANZU, TileType.PINZU, TileType.SOUZU]

    def is_honor_tile(self, tile: str) -> bool:
        """字牌かどうかを判定"""
        return self.get_tile_type(tile) == TileType.JIHAI

    def is_terminal_tile(self, tile: str) -> bool:
        """么九牌かどうかを判定"""
        if self.is_honor_tile(tile):
            return True
        if self.is_number_tile(tile):
            return tile[0] in ["1", "9"]
        return False

    def is_red_dora(self, tile: str) -> bool:
        """赤ドラかどうかを判定"""
        return self.get_tile_type(tile) == TileType.AKADORA

    def get_tile_number(self, tile: str) -> int:
        """数牌の数字を取得（数牌以外は-1）"""
        if self.is_number_tile(tile) and len(tile) >= 2:
            try:
                return int(tile[0])
            except ValueError:
                return -1
        return -1

    def get_tile_suit(self, tile: str) -> str:
        """数牌のスートを取得（m/p/s）"""
        if self.is_number_tile(tile) and len(tile) >= 2:
            return tile[1]
        return ""

    def convert_to_english(self, tile: str) -> str:
        """牌を英語表記に変換"""
        if tile in self.JIHAI_ENGLISH:
            return self.JIHAI_ENGLISH[tile]
        return tile

    def get_tile_count(self) -> int:
        """牌の総数を取得"""
        return len(self._all_tiles)

    def get_standard_tile_set(self) -> dict[str, int]:
        """標準的な牌セット（各牌4枚）を取得"""
        tile_set = {}
        for tile in self._all_tiles:
            if self.is_red_dora(tile):
                tile_set[tile] = 1  # 赤ドラは1枚
            else:
                tile_set[tile] = 4  # 通常牌は4枚
        return tile_set

    def get_tenhou_notation(self, tile_id: int) -> str:
        """牌IDから天鳳記法を取得"""
        tile = self.get_tile_by_id(tile_id)
        return self.convert_to_tenhou_notation(tile)

    def convert_to_tenhou_notation(self, tile: str) -> str:
        """牌を天鳳記法に変換"""
        if not tile:
            return ""

        # 既に天鳳記法の場合
        if tile.endswith(("m", "p", "s", "z")) and len(tile) == 2:
            return tile

        # 字牌の変換
        if tile in self.JIHAI_TENHOU:
            return self.JIHAI_TENHOU[tile]

        # 赤ドラの変換
        if tile in self.AKADORA_TENHOU:
            return self.AKADORA_TENHOU[tile]

        # 数牌の変換（既に正しい形式の場合）
        if self.is_number_tile(tile):
            return tile

        # その他の場合はそのまま返す
        return tile

    def convert_from_tenhou_notation(self, tenhou_tile: str) -> str:
        """天鳳記法から標準記法に変換"""
        if not tenhou_tile or len(tenhou_tile) != 2:
            return tenhou_tile

        # 赤ドラの逆変換
        for standard, tenhou in self.AKADORA_TENHOU.items():
            if tenhou_tile == tenhou:
                return standard

        # 字牌の逆変換
        for standard, tenhou in self.JIHAI_TENHOU.items():
            if tenhou_tile == tenhou:
                return standard

        # 数牌はそのまま
        if tenhou_tile.endswith(("m", "p", "s")) and tenhou_tile[0].isdigit():
            return tenhou_tile

        return tenhou_tile

    def is_tenhou_notation(self, tile: str) -> bool:
        """天鳳記法かどうかを判定"""
        if not tile or len(tile) != 2:
            return False

        # 数牌の天鳳記法チェック
        if tile.endswith(("m", "p", "s")):
            return tile[0] in "0123456789"

        # 字牌の天鳳記法チェック
        if tile.endswith("z"):
            return tile[0] in "1234567"

        return False
