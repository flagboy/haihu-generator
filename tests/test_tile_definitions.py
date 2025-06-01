"""
麻雀牌定義モジュールのテスト
"""

import pytest
from src.utils.tile_definitions import TileDefinitions, TileType


class TestTileDefinitions:
    """TileDefinitionsクラスのテスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        self.tile_def = TileDefinitions()
    
    def test_get_all_tiles(self):
        """全牌取得のテスト"""
        all_tiles = self.tile_def.get_all_tiles()
        
        # 基本的な牌数の確認（萬子9 + 筒子9 + 索子9 + 字牌7 + 赤ドラ3 = 37）
        assert len(all_tiles) == 37
        
        # 各種類の牌が含まれていることを確認
        assert "1m" in all_tiles
        assert "9p" in all_tiles
        assert "5s" in all_tiles
        assert "東" in all_tiles
        assert "5mr" in all_tiles
    
    def test_get_tiles_by_type(self):
        """種類別牌取得のテスト"""
        # 萬子
        manzu_tiles = self.tile_def.get_tiles_by_type(TileType.MANZU)
        assert len(manzu_tiles) == 9
        assert "1m" in manzu_tiles
        assert "9m" in manzu_tiles
        
        # 筒子
        pinzu_tiles = self.tile_def.get_tiles_by_type(TileType.PINZU)
        assert len(pinzu_tiles) == 9
        assert "1p" in pinzu_tiles
        assert "9p" in pinzu_tiles
        
        # 索子
        souzu_tiles = self.tile_def.get_tiles_by_type(TileType.SOUZU)
        assert len(souzu_tiles) == 9
        assert "1s" in souzu_tiles
        assert "9s" in souzu_tiles
        
        # 字牌
        jihai_tiles = self.tile_def.get_tiles_by_type(TileType.JIHAI)
        assert len(jihai_tiles) == 7
        assert "東" in jihai_tiles
        assert "中" in jihai_tiles
        
        # 赤ドラ
        akadora_tiles = self.tile_def.get_tiles_by_type(TileType.AKADORA)
        assert len(akadora_tiles) == 3
        assert "5mr" in akadora_tiles
        assert "5pr" in akadora_tiles
        assert "5sr" in akadora_tiles
    
    def test_get_tile_type(self):
        """牌種類取得のテスト"""
        assert self.tile_def.get_tile_type("1m") == TileType.MANZU
        assert self.tile_def.get_tile_type("5p") == TileType.PINZU
        assert self.tile_def.get_tile_type("9s") == TileType.SOUZU
        assert self.tile_def.get_tile_type("東") == TileType.JIHAI
        assert self.tile_def.get_tile_type("5mr") == TileType.AKADORA
        assert self.tile_def.get_tile_type("invalid") is None
    
    def test_tile_id_mapping(self):
        """牌IDマッピングのテスト"""
        # IDから牌への変換
        tile_1m = self.tile_def.get_tile_by_id(0)  # 最初の牌
        assert tile_1m == "1m"
        
        # 牌からIDへの変換
        id_1m = self.tile_def.get_tile_id("1m")
        assert id_1m == 0
        
        # 往復変換の確認
        for tile in ["1m", "5p", "9s", "東", "5mr"]:
            tile_id = self.tile_def.get_tile_id(tile)
            recovered_tile = self.tile_def.get_tile_by_id(tile_id)
            assert recovered_tile == tile
        
        # 無効な牌のテスト
        assert self.tile_def.get_tile_id("invalid") == -1
        assert self.tile_def.get_tile_by_id(-1) == ""
    
    def test_is_valid_tile(self):
        """有効牌判定のテスト"""
        # 有効な牌
        assert self.tile_def.is_valid_tile("1m") is True
        assert self.tile_def.is_valid_tile("5p") is True
        assert self.tile_def.is_valid_tile("9s") is True
        assert self.tile_def.is_valid_tile("東") is True
        assert self.tile_def.is_valid_tile("5mr") is True
        
        # 無効な牌
        assert self.tile_def.is_valid_tile("0m") is False
        assert self.tile_def.is_valid_tile("10p") is False
        assert self.tile_def.is_valid_tile("invalid") is False
        assert self.tile_def.is_valid_tile("") is False
    
    def test_is_number_tile(self):
        """数牌判定のテスト"""
        # 数牌
        assert self.tile_def.is_number_tile("1m") is True
        assert self.tile_def.is_number_tile("5p") is True
        assert self.tile_def.is_number_tile("9s") is True
        
        # 数牌以外
        assert self.tile_def.is_number_tile("東") is False
        assert self.tile_def.is_number_tile("5mr") is False
        assert self.tile_def.is_number_tile("invalid") is False
    
    def test_is_honor_tile(self):
        """字牌判定のテスト"""
        # 字牌
        assert self.tile_def.is_honor_tile("東") is True
        assert self.tile_def.is_honor_tile("南") is True
        assert self.tile_def.is_honor_tile("白") is True
        assert self.tile_def.is_honor_tile("中") is True
        
        # 字牌以外
        assert self.tile_def.is_honor_tile("1m") is False
        assert self.tile_def.is_honor_tile("5mr") is False
    
    def test_is_terminal_tile(self):
        """么九牌判定のテスト"""
        # 么九牌（1,9の数牌と字牌）
        assert self.tile_def.is_terminal_tile("1m") is True
        assert self.tile_def.is_terminal_tile("9m") is True
        assert self.tile_def.is_terminal_tile("1p") is True
        assert self.tile_def.is_terminal_tile("9p") is True
        assert self.tile_def.is_terminal_tile("1s") is True
        assert self.tile_def.is_terminal_tile("9s") is True
        assert self.tile_def.is_terminal_tile("東") is True
        assert self.tile_def.is_terminal_tile("中") is True
        
        # 么九牌以外
        assert self.tile_def.is_terminal_tile("2m") is False
        assert self.tile_def.is_terminal_tile("5p") is False
        assert self.tile_def.is_terminal_tile("8s") is False
    
    def test_is_red_dora(self):
        """赤ドラ判定のテスト"""
        # 赤ドラ
        assert self.tile_def.is_red_dora("5mr") is True
        assert self.tile_def.is_red_dora("5pr") is True
        assert self.tile_def.is_red_dora("5sr") is True
        
        # 赤ドラ以外
        assert self.tile_def.is_red_dora("5m") is False
        assert self.tile_def.is_red_dora("5p") is False
        assert self.tile_def.is_red_dora("5s") is False
        assert self.tile_def.is_red_dora("東") is False
    
    def test_get_tile_number(self):
        """牌の数字取得のテスト"""
        # 数牌
        assert self.tile_def.get_tile_number("1m") == 1
        assert self.tile_def.get_tile_number("5p") == 5
        assert self.tile_def.get_tile_number("9s") == 9
        
        # 数牌以外
        assert self.tile_def.get_tile_number("東") == -1
        assert self.tile_def.get_tile_number("5mr") == -1
        assert self.tile_def.get_tile_number("invalid") == -1
    
    def test_get_tile_suit(self):
        """牌のスート取得のテスト"""
        # 数牌
        assert self.tile_def.get_tile_suit("1m") == "m"
        assert self.tile_def.get_tile_suit("5p") == "p"
        assert self.tile_def.get_tile_suit("9s") == "s"
        
        # 数牌以外
        assert self.tile_def.get_tile_suit("東") == ""
        assert self.tile_def.get_tile_suit("5mr") == ""
    
    def test_convert_to_english(self):
        """英語表記変換のテスト"""
        # 字牌の英語変換
        assert self.tile_def.convert_to_english("東") == "E"
        assert self.tile_def.convert_to_english("南") == "S"
        assert self.tile_def.convert_to_english("西") == "W"
        assert self.tile_def.convert_to_english("北") == "N"
        assert self.tile_def.convert_to_english("白") == "P"
        assert self.tile_def.convert_to_english("發") == "F"
        assert self.tile_def.convert_to_english("中") == "C"
        
        # 数牌はそのまま
        assert self.tile_def.convert_to_english("1m") == "1m"
        assert self.tile_def.convert_to_english("5p") == "5p"
    
    def test_get_standard_tile_set(self):
        """標準牌セット取得のテスト"""
        tile_set = self.tile_def.get_standard_tile_set()
        
        # 通常牌は4枚
        assert tile_set["1m"] == 4
        assert tile_set["5p"] == 4
        assert tile_set["東"] == 4
        
        # 赤ドラは1枚
        assert tile_set["5mr"] == 1
        assert tile_set["5pr"] == 1
        assert tile_set["5sr"] == 1
        
        # 総牌数の確認（通常牌34種×4枚 + 赤ドラ3種×1枚 = 139枚）
        total_tiles = sum(tile_set.values())
        assert total_tiles == 139