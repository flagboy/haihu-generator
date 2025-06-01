"""
天鳳ゲームデータモデルのテスト
"""

import pytest
from datetime import datetime
from dataclasses import FrozenInstanceError

from src.models.tenhou_game_data import (
    TenhouTile, TenhouAction, TenhouDrawAction, TenhouDiscardAction,
    TenhouCallAction, TenhouRiichiAction, TenhouAgariAction,
    TenhouPlayerState, TenhouGameRule, TenhouGameResult, TenhouGameData,
    TenhouGameDataBuilder, TenhouActionType, TenhouCallType, TenhouGameType
)


class TestTenhouTile:
    """天鳳牌データのテストクラス"""
    
    def test_tile_creation_basic(self):
        """基本的な牌作成のテスト"""
        tile = TenhouTile("1m")
        assert tile.notation == "1m"
        assert tile.original == "1m"
        assert tile.is_red_dora is False
    
    def test_tile_creation_red_dora(self):
        """赤ドラ牌作成のテスト"""
        tile = TenhouTile("0m")
        assert tile.notation == "0m"
        assert tile.is_red_dora is True
    
    def test_tile_creation_with_original(self):
        """元記法指定での牌作成テスト"""
        tile = TenhouTile("1z", original="東")
        assert tile.notation == "1z"
        assert tile.original == "東"
        assert tile.is_red_dora is False
    
    def test_to_standard_notation(self):
        """標準記法変換のテスト"""
        # 数牌
        tile1 = TenhouTile("5m")
        assert tile1.to_standard_notation() == "5m"
        
        # 字牌
        tile2 = TenhouTile("1z")
        standard = tile2.to_standard_notation()
        assert standard == "東"
        
        # 赤ドラ
        tile3 = TenhouTile("0p")
        standard = tile3.to_standard_notation()
        assert standard == "5pr"
    
    def test_is_valid(self):
        """牌の妥当性判定テスト"""
        # 有効な牌
        valid_tiles = ["1m", "9p", "5s", "1z", "7z", "0m", "0p", "0s"]
        for notation in valid_tiles:
            tile = TenhouTile(notation)
            assert tile.is_valid(), f"牌 '{notation}' が無効と判定されました"
        
        # 無効な牌
        invalid_tiles = ["0z", "10m", "abc", "", "1x"]
        for notation in invalid_tiles:
            tile = TenhouTile(notation)
            assert not tile.is_valid(), f"牌 '{notation}' が有効と判定されました"


class TestTenhouActions:
    """天鳳アクションのテストクラス"""
    
    def test_draw_action(self):
        """ツモアクションのテスト"""
        tile = TenhouTile("3m")
        action = TenhouDrawAction(player=1, tile=tile)
        
        assert action.action_type == TenhouActionType.DRAW
        assert action.player == 1
        assert action.tile == tile
        
        # 天鳳形式変換
        tenhou_format = action.to_tenhou_format()
        assert tenhou_format == ["T1", "3m"]
    
    def test_discard_action(self):
        """打牌アクションのテスト"""
        tile = TenhouTile("7p")
        
        # 通常打牌
        action = TenhouDiscardAction(player=2, tile=tile)
        assert action.action_type == TenhouActionType.DISCARD
        assert action.is_riichi is False
        assert action.to_tenhou_format() == ["D2", "7p"]
        
        # リーチ打牌
        riichi_action = TenhouDiscardAction(player=2, tile=tile, is_riichi=True)
        assert riichi_action.to_tenhou_format() == ["D2", "7p", "r"]
        
        # ツモ切り
        tsumogiri_action = TenhouDiscardAction(player=2, tile=tile, is_tsumogiri=True)
        assert tsumogiri_action.to_tenhou_format() == ["D2", "7p", "t"]
    
    def test_call_action(self):
        """鳴きアクションのテスト"""
        tiles = [TenhouTile("2s"), TenhouTile("2s"), TenhouTile("2s")]
        action = TenhouCallAction(
            player=3,
            call_type=TenhouCallType.PON,
            tiles=tiles,
            from_player=0
        )
        
        assert action.action_type == TenhouActionType.CALL
        assert action.call_type == TenhouCallType.PON
        assert len(action.tiles) == 3
        assert action.from_player == 0
        
        tenhou_format = action.to_tenhou_format()
        assert tenhou_format == ["N3", "pon", ["2s", "2s", "2s"], 0]
    
    def test_riichi_action(self):
        """リーチアクションのテスト"""
        action = TenhouRiichiAction(player=0, step=1)
        
        assert action.action_type == TenhouActionType.RIICHI
        assert action.step == 1
        assert action.to_tenhou_format() == ["REACH0", 1]
    
    def test_agari_action(self):
        """和了アクションのテスト"""
        # ツモ和了
        tsumo_action = TenhouAgariAction(
            player=1,
            is_tsumo=True,
            han=4,
            fu=30,
            score=7700
        )
        
        assert tsumo_action.action_type == TenhouActionType.AGARI
        assert tsumo_action.is_tsumo is True
        assert tsumo_action.to_tenhou_format() == ["AGARI1", "tsumo", 4, 30, 7700]
        
        # ロン和了
        ron_action = TenhouAgariAction(
            player=2,
            is_tsumo=False,
            target_player=3,
            han=2,
            fu=40,
            score=2600
        )
        
        assert ron_action.is_tsumo is False
        assert ron_action.target_player == 3
        assert ron_action.to_tenhou_format() == ["AGARI2", "ron3", 2, 40, 2600]


class TestTenhouPlayerState:
    """プレイヤー状態のテストクラス"""
    
    def test_player_creation(self):
        """プレイヤー作成のテスト"""
        player = TenhouPlayerState(player_id=0, name="テストプレイヤー")
        
        assert player.player_id == 0
        assert player.name == "テストプレイヤー"
        assert player.score == 25000  # デフォルト値
        assert len(player.hand) == 0
        assert len(player.discards) == 0
        assert len(player.calls) == 0
        assert player.is_riichi is False
        assert player.riichi_turn is None
    
    def test_add_remove_tile(self):
        """手牌の追加・削除テスト"""
        player = TenhouPlayerState(0, "テスト")
        tile1 = TenhouTile("1m")
        tile2 = TenhouTile("2m")
        
        # 牌の追加
        player.add_tile(tile1)
        player.add_tile(tile2)
        assert player.get_hand_count() == 2
        
        # 牌の削除
        removed = player.remove_tile("1m")
        assert removed is not None
        assert removed.notation == "1m"
        assert player.get_hand_count() == 1
        
        # 存在しない牌の削除
        not_found = player.remove_tile("9z")
        assert not_found is None
        assert player.get_hand_count() == 1
    
    def test_add_discard(self):
        """捨て牌追加のテスト"""
        player = TenhouPlayerState(0, "テスト")
        tile = TenhouTile("5p")
        
        player.add_discard(tile)
        assert len(player.discards) == 1
        assert player.discards[0] == tile
    
    def test_declare_riichi(self):
        """リーチ宣言のテスト"""
        player = TenhouPlayerState(0, "テスト")
        
        player.declare_riichi(turn=10)
        assert player.is_riichi is True
        assert player.riichi_turn == 10


class TestTenhouGameRule:
    """ゲームルールのテストクラス"""
    
    def test_rule_creation_default(self):
        """デフォルトルール作成のテスト"""
        rule = TenhouGameRule()
        
        assert rule.game_type == TenhouGameType.TONPUU
        assert rule.red_dora is True
        assert rule.kuitan is True
        assert rule.display_name == "四麻東風戦"
    
    def test_rule_creation_custom(self):
        """カスタムルール作成のテスト"""
        rule = TenhouGameRule(
            game_type=TenhouGameType.HANCHAN,
            red_dora=False,
            kuitan=False,
            display_name="四麻半荘戦"
        )
        
        assert rule.game_type == TenhouGameType.HANCHAN
        assert rule.red_dora is False
        assert rule.kuitan is False
        assert rule.display_name == "四麻半荘戦"
    
    def test_to_tenhou_format(self):
        """天鳳形式変換のテスト"""
        rule = TenhouGameRule(
            game_type=TenhouGameType.HANCHAN,
            red_dora=True,
            kuitan=False,
            display_name="テストルール"
        )
        
        tenhou_format = rule.to_tenhou_format()
        expected = {
            "disp": "テストルール",
            "aka": 1,
            "kuitan": 0,
            "tonnan": 1
        }
        assert tenhou_format == expected


class TestTenhouGameResult:
    """ゲーム結果のテストクラス"""
    
    def test_result_creation_default(self):
        """デフォルト結果作成のテスト"""
        result = TenhouGameResult()
        
        assert result.rankings == [1, 2, 3, 4]
        assert result.final_scores == [25000, 25000, 25000, 25000]
        assert result.uma == [15, 5, -5, -15]
    
    def test_result_creation_custom(self):
        """カスタム結果作成のテスト"""
        result = TenhouGameResult(
            rankings=[2, 1, 4, 3],
            final_scores=[35000, 30000, 20000, 15000],
            uma=[10, 5, -5, -10]
        )
        
        assert result.rankings == [2, 1, 4, 3]
        assert result.final_scores == [35000, 30000, 20000, 15000]
        assert result.uma == [10, 5, -5, -10]
    
    def test_to_tenhou_format(self):
        """天鳳形式変換のテスト"""
        result = TenhouGameResult(
            rankings=[1, 3, 2, 4],
            final_scores=[32000, 28000, 25000, 15000],
            uma=[20, 10, -10, -20]
        )
        
        tenhou_format = result.to_tenhou_format()
        expected = {
            "順位": [1, 3, 2, 4],
            "得点": [32000, 28000, 25000, 15000],
            "ウマ": [20, 10, -10, -20]
        }
        assert tenhou_format == expected


class TestTenhouGameData:
    """天鳳ゲームデータのテストクラス"""
    
    def setup_method(self):
        """テストメソッドの前処理"""
        self.players = [
            TenhouPlayerState(0, "プレイヤー1"),
            TenhouPlayerState(1, "プレイヤー2"),
            TenhouPlayerState(2, "プレイヤー3"),
            TenhouPlayerState(3, "プレイヤー4")
        ]
        self.rule = TenhouGameRule()
        self.result = TenhouGameResult()
    
    def test_game_data_creation(self):
        """ゲームデータ作成のテスト"""
        game_data = TenhouGameData(
            title="テストゲーム",
            players=self.players,
            rule=self.rule,
            result=self.result
        )
        
        assert game_data.title == "テストゲーム"
        assert len(game_data.players) == 4
        assert game_data.rule == self.rule
        assert game_data.result == self.result
        assert len(game_data.actions) == 0
        assert game_data.game_id.startswith("game_")
    
    def test_game_data_auto_complete_players(self):
        """プレイヤー自動補完のテスト"""
        # 2人のプレイヤーのみ指定
        incomplete_players = [
            TenhouPlayerState(0, "プレイヤー1"),
            TenhouPlayerState(1, "プレイヤー2")
        ]
        
        game_data = TenhouGameData(
            title="テスト",
            players=incomplete_players,
            rule=self.rule
        )
        
        # 4人に自動補完されることを確認
        assert len(game_data.players) == 4
        assert game_data.players[2].name == "プレイヤー3"
        assert game_data.players[3].name == "プレイヤー4"
    
    def test_add_action(self):
        """アクション追加のテスト"""
        game_data = TenhouGameData(
            title="テスト",
            players=self.players,
            rule=self.rule
        )
        
        action = TenhouDrawAction(player=0, tile=TenhouTile("1m"))
        game_data.add_action(action)
        
        assert len(game_data.actions) == 1
        assert game_data.actions[0] == action
        assert game_data.get_current_turn() == 1
    
    def test_get_player(self):
        """プレイヤー取得のテスト"""
        game_data = TenhouGameData(
            title="テスト",
            players=self.players,
            rule=self.rule
        )
        
        # 有効なプレイヤーID
        player = game_data.get_player(1)
        assert player is not None
        assert player.player_id == 1
        assert player.name == "プレイヤー2"
        
        # 無効なプレイヤーID
        invalid_player = game_data.get_player(5)
        assert invalid_player is None
    
    def test_validate_structure(self):
        """データ構造検証のテスト"""
        game_data = TenhouGameData(
            title="テスト",
            players=self.players,
            rule=self.rule
        )
        
        # 有効なデータ
        is_valid, errors = game_data.validate_structure()
        assert is_valid is True
        assert len(errors) == 0
        
        # 無効なアクションを追加
        invalid_action = TenhouDrawAction(player=5, tile=TenhouTile("1m"))  # 無効なプレイヤーID
        game_data.add_action(invalid_action)
        
        is_valid, errors = game_data.validate_structure()
        assert is_valid is False
        assert len(errors) > 0
    
    def test_to_tenhou_format(self):
        """天鳳形式変換のテスト"""
        game_data = TenhouGameData(
            title="テストゲーム",
            players=self.players,
            rule=self.rule,
            result=self.result
        )
        
        # アクションを追加
        action = TenhouDrawAction(player=0, tile=TenhouTile("1m"))
        game_data.add_action(action)
        
        tenhou_format = game_data.to_tenhou_format()
        
        assert tenhou_format["title"] == "テストゲーム"
        assert len(tenhou_format["name"]) == 4
        assert "rule" in tenhou_format
        assert len(tenhou_format["log"]) == 1
        assert len(tenhou_format["sc"]) == 4
        assert "owari" in tenhou_format
    
    def test_get_statistics(self):
        """統計情報取得のテスト"""
        game_data = TenhouGameData(
            title="テスト",
            players=self.players,
            rule=self.rule
        )
        
        # 複数のアクションを追加
        game_data.add_action(TenhouDrawAction(player=0, tile=TenhouTile("1m")))
        game_data.add_action(TenhouDiscardAction(player=0, tile=TenhouTile("9z")))
        game_data.add_action(TenhouDrawAction(player=1, tile=TenhouTile("2p")))
        
        stats = game_data.get_statistics()
        
        assert stats["total_actions"] == 3
        assert stats["action_breakdown"]["T"] == 2  # ツモ2回
        assert stats["action_breakdown"]["D"] == 1  # 打牌1回
        assert stats["game_duration"] == 3
        assert stats["players"] == 4


class TestTenhouGameDataBuilder:
    """天鳳ゲームデータビルダーのテストクラス"""
    
    def test_builder_basic(self):
        """基本的なビルダー使用のテスト"""
        builder = TenhouGameDataBuilder()
        
        game_data = (builder
                    .set_title("ビルダーテスト")
                    .add_player("Alice", 30000)
                    .add_player("Bob", 25000)
                    .add_player("Charlie", 25000)
                    .add_player("David", 20000)
                    .build())
        
        assert game_data.title == "ビルダーテスト"
        assert len(game_data.players) == 4
        assert game_data.players[0].name == "Alice"
        assert game_data.players[0].score == 30000
    
    def test_builder_with_actions(self):
        """アクション付きビルダーのテスト"""
        builder = TenhouGameDataBuilder()
        action = TenhouDrawAction(player=0, tile=TenhouTile("1m"))
        
        game_data = (builder
                    .set_title("アクションテスト")
                    .add_player("Player1")
                    .add_action(action)
                    .build())
        
        assert len(game_data.actions) == 1
        assert game_data.actions[0] == action
    
    def test_builder_reset(self):
        """ビルダーリセットのテスト"""
        builder = TenhouGameDataBuilder()
        
        # 最初のビルド
        game_data1 = (builder
                     .set_title("ゲーム1")
                     .add_player("Player1")
                     .build())
        
        # リセット後のビルド
        game_data2 = (builder
                     .reset()
                     .set_title("ゲーム2")
                     .add_player("Player2")
                     .build())
        
        assert game_data1.title == "ゲーム1"
        assert game_data2.title == "ゲーム2"
        assert game_data1.players[0].name == "Player1"
        assert game_data2.players[0].name == "Player2"
    
    def test_builder_with_rule_and_result(self):
        """ルールと結果付きビルダーのテスト"""
        rule = TenhouGameRule(game_type=TenhouGameType.HANCHAN)
        result = TenhouGameResult(rankings=[2, 1, 4, 3])
        
        builder = TenhouGameDataBuilder()
        game_data = (builder
                    .set_title("完全テスト")
                    .set_rule(rule)
                    .set_result(result)
                    .build())
        
        assert game_data.rule == rule
        assert game_data.result == result
        assert game_data.rule.game_type == TenhouGameType.HANCHAN
        assert game_data.result.rankings == [2, 1, 4, 3]


class TestEnumTypes:
    """列挙型のテストクラス"""
    
    def test_action_type_enum(self):
        """アクション種別列挙型のテスト"""
        assert TenhouActionType.DRAW.value == "T"
        assert TenhouActionType.DISCARD.value == "D"
        assert TenhouActionType.CALL.value == "N"
        assert TenhouActionType.RIICHI.value == "REACH"
        assert TenhouActionType.AGARI.value == "AGARI"
    
    def test_call_type_enum(self):
        """鳴き種別列挙型のテスト"""
        assert TenhouCallType.CHI.value == "chi"
        assert TenhouCallType.PON.value == "pon"
        assert TenhouCallType.KAN.value == "kan"
        assert TenhouCallType.ANKAN.value == "ankan"
    
    def test_game_type_enum(self):
        """ゲーム種別列挙型のテスト"""
        assert TenhouGameType.TONPUU.value == 0
        assert TenhouGameType.HANCHAN.value == 1