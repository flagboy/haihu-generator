"""
天鳳JSONフォーマッターのテスト
"""

import json

import pytest

from src.models.tenhou_game_data import (
    TenhouAgariAction,
    TenhouCallAction,
    TenhouCallType,
    TenhouDiscardAction,
    TenhouDrawAction,
    TenhouGameData,
    TenhouGameResult,
    TenhouGameRule,
    TenhouGameType,
    TenhouPlayerState,
    TenhouRiichiAction,
    TenhouTile,
)
from src.output.tenhou_json_formatter import TenhouJsonFormatter


class TestTenhouJsonFormatter:
    """天鳳JSONフォーマッターのテストクラス"""

    def setup_method(self):
        """テストメソッドの前処理"""
        self.formatter = TenhouJsonFormatter()
        self.sample_game_data = self._create_sample_game_data()

    def _create_sample_game_data(self) -> TenhouGameData:
        """サンプルゲームデータを作成"""
        players = [
            TenhouPlayerState(0, "プレイヤー1", score=25000),
            TenhouPlayerState(1, "プレイヤー2", score=25000),
            TenhouPlayerState(2, "プレイヤー3", score=25000),
            TenhouPlayerState(3, "プレイヤー4", score=25000),
        ]

        rule = TenhouGameRule(
            game_type=TenhouGameType.TONPUU, red_dora=True, kuitan=True, display_name="四麻東風戦"
        )

        result = TenhouGameResult(
            rankings=[1, 2, 3, 4], final_scores=[30000, 25000, 25000, 20000], uma=[15, 5, -5, -15]
        )

        game_data = TenhouGameData(
            title="テストゲーム 20240601-120000", players=players, rule=rule, result=result
        )

        # サンプルアクションを追加
        game_data.add_action(TenhouDrawAction(player=0, tile=TenhouTile("1m")))
        game_data.add_action(TenhouDiscardAction(player=0, tile=TenhouTile("9z"), is_riichi=False))

        return game_data

    def test_format_game_data_basic(self):
        """基本的なゲームデータフォーマットのテスト"""
        result = self.formatter.format_game_data(self.sample_game_data)

        # JSON形式であることを確認
        assert isinstance(result, str)
        data = json.loads(result)

        # 必須フィールドの存在確認
        required_fields = ["title", "name", "rule", "log", "sc", "owari"]
        for field in required_fields:
            assert field in data, f"必須フィールド '{field}' が見つかりません"

    def test_format_game_data_structure(self):
        """ゲームデータ構造のテスト"""
        result = self.formatter.format_game_data(self.sample_game_data)
        data = json.loads(result)

        # タイトル
        assert data["title"] == "テストゲーム 20240601-120000"

        # プレイヤー名
        assert len(data["name"]) == 4
        assert data["name"][0] == "プレイヤー1"

        # ルール
        assert data["rule"]["disp"] == "四麻東風戦"
        assert data["rule"]["aka"] == 1
        assert data["rule"]["kuitan"] == 1
        assert data["rule"]["tonnan"] == 0

        # スコア
        assert len(data["sc"]) == 4
        assert all(isinstance(score, int) for score in data["sc"])

        # ログ
        assert isinstance(data["log"], list)
        assert len(data["log"]) == 2  # サンプルアクション数

    def test_convert_draw_action(self):
        """ツモアクション変換のテスト"""
        action = TenhouDrawAction(player=1, tile=TenhouTile("5m"))

        # to_tenhou_formatメソッドを直接テスト
        result = action.to_tenhou_format()
        expected = ["T1", "5m"]
        assert result == expected

    def test_convert_discard_action(self):
        """打牌アクション変換のテスト"""
        # 通常の打牌
        action = TenhouDiscardAction(player=2, tile=TenhouTile("7p"), is_riichi=False)

        result = action.to_tenhou_format()
        expected = ["D2", "7p"]
        assert result == expected

        # リーチ打牌
        action_riichi = TenhouDiscardAction(player=2, tile=TenhouTile("7p"), is_riichi=True)

        result_riichi = action_riichi.to_tenhou_format()
        expected_riichi = ["D2", "7p", "r"]
        assert result_riichi == expected_riichi

    def test_convert_call_action(self):
        """鳴きアクション変換のテスト"""
        # ポン
        action = TenhouCallAction(
            player=3,
            call_type=TenhouCallType.PON,
            tiles=[TenhouTile("3s"), TenhouTile("3s"), TenhouTile("3s")],
            from_player=0,
        )

        result = action.to_tenhou_format()
        expected = ["N3", "pon", ["3s", "3s", "3s"], 0]
        assert result == expected

    def test_convert_riichi_action(self):
        """リーチアクション変換のテスト"""
        action = TenhouRiichiAction(player=1, step=1)

        result = action.to_tenhou_format()
        expected = ["REACH1", 1]
        assert result == expected

    def test_convert_agari_action(self):
        """和了アクション変換のテスト"""
        # ツモ和了
        tsumo_action = TenhouAgariAction(player=0, is_tsumo=True, han=3, fu=30, score=3900)

        result = tsumo_action.to_tenhou_format()
        expected = ["AGARI0", "tsumo", 3, 30, 3900]
        assert result == expected

        # ロン和了
        ron_action = TenhouAgariAction(
            player=1, is_tsumo=False, target_player=2, han=2, fu=30, score=2000
        )

        result = ron_action.to_tenhou_format()
        expected = ["AGARI1", "ron2", 2, 30, 2000]
        assert result == expected

    def test_get_tenhou_tile_caching(self):
        """牌変換キャッシュのテスト"""
        # 初回変換
        result1 = self.formatter._get_tenhou_tile("1m")
        assert result1 == "1m"

        # キャッシュからの取得
        result2 = self.formatter._get_tenhou_tile("1m")
        assert result2 == "1m"

        # キャッシュサイズの確認
        cache_stats = self.formatter.get_cache_stats()
        assert cache_stats["tile_cache_size"] >= 1

    def test_format_compact(self):
        """コンパクト形式フォーマットのテスト"""
        result = self.formatter.format_compact(self.sample_game_data)
        data = json.loads(result)

        # コンパクト版には最小限のフィールドのみ
        assert "log" in data
        assert "sc" in data
        assert "title" not in data
        assert "name" not in data

    def test_validate_output_valid(self):
        """有効なJSON出力の検証テスト"""
        result = self.formatter.format_game_data(self.sample_game_data)
        is_valid = self.formatter.validate_output(result)
        assert is_valid is True

    def test_validate_output_invalid(self):
        """無効なJSON出力の検証テスト"""
        invalid_json = '{"title": "test", "incomplete": true'
        is_valid = self.formatter.validate_output(invalid_json)
        assert is_valid is False

    def test_validate_output_missing_fields(self):
        """必須フィールド欠如の検証テスト"""
        incomplete_data = {
            "title": "test",
            "name": ["player1"],
            # 他の必須フィールドが欠如
        }
        incomplete_json = json.dumps(incomplete_data)
        is_valid = self.formatter.validate_output(incomplete_json)
        assert is_valid is False

    def test_clear_cache(self):
        """キャッシュクリアのテスト"""
        # キャッシュにデータを追加
        self.formatter._get_tenhou_tile("1m")
        self.formatter._get_tenhou_tile("2p")

        # キャッシュクリア前の確認
        cache_stats = self.formatter.get_cache_stats()
        assert cache_stats["tile_cache_size"] > 0

        # キャッシュクリア
        self.formatter.clear_cache()

        # キャッシュクリア後の確認
        cache_stats = self.formatter.get_cache_stats()
        assert cache_stats["tile_cache_size"] == 0
        assert cache_stats["format_cache_size"] == 0

    def test_get_game_title_default(self):
        """デフォルトゲームタイトル生成のテスト"""
        data = {"game_type": "四麻半荘戦"}
        title = self.formatter._get_game_title(data)
        assert "四麻半荘戦" in title
        assert len(title) > len("四麻半荘戦")  # タイムスタンプが追加されている

    def test_get_player_names_various_formats(self):
        """様々な形式のプレイヤー名取得テスト"""
        # 辞書形式のプレイヤー
        data1 = {"players": [{"name": "Alice"}, {"name": "Bob"}]}
        names1 = self.formatter._get_player_names(data1)
        assert names1 == ["Alice", "Bob"]

        # 文字列リスト形式
        data2 = {"players": ["Charlie", "David"]}
        names2 = self.formatter._get_player_names(data2)
        assert names2 == ["Charlie", "David"]

        # プレイヤーデータなし
        data3 = {}
        names3 = self.formatter._get_player_names(data3)
        assert len(names3) == 4  # デフォルトの4人
        assert all("プレイヤー" in name for name in names3)

    def test_get_final_scores_conversion(self):
        """最終スコア変換のテスト"""
        data = {"final_scores": [30000, 25000, 25000, 20000]}
        scores = self.formatter._get_final_scores(data)
        assert scores == [30000, 25000, 25000, 20000]
        assert all(isinstance(score, int) for score in scores)

        # デフォルト値のテスト
        data_empty = {}
        scores_default = self.formatter._get_final_scores(data_empty)
        assert scores_default == [25000, 25000, 25000, 25000]

    def test_performance_large_data(self):
        """大量データでの性能テスト"""
        # 大量のアクションを持つゲームデータを作成
        large_game_data = self.sample_game_data

        # 1000個のアクションを追加
        for i in range(1000):
            action = TenhouDrawAction(player=i % 4, tile=TenhouTile(f"{(i % 9) + 1}m"))
            large_game_data.add_action(action)

        # フォーマット実行時間を測定
        import time

        start_time = time.time()
        result = self.formatter.format_game_data(large_game_data)
        end_time = time.time()

        # 結果の妥当性確認
        assert isinstance(result, str)
        data = json.loads(result)
        assert len(data["log"]) > 1000

        # 性能要件確認（1秒以内）
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"処理時間が長すぎます: {processing_time:.3f}秒"

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 不正なデータでのフォーマット
        invalid_data = None

        with pytest.raises((TypeError, AttributeError)):  # TypeErrorまたはAttributeErrorをキャッチ
            self.formatter.format_game_data(invalid_data)

    def test_json_output_format(self):
        """JSON出力形式の詳細テスト"""
        result = self.formatter.format_game_data(self.sample_game_data)

        # JSON形式の確認
        assert result.startswith("{")
        assert result.endswith("}")

        # コンパクト形式（空白なし）の確認
        assert ", " not in result  # separatorsで空白を削除
        assert ": " not in result

        # 日本語文字の確認（ensure_ascii=False）
        data = json.loads(result)
        assert any("プレイヤー" in name for name in data["name"])
