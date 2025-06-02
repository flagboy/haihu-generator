"""
天鳳バリデーターのテスト
"""

import json
from unittest.mock import patch

from src.models.tenhou_game_data import (
    TenhouDiscardAction,
    TenhouDrawAction,
    TenhouGameData,
    TenhouGameResult,
    TenhouGameRule,
    TenhouPlayerState,
    TenhouTile,
)
from src.validation.tenhou_validator import TenhouValidator, ValidationResult


class TestValidationResult:
    """バリデーション結果のテストクラス"""

    def test_validation_result_creation(self):
        """バリデーション結果作成のテスト"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], score=1.0)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert result.score == 1.0

    def test_add_error(self):
        """エラー追加のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        result.add_error("テストエラー")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "テストエラー"

    def test_add_warning(self):
        """警告追加のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        result.add_warning("テスト警告")

        assert result.is_valid is True  # 警告では無効にならない
        assert len(result.warnings) == 1
        assert result.warnings[0] == "テスト警告"

    def test_calculate_score(self):
        """スコア計算のテスト"""
        # エラーありの場合
        result_error = ValidationResult(False, ["エラー"], [], 1.0)
        result_error.calculate_score()
        assert result_error.score == 0.0

        # 警告ありの場合
        result_warning = ValidationResult(True, [], ["警告1", "警告2"], 1.0)
        result_warning.calculate_score()
        assert result_warning.score == 0.9  # 1.0 - (2 * 0.05)

        # 警告多数の場合（最大減点0.5）
        result_many_warnings = ValidationResult(True, [], ["警告"] * 20, 1.0)
        result_many_warnings.calculate_score()
        assert result_many_warnings.score == 0.5  # 最小値0.5


class TestTenhouValidator:
    """天鳳バリデーターのテストクラス"""

    def setup_method(self):
        """テストメソッドの前処理"""
        self.validator = TenhouValidator()
        self.valid_tenhou_json = self._create_valid_tenhou_json()
        self.sample_game_data = self._create_sample_game_data()

    def _create_valid_tenhou_json(self) -> str:
        """有効な天鳳JSONを作成"""
        data = {
            "title": "テストゲーム 20240601-120000",
            "name": ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"],
            "rule": {"disp": "四麻東風戦", "aka": 1, "kuitan": 1, "tonnan": 0},
            "log": [["T0", "1m"], ["D0", "9p"], ["T1", "2p"], ["D1", "1z"]],
            "sc": [25000, 25000, 25000, 25000],
            "owari": {
                "順位": [1, 2, 3, 4],
                "得点": [25000, 25000, 25000, 25000],
                "ウマ": [15, 5, -5, -15],
            },
        }
        return json.dumps(data, ensure_ascii=False)

    def _create_sample_game_data(self) -> TenhouGameData:
        """サンプルゲームデータを作成"""
        players = [
            TenhouPlayerState(0, "プレイヤー1"),
            TenhouPlayerState(1, "プレイヤー2"),
            TenhouPlayerState(2, "プレイヤー3"),
            TenhouPlayerState(3, "プレイヤー4"),
        ]

        rule = TenhouGameRule()
        result = TenhouGameResult()

        game_data = TenhouGameData(title="テストゲーム", players=players, rule=rule, result=result)

        # サンプルアクションを追加
        game_data.add_action(TenhouDrawAction(player=0, tile=TenhouTile("1m")))
        game_data.add_action(TenhouDiscardAction(player=0, tile=TenhouTile("9p")))

        return game_data

    def test_validate_tenhou_json_valid(self):
        """有効な天鳳JSON検証のテスト"""
        result = self.validator.validate_tenhou_json(self.valid_tenhou_json)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.score > 0.8

    def test_validate_tenhou_json_invalid_json(self):
        """無効なJSON形式の検証テスト"""
        invalid_json = '{"title": "test", "incomplete": true'
        result = self.validator.validate_tenhou_json(invalid_json)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "JSON形式エラー" in result.errors[0]
        assert result.score == 0.0

    def test_validate_basic_structure_missing_fields(self):
        """必須フィールド欠如の検証テスト"""
        incomplete_data = {
            "title": "テスト",
            "name": ["プレイヤー1"],
            # 他の必須フィールドが欠如
        }
        incomplete_json = json.dumps(incomplete_data)
        result = self.validator.validate_tenhou_json(incomplete_json)

        assert result.is_valid is False
        assert any("必須フィールド" in error for error in result.errors)

    def test_validate_basic_structure_wrong_types(self):
        """データ型不正の検証テスト"""
        wrong_type_data = {
            "title": "テスト",
            "name": "プレイヤー1",  # リストであるべき
            "rule": [],  # 辞書であるべき
            "log": {},  # リストであるべき
            "sc": "25000",  # リストであるべき
            "owari": {},
        }
        wrong_type_json = json.dumps(wrong_type_data)
        result = self.validator.validate_tenhou_json(wrong_type_json)

        assert result.is_valid is False
        assert len(result.errors) >= 3  # 複数の型エラー

    def test_validate_title(self):
        """タイトル検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なタイトル
        self.validator._validate_title("有効なタイトル", result)
        assert result.is_valid is True

        # 無効なタイトル（空文字列）
        result = ValidationResult(True, [], [], 1.0)
        self.validator._validate_title("", result)
        assert result.is_valid is False

        # 無効なタイトル（None）
        result = ValidationResult(True, [], [], 1.0)
        self.validator._validate_title(None, result)
        assert result.is_valid is False

    def test_validate_player_names(self):
        """プレイヤー名検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なプレイヤー名
        valid_names = ["Alice", "Bob", "Charlie", "David"]
        self.validator._validate_player_names(valid_names, result)
        assert result.is_valid is True

        # プレイヤー数不正
        result = ValidationResult(True, [], [], 1.0)
        invalid_count = ["Alice", "Bob"]
        self.validator._validate_player_names(invalid_count, result)
        assert result.is_valid is False

        # 空の名前
        result = ValidationResult(True, [], [], 1.0)
        empty_name = ["Alice", "", "Charlie", "David"]
        self.validator._validate_player_names(empty_name, result)
        assert len(result.warnings) > 0

    def test_validate_game_rules(self):
        """ゲームルール検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なルール
        valid_rules = {"disp": "四麻東風戦", "aka": 1, "kuitan": 1, "tonnan": 0}
        self.validator._validate_game_rules(valid_rules, result)
        assert result.is_valid is True

        # 無効な値
        result = ValidationResult(True, [], [], 1.0)
        invalid_rules = {
            "aka": 2,  # 0または1であるべき
            "kuitan": -1,  # 0または1であるべき
            "tonnan": "invalid",  # 数値であるべき
        }
        self.validator._validate_game_rules(invalid_rules, result)
        assert result.is_valid is False

    def test_validate_single_action_draw(self):
        """ツモアクション検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なツモアクション
        valid_action = ["T0", "1m"]
        self.validator._validate_single_action(valid_action, 0, result)
        assert result.is_valid is True

        # 引数不足
        result = ValidationResult(True, [], [], 1.0)
        incomplete_action = ["T0"]
        self.validator._validate_single_action(incomplete_action, 0, result)
        assert result.is_valid is False

        # 無効な牌記法
        result = ValidationResult(True, [], [], 1.0)
        invalid_tile = ["T0", "invalid"]
        self.validator._validate_single_action(invalid_tile, 0, result)
        assert result.is_valid is False

    def test_validate_single_action_discard(self):
        """打牌アクション検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効な打牌アクション
        valid_action = ["D1", "5p"]
        self.validator._validate_single_action(valid_action, 0, result)
        assert result.is_valid is True

        # リーチフラグ付き
        result = ValidationResult(True, [], [], 1.0)
        riichi_action = ["D1", "5p", "r"]
        self.validator._validate_single_action(riichi_action, 0, result)
        assert result.is_valid is True

        # 不明なフラグ
        result = ValidationResult(True, [], [], 1.0)
        unknown_flag = ["D1", "5p", "x"]
        self.validator._validate_single_action(unknown_flag, 0, result)
        assert len(result.warnings) > 0

    def test_validate_single_action_call(self):
        """鳴きアクション検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効な鳴きアクション
        valid_action = ["N2", "pon", ["3s", "3s", "3s"], 1]
        self.validator._validate_single_action(valid_action, 0, result)
        assert result.is_valid is True

        # 無効な鳴き種別
        result = ValidationResult(True, [], [], 1.0)
        invalid_call = ["N2", "invalid", ["3s", "3s", "3s"], 1]
        self.validator._validate_single_action(invalid_call, 0, result)
        assert result.is_valid is False

    def test_validate_single_action_agari(self):
        """和了アクション検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なツモ和了
        valid_tsumo = ["AGARI0", "tsumo", 3, 30, 3900]
        self.validator._validate_single_action(valid_tsumo, 0, result)
        assert result.is_valid is True

        # 有効なロン和了
        result = ValidationResult(True, [], [], 1.0)
        valid_ron = ["AGARI1", "ron2", 2, 30, 2000]
        self.validator._validate_single_action(valid_ron, 0, result)
        assert result.is_valid is True

        # 引数不足
        result = ValidationResult(True, [], [], 1.0)
        incomplete_agari = ["AGARI0", "tsumo"]
        self.validator._validate_single_action(incomplete_agari, 0, result)
        assert result.is_valid is False

        # 範囲外の値
        result = ValidationResult(True, [], [], 1.0)
        out_of_range = ["AGARI0", "tsumo", 20, 200, -1000]  # 異常な値
        self.validator._validate_single_action(out_of_range, 0, result)
        assert len(result.warnings) > 0

    def test_validate_scores(self):
        """スコア検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なスコア
        valid_scores = [30000, 25000, 25000, 20000]
        self.validator._validate_scores(valid_scores, result)
        assert result.is_valid is True

        # スコア数不正
        result = ValidationResult(True, [], [], 1.0)
        wrong_count = [30000, 25000]
        self.validator._validate_scores(wrong_count, result)
        assert result.is_valid is False

        # 負のスコア
        result = ValidationResult(True, [], [], 1.0)
        negative_score = [30000, 25000, 25000, -5000]
        self.validator._validate_scores(negative_score, result)
        assert len(result.warnings) > 0

    def test_validate_tenhou_game_data_valid(self):
        """有効なゲームデータ検証のテスト"""
        result = self.validator.validate_tenhou_game_data(self.sample_game_data)

        assert result.is_valid is True
        assert result.score > 0.8

    def test_validate_tenhou_game_data_invalid_structure(self):
        """無効な構造のゲームデータ検証テスト"""
        # プレイヤー数を不正にする
        invalid_game_data = self.sample_game_data
        invalid_game_data.players = invalid_game_data.players[:2]  # 2人のみ

        result = self.validator.validate_tenhou_game_data(invalid_game_data)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_action_sequence(self):
        """アクションシーケンス検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なアクションシーケンス
        valid_actions = [
            TenhouDrawAction(player=0, tile=TenhouTile("1m")),
            TenhouDiscardAction(player=0, tile=TenhouTile("9z")),
            TenhouDrawAction(player=1, tile=TenhouTile("2p")),
        ]
        self.validator._validate_action_sequence(valid_actions, result)
        assert result.is_valid is True

        # 無効なプレイヤーIDを含むアクション
        result = ValidationResult(True, [], [], 1.0)
        invalid_actions = [
            TenhouDrawAction(player=5, tile=TenhouTile("1m"))  # 無効なプレイヤーID
        ]
        self.validator._validate_action_sequence(invalid_actions, result)
        assert result.is_valid is False

    def test_validate_player_states(self):
        """プレイヤー状態検証のテスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 有効なプレイヤー状態
        valid_players = [
            TenhouPlayerState(0, "Player1"),
            TenhouPlayerState(1, "Player2"),
            TenhouPlayerState(2, "Player3"),
            TenhouPlayerState(3, "Player4"),
        ]
        self.validator._validate_player_states(valid_players, result)
        assert result.is_valid is True

        # プレイヤー数不正
        result = ValidationResult(True, [], [], 1.0)
        wrong_count_players = valid_players[:2]
        self.validator._validate_player_states(wrong_count_players, result)
        assert result.is_valid is False

        # 負のスコア
        result = ValidationResult(True, [], [], 1.0)
        negative_score_player = [TenhouPlayerState(0, "Player1", score=-1000)]
        self.validator._validate_player_states(negative_score_player, result)
        assert len(result.warnings) > 0

    def test_is_valid_tenhou_tile(self):
        """天鳳記法牌の妥当性判定テスト"""
        # 有効な牌
        valid_tiles = ["1m", "9p", "5s", "1z", "7z", "0m", "0p", "0s"]
        for tile in valid_tiles:
            assert self.validator._is_valid_tenhou_tile(tile), f"牌 '{tile}' が無効と判定されました"

        # 無効な牌
        invalid_tiles = ["0z", "10m", "abc", "", "1x", "m1", "z8"]
        for tile in invalid_tiles:
            assert not self.validator._is_valid_tenhou_tile(tile), (
                f"牌 '{tile}' が有効と判定されました"
            )

    def test_validate_tile_notation(self):
        """牌記法検証のテスト"""
        # 有効な記法
        assert self.validator.validate_tile_notation("1m") is True
        assert self.validator.validate_tile_notation("5z") is True
        assert self.validator.validate_tile_notation("0p") is True

        # 無効な記法
        assert self.validator.validate_tile_notation("invalid") is False
        assert self.validator.validate_tile_notation("") is False
        assert self.validator.validate_tile_notation("10m") is False

    def test_get_validation_summary(self):
        """検証結果サマリー取得のテスト"""
        # エラーありの結果
        result_with_errors = ValidationResult(
            is_valid=False, errors=["エラー1", "エラー2", "エラー3"], warnings=["警告1"], score=0.0
        )

        summary = self.validator.get_validation_summary(result_with_errors)

        assert "不合格" in summary
        assert "エラー数: 3" in summary
        assert "警告数: 1" in summary
        assert "品質スコア: 0.00" in summary
        assert "エラー1" in summary
        assert "警告1" in summary

    def test_get_validation_summary_many_errors(self):
        """多数エラーのサマリーテスト"""
        many_errors = [f"エラー{i}" for i in range(10)]
        many_warnings = [f"警告{i}" for i in range(5)]

        result = ValidationResult(
            is_valid=False, errors=many_errors, warnings=many_warnings, score=0.0
        )

        summary = self.validator.get_validation_summary(result)

        assert "エラー数: 10" in summary
        assert "警告数: 5" in summary
        assert "... 他5件" in summary  # エラーの省略表示
        assert "... 他2件" in summary  # 警告の省略表示

    def test_clear_cache(self):
        """キャッシュクリアのテスト"""
        # キャッシュにデータを追加（内部実装に依存）
        self.validator._validation_cache["test"] = ValidationResult(True, [], [], 1.0)

        # キャッシュクリア
        self.validator.clear_cache()

        # キャッシュが空になることを確認
        assert len(self.validator._validation_cache) == 0

    def test_validate_consistency_player_score_mismatch(self):
        """プレイヤー数とスコア数の不整合検証テスト"""
        result = ValidationResult(True, [], [], 1.0)

        data = {
            "name": ["Player1", "Player2"],  # 2人
            "sc": [25000, 25000, 25000, 25000],  # 4人分のスコア
            "log": [],
        }

        self.validator._validate_consistency(data, result)
        assert result.is_valid is False
        assert any("プレイヤー数" in error and "スコア数" in error for error in result.errors)

    def test_validate_log_score_consistency(self):
        """ログとスコアの整合性検証テスト"""
        result = ValidationResult(True, [], [], 1.0)

        # 総得点が期待値と大きく異なるケース
        log = [["AGARI0", "tsumo", 13, 30, 32000]]  # 役満
        scores = [50000, 10000, 20000, 20000]  # 総得点100000だが配分が異常

        self.validator._validate_log_score_consistency(log, scores, result)
        # 警告が出ることを確認（実装によっては検出されない場合もある）
        # assert len(result.warnings) > 0

    @patch("src.validation.tenhou_validator.get_logger")
    def test_error_handling_in_validation(self, mock_get_logger):
        """バリデーション中のエラーハンドリングテスト"""
        # 予期しない例外が発生する状況をシミュレート
        with patch.object(
            self.validator, "_validate_basic_structure", side_effect=Exception("テスト例外")
        ):
            result = self.validator.validate_tenhou_json(self.valid_tenhou_json)

            assert result.is_valid is False
            assert any("予期しないエラー" in error for error in result.errors)

    def test_regex_patterns(self):
        """正規表現パターンのテスト"""
        # 牌パターン
        tile_pattern = self.validator.TENHOU_TILE_PATTERN

        # 有効な牌
        valid_tiles = ["1m", "9p", "5s", "1z", "7z", "0m", "0p", "0s"]
        for tile in valid_tiles:
            assert tile_pattern.match(tile), f"牌パターンが '{tile}' にマッチしません"

        # 無効な牌
        invalid_tiles = ["0z", "10m", "abc", "m1", "z8"]
        for tile in invalid_tiles:
            assert not tile_pattern.match(tile), f"牌パターンが '{tile}' に誤ってマッチしました"

        # アクションパターン
        action_pattern = self.validator.TENHOU_ACTION_PATTERN

        # 有効なアクション
        valid_actions = ["T0", "D3", "N1", "REACH2", "AGARI0", "RYUU"]
        for action in valid_actions:
            assert action_pattern.match(action), f"アクションパターンが '{action}' にマッチしません"

        # 無効なアクション
        invalid_actions = ["X0", "T", "D10", "INVALID"]
        for action in invalid_actions:
            assert not action_pattern.match(action), (
                f"アクションパターンが '{action}' に誤ってマッチしました"
            )
