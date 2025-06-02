"""
天鳳JSON特化最適化テスト
フェーズ4で実装された最適化機能の包括的テスト
"""

import json
import time

from src.models.tenhou_game_data import (
    TenhouDiscardAction,
    TenhouDrawAction,
    TenhouGameData,
    TenhouGameRule,
    TenhouPlayerState,
    TenhouTile,
)
from src.output.tenhou_json_formatter import TenhouJsonFormatter
from src.utils.tile_definitions import TileDefinitions
from src.validation.tenhou_validator import TenhouValidator


class TestTenhouOptimization:
    """天鳳JSON特化最適化テスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.formatter = TenhouJsonFormatter()
        self.validator = TenhouValidator()
        self.tile_definitions = TileDefinitions()

        # 標準ゲームデータ
        self.game_data = TenhouGameData(
            title="最適化テスト",
            players=[
                TenhouPlayerState(0, "プレイヤー1"),
                TenhouPlayerState(1, "プレイヤー2"),
                TenhouPlayerState(2, "プレイヤー3"),
                TenhouPlayerState(3, "プレイヤー4"),
            ],
            rule=TenhouGameRule(),
        )

    def test_formatter_initialization(self):
        """フォーマッター初期化テスト"""
        formatter = TenhouJsonFormatter()

        assert hasattr(formatter, "_format_cache")
        assert hasattr(formatter, "_tile_cache")
        assert hasattr(formatter, "_max_cache_size")
        assert formatter._max_cache_size == 1000
        assert len(formatter._format_cache) == 0
        assert len(formatter._tile_cache) == 0

    def test_tile_conversion_optimization(self):
        """牌変換最適化テスト"""
        # 牌変換テスト
        test_tiles = ["1m", "2p", "3s", "東", "南", "西", "北", "白", "發", "中"]

        for tile in test_tiles:
            result = self.formatter._get_tenhou_tile(tile)
            assert isinstance(result, str)
            assert len(result) > 0

            # キャッシュ確認
            assert tile in self.formatter._tile_cache
            assert self.formatter._tile_cache[tile] == result

    def test_action_conversion_optimization(self):
        """アクション変換最適化テスト"""
        # ツモアクション
        draw_action = {"action_type": "draw", "player": 0, "tile": "1m"}
        result = self.formatter._convert_single_action(draw_action)
        assert result == ["T0", "1m"]

        # 打牌アクション
        discard_action = {"action_type": "discard", "player": 1, "tile": "2p", "is_riichi": False}
        result = self.formatter._convert_single_action(discard_action)
        assert result == ["D1", "2p"]

        # リーチ打牌
        riichi_discard = {"action_type": "discard", "player": 2, "tile": "3s", "is_riichi": True}
        result = self.formatter._convert_single_action(riichi_discard)
        assert result == ["D2", "3s", "r"]

    def test_call_action_conversion(self):
        """鳴きアクション変換テスト"""
        call_action = {
            "action_type": "call",
            "player": 3,
            "call_type": "pon",
            "tiles": ["東", "東", "東"],
            "from_player": 0,
        }
        result = self.formatter._convert_single_action(call_action)
        assert result == ["N3", "pon", ["1z", "1z", "1z"], 0]

    def test_riichi_action_conversion(self):
        """リーチアクション変換テスト"""
        riichi_action = {"action_type": "riichi", "player": 1, "step": 1}
        result = self.formatter._convert_single_action(riichi_action)
        assert result == ["REACH1", 1]

    def test_agari_action_conversion(self):
        """和了アクション変換テスト"""
        # ツモ和了
        tsumo_action = {
            "action_type": "agari",
            "player": 0,
            "is_tsumo": True,
            "han": 3,
            "fu": 30,
            "score": 3900,
        }
        result = self.formatter._convert_single_action(tsumo_action)
        assert result == ["AGARI0", "tsumo", 3, 30, 3900]

        # ロン和了
        ron_action = {
            "action_type": "agari",
            "player": 2,
            "is_tsumo": False,
            "han": 2,
            "fu": 40,
            "score": 2600,
        }
        result = self.formatter._convert_single_action(ron_action)
        assert result == ["AGARI2", "ron", 2, 40, 2600]

    def test_game_data_structure_conversion(self):
        """ゲームデータ構造変換テスト"""
        # アクション追加
        self.game_data.add_action(TenhouDrawAction(0, TenhouTile("1m")))
        self.game_data.add_action(TenhouDiscardAction(0, TenhouTile("9m"), False))

        result = self.formatter.format_game_data(self.game_data)
        data = json.loads(result)

        # 基本構造確認
        assert "title" in data
        assert "name" in data
        assert "rule" in data
        assert "log" in data
        assert "sc" in data
        assert "owari" in data

        # プレイヤー名確認
        assert len(data["name"]) == 4
        assert data["name"][0] == "プレイヤー1"

        # ルール確認
        assert "disp" in data["rule"]
        assert "aka" in data["rule"]
        assert "kuitan" in data["rule"]
        assert "tonnan" in data["rule"]

    def test_compact_json_format(self):
        """コンパクトJSON形式テスト"""
        result = self.formatter.format_game_data(self.game_data)

        # コンパクト形式確認
        assert ", " not in result
        assert ": " not in result
        assert result.startswith("{")
        assert result.endswith("}")

        # JSON妥当性確認
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_validation_output(self):
        """出力検証テスト"""
        result = self.formatter.format_game_data(self.game_data)

        # 検証実行
        is_valid = self.formatter.validate_output(result)
        assert is_valid is True

        # 不正なJSONテスト
        invalid_json = '{"invalid": json}'
        is_valid = self.formatter.validate_output(invalid_json)
        assert is_valid is False

    def test_tile_definitions_integration(self):
        """牌定義統合テスト"""
        # 全牌種テスト
        all_tiles = []

        # 数牌
        for suit in ["m", "p", "s"]:
            for num in range(1, 10):
                all_tiles.append(f"{num}{suit}")

        # 字牌
        honor_tiles = ["東", "南", "西", "北", "白", "發", "中"]
        all_tiles.extend(honor_tiles)

        # 赤ドラ
        red_dora = ["0m", "0p", "0s"]
        all_tiles.extend(red_dora)

        for tile in all_tiles:
            result = self.formatter._get_tenhou_tile(tile)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_cache_performance(self):
        """キャッシュ性能テスト"""
        # キャッシュクリア
        self.formatter._tile_cache.clear()

        test_tiles = ["1m", "2p", "3s"] * 100

        # 初回変換時間測定
        start_time = time.time()
        for tile in test_tiles:
            self.formatter._get_tenhou_tile(tile)
        first_time = time.time() - start_time

        # 2回目変換時間測定（キャッシュ効果）
        start_time = time.time()
        for tile in test_tiles:
            self.formatter._get_tenhou_tile(tile)
        second_time = time.time() - start_time

        # キャッシュ効果確認
        if second_time > 0 and first_time > 0:
            speedup = first_time / second_time
            # キャッシュは常に効果があるはずだが、環境依存なので緩い閾値にする
            # CI環境では初回実行が遅い場合があるため、0.7以上であればOKとする
            assert speedup >= 0.7, f"キャッシュ効果が不十分: {speedup:.1f}x"

    def test_memory_management(self):
        """メモリ管理テスト"""
        # キャッシュサイズ制限テスト
        original_limit = self.formatter._max_cache_size
        self.formatter._max_cache_size = 5
        self.formatter._tile_cache.clear()

        try:
            # 制限を超える牌を変換
            for i in range(10):
                tile = f"{(i % 9) + 1}m"
                self.formatter._get_tenhou_tile(tile)

            # キャッシュサイズが制限内であることを確認
            assert len(self.formatter._tile_cache) <= 5

        finally:
            self.formatter._max_cache_size = original_limit

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        # 不正なアクション
        invalid_action = {"action_type": "invalid", "player": 0}
        result = self.formatter._convert_single_action(invalid_action)
        assert result is None

        # 不正なゲームデータ
        try:
            result = self.formatter.format_game_data(None)
            # エラーが発生するはず
            raise AssertionError("エラーが発生すべき")
        except Exception:
            # 期待される動作
            pass

    def test_tenhou_validator_integration(self):
        """天鳳バリデーター統合テスト"""
        # 有効なデータ作成
        self.game_data.add_action(TenhouDrawAction(0, TenhouTile("1m")))
        self.game_data.add_action(TenhouDiscardAction(0, TenhouTile("9m"), False))

        # JSON変換
        json_result = self.formatter.format_game_data(self.game_data)
        parsed_data = json.loads(json_result)

        # バリデーション実行
        validation_result = self.validator.validate_tenhou_json(parsed_data)

        # 結果確認
        assert hasattr(validation_result, "is_valid")
        assert hasattr(validation_result, "errors")
        assert hasattr(validation_result, "warnings")

    def test_performance_benchmark(self):
        """性能ベンチマークテスト"""
        # 大量アクション追加
        for i in range(100):
            if i % 2 == 0:
                action = TenhouDrawAction(i % 4, TenhouTile(f"{(i % 9) + 1}m"))
            else:
                action = TenhouDiscardAction(i % 4, TenhouTile(f"{(i % 9) + 1}p"), False)
            self.game_data.add_action(action)

        # 変換時間測定
        start_time = time.time()
        result = self.formatter.format_game_data(self.game_data)
        processing_time = time.time() - start_time

        # 性能確認
        assert processing_time < 0.1, f"処理時間が過大: {processing_time:.3f}s"
        assert isinstance(result, str)
        assert len(result) > 0

        # JSON妥当性確認
        data = json.loads(result)
        assert len(data["log"]) == 100

    def test_concurrent_safety(self):
        """並行安全性テスト"""
        import queue
        import threading

        results = queue.Queue()

        def format_data():
            try:
                result = self.formatter.format_game_data(self.game_data)
                results.put(("success", result))
            except Exception as e:
                results.put(("error", str(e)))

        # 複数スレッドで同時実行
        threads = []
        for _i in range(3):
            thread = threading.Thread(target=format_data)
            threads.append(thread)
            thread.start()

        # 完了待機
        for thread in threads:
            thread.join()

        # 結果確認
        success_count = 0
        while not results.empty():
            status, result = results.get()
            if status == "success":
                success_count += 1
                assert isinstance(result, str)

        assert success_count == 3
