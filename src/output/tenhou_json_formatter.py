"""
天鳳JSON専用フォーマッター
天鳳形式に特化した高速JSON変換処理を提供
"""

import json
import time
from datetime import datetime
from typing import Any

from ..utils.logger import get_logger
from ..utils.tile_definitions import TileDefinitions


class TenhouJsonFormatter:
    """天鳳JSON形式専用フォーマッター"""

    def __init__(self):
        """フォーマッターの初期化"""
        self.tile_definitions = TileDefinitions()
        self.logger = get_logger(__name__)
        # メモリ最適化：キャッシュサイズ制限
        self._format_cache: dict[str, str] = {}
        self._tile_cache: dict[str, str] = {}
        self._max_cache_size = 1000  # キャッシュサイズ制限

    def format_game_data(self, game_data: Any) -> str:
        """ゲームデータを天鳳JSON形式に変換

        Args:
            game_data: ゲームデータオブジェクト

        Returns:
            str: 天鳳JSON形式の文字列
        """
        start_time = time.time()

        try:
            # データ構造の最適化された変換
            tenhou_data = self._convert_to_tenhou_structure(game_data)

            # 高速JSON変換（separatorsで空白を削除）
            json_str = json.dumps(
                tenhou_data, ensure_ascii=False, separators=(",", ":"), sort_keys=False
            )

            processing_time = time.time() - start_time
            self.logger.info(f"天鳳JSON変換完了: {processing_time:.3f}秒")

            return json_str

        except Exception as e:
            self.logger.error(f"天鳳JSON変換エラー: {e}")
            raise

    def _convert_to_tenhou_structure(self, game_data: Any) -> dict[str, Any]:
        """ゲームデータを天鳳構造に変換"""
        if hasattr(game_data, "__dict__"):
            data_dict = game_data.__dict__
        elif hasattr(game_data, "_asdict"):
            data_dict = game_data._asdict()
        else:
            data_dict = game_data

        return {
            "title": self._get_game_title(data_dict),
            "name": self._get_player_names(data_dict),
            "rule": self._get_game_rules(data_dict),
            "log": self._convert_game_log(data_dict.get("actions", [])),
            "sc": self._get_final_scores(data_dict),
            "owari": self._get_game_result(data_dict),
        }

    def _get_game_title(self, data: dict[str, Any]) -> str:
        """ゲームタイトルを取得"""
        # 既存のtitleがあればそれを使用
        if "title" in data:
            return data["title"]

        # titleがない場合は生成
        timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d-%H%M%S"))
        game_type = data.get("game_type", "四麻東風戦")
        return f"{game_type} {timestamp}"

    def _get_player_names(self, data: dict[str, Any]) -> list[str]:
        """プレイヤー名リストを取得"""
        players = data.get("players", [])
        if isinstance(players, list) and len(players) > 0:
            names = []
            for i, player in enumerate(players):
                if hasattr(player, "name"):
                    names.append(player.name)
                elif isinstance(player, dict):
                    names.append(player.get("name", f"プレイヤー{i + 1}"))
                else:
                    names.append(str(player))
            return names
        return ["プレイヤー1", "プレイヤー2", "プレイヤー3", "プレイヤー4"]

    def _get_game_rules(self, data: dict[str, Any]) -> dict[str, Any]:
        """ゲームルールを取得"""
        rules = data.get("rules", {})
        return {
            "disp": rules.get("display_name", "四麻東風戦"),
            "aka": 1 if rules.get("red_dora", True) else 0,
            "kuitan": 1 if rules.get("kuitan", True) else 0,
            "tonnan": 0 if rules.get("game_length", "tonpuu") == "tonpuu" else 1,
        }

    def _convert_game_log(self, actions: list[Any]) -> list[list[Any]]:
        """ゲームログを天鳳形式に変換"""
        converted_log = []

        for action in actions:
            # アクションオブジェクトがto_tenhou_formatメソッドを持つ場合は直接使用
            if hasattr(action, "to_tenhou_format"):
                converted_action = action.to_tenhou_format()
                if converted_action:
                    converted_log.append(converted_action)
            else:
                # アクションオブジェクトから辞書に変換
                if hasattr(action, "__dict__"):
                    action_dict = action.__dict__.copy()
                    # action_typeが存在しない場合、クラス名から推定
                    if "action_type" not in action_dict:
                        class_name = action.__class__.__name__
                        if "Draw" in class_name:
                            action_dict["action_type"] = "draw"
                        elif "Discard" in class_name:
                            action_dict["action_type"] = "discard"
                        elif "Call" in class_name:
                            action_dict["action_type"] = "call"
                        elif "Riichi" in class_name:
                            action_dict["action_type"] = "riichi"
                        elif "Agari" in class_name:
                            action_dict["action_type"] = "agari"

                    # TenhouTileオブジェクトを文字列に変換
                    if "tile" in action_dict and hasattr(action_dict["tile"], "notation"):
                        action_dict["tile"] = action_dict["tile"].notation
                    if "tiles" in action_dict:
                        tiles = action_dict["tiles"]
                        if isinstance(tiles, list):
                            action_dict["tiles"] = [
                                tile.notation if hasattr(tile, "notation") else str(tile)
                                for tile in tiles
                            ]
                else:
                    action_dict = action

                converted_action = self._convert_single_action(action_dict)
                if converted_action:
                    converted_log.append(converted_action)

        return converted_log

    def _convert_single_action(self, action: dict[str, Any]) -> list[Any] | None:
        """単一アクションを天鳳形式に変換（最適化版）"""
        # アクションタイプの判定を高速化
        action_type = action.get("action_type") or action.get("type", "")
        player = action.get("player", 0)

        # 高速変換マップ
        if action_type == "draw":
            tile = self._get_tenhou_tile(action.get("tile", ""))
            return [f"T{player}", tile]
        elif action_type == "discard":
            tile = self._get_tenhou_tile(action.get("tile", ""))
            if action.get("is_riichi", False):
                return [f"D{player}", tile, "r"]
            return [f"D{player}", tile]
        elif action_type == "call":
            call_type = action.get("call_type", "")
            tiles = [self._get_tenhou_tile(t) for t in action.get("tiles", [])]
            from_player = action.get("from_player", 0)
            return [f"N{player}", call_type, tiles, from_player]
        elif action_type == "riichi":
            step = action.get("step", 1)
            return [f"REACH{player}", step]
        elif action_type == "agari":
            agari_type = "tsumo" if action.get("is_tsumo", False) else "ron"
            han = action.get("han", 0)
            fu = action.get("fu", 0)
            score = action.get("score", 0)
            return [f"AGARI{player}", agari_type, han, fu, score]

        return None

    def _convert_draw_action(self, action: dict[str, Any]) -> list[Any]:
        """ツモアクションを変換"""
        player = action.get("player", 0)
        tile = action.get("tile", "")
        tenhou_tile = self._get_tenhou_tile(tile)
        return [f"T{player}", tenhou_tile]

    def _convert_discard_action(self, action: dict[str, Any]) -> list[Any]:
        """打牌アクションを変換"""
        player = action.get("player", 0)
        tile = action.get("tile", "")
        tenhou_tile = self._get_tenhou_tile(tile)
        riichi = action.get("riichi", False)

        if riichi:
            return [f"D{player}", tenhou_tile, "r"]
        else:
            return [f"D{player}", tenhou_tile]

    def _convert_call_action(self, action: dict[str, Any]) -> list[Any]:
        """鳴きアクションを変換"""
        player = action.get("player", 0)
        call_type = action.get("call_type", "")
        tiles = action.get("tiles", [])

        tenhou_tiles = [self._get_tenhou_tile(tile) for tile in tiles]

        call_mapping = {"chi": "chi", "pon": "pon", "kan": "kan"}
        call_name = call_mapping.get(call_type, call_type)
        return [f"N{player}", call_name, tenhou_tiles]

    def _convert_riichi_action(self, action: dict[str, Any]) -> list[Any]:
        """リーチアクションを変換"""
        player = action.get("player", 0)
        return [f"REACH{player}"]

    def _convert_tsumo_action(self, action: dict[str, Any]) -> list[Any]:
        """ツモ和了アクションを変換"""
        player = action.get("player", 0)
        han = action.get("han", 0)
        fu = action.get("fu", 0)
        score = action.get("score", 0)
        return [f"AGARI{player}", "tsumo", han, fu, score]

    def _convert_ron_action(self, action: dict[str, Any]) -> list[Any]:
        """ロン和了アクションを変換"""
        player = action.get("player", 0)
        target = action.get("target", 0)
        han = action.get("han", 0)
        fu = action.get("fu", 0)
        score = action.get("score", 0)
        return [f"AGARI{player}", f"ron{target}", han, fu, score]

    def _get_tenhou_tile(self, tile: str) -> str:
        """牌を天鳳記法に変換（最適化キャッシュ付き）"""
        if tile in self._tile_cache:
            return self._tile_cache[tile]

        # キャッシュサイズ制限
        if len(self._tile_cache) >= self._max_cache_size:
            # 古いエントリを削除（FIFO）
            oldest_key = next(iter(self._tile_cache))
            del self._tile_cache[oldest_key]

        tenhou_tile = self.tile_definitions.convert_to_tenhou_notation(tile)
        self._tile_cache[tile] = tenhou_tile
        return tenhou_tile

    def _get_final_scores(self, data: dict[str, Any]) -> list[int]:
        """最終スコアを取得"""
        scores = data.get("final_scores", [25000, 25000, 25000, 25000])
        return [int(score) for score in scores]

    def _get_game_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """ゲーム結果を取得"""
        result = data.get("result", {})
        if hasattr(result, "to_tenhou_format"):
            return result.to_tenhou_format()
        elif isinstance(result, dict):
            return {
                "順位": result.get("rankings", [1, 2, 3, 4]),
                "得点": result.get("final_scores", [25000, 25000, 25000, 25000]),
                "ウマ": result.get("uma", [15, 5, -5, -15]),
            }
        else:
            return {
                "順位": [1, 2, 3, 4],
                "得点": [25000, 25000, 25000, 25000],
                "ウマ": [15, 5, -5, -15],
            }

    def format_compact(self, game_data: Any) -> str:
        """コンパクト形式での天鳳JSON変換

        Args:
            game_data: ゲームデータオブジェクト

        Returns:
            str: コンパクトな天鳳JSON形式の文字列
        """
        tenhou_data = self._convert_to_tenhou_structure(game_data)

        # 最小限の情報のみを含むコンパクト版
        compact_data = {"log": tenhou_data["log"], "sc": tenhou_data["sc"]}

        return json.dumps(compact_data, ensure_ascii=False, separators=(",", ":"))

    def validate_output(self, json_str: str) -> bool:
        """出力されたJSONの妥当性を検証

        Args:
            json_str: 検証するJSON文字列

        Returns:
            bool: 妥当性の判定結果
        """
        try:
            data = json.loads(json_str)

            # 必須フィールドの存在確認
            required_fields = ["title", "name", "rule", "log", "sc", "owari"]
            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"必須フィールド '{field}' が見つかりません")
                    return False

            # ログデータの形式確認
            if not isinstance(data["log"], list):
                self.logger.warning("ログデータがリスト形式ではありません")
                return False

            # スコアデータの形式確認
            if not isinstance(data["sc"], list) or len(data["sc"]) != 4:
                self.logger.warning("スコアデータの形式が正しくありません")
                return False

            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON形式エラー: {e}")
            return False
        except Exception as e:
            self.logger.error(f"検証エラー: {e}")
            return False

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._format_cache.clear()
        self._tile_cache.clear()
        self.logger.debug("フォーマッターキャッシュをクリアしました")

    def get_cache_stats(self) -> dict[str, int]:
        """キャッシュ統計を取得"""
        return {
            "format_cache_size": len(self._format_cache),
            "tile_cache_size": len(self._tile_cache),
        }
