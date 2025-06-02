"""
処理結果の保存と変換を担当するクラス
"""

from pathlib import Path
from typing import Any

from ..models.tenhou_game_data import TenhouGameData
from ..output.tenhou_json_formatter import TenhouJsonFormatter
from ..utils.config import ConfigManager
from ..utils.file_io import FileIOHelper
from ..utils.logger import get_logger


class ResultProcessor:
    """処理結果の保存と変換を行うクラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config_manager = config_manager
        self.config = config_manager._config
        self.logger = get_logger(self.__class__.__name__)

        # 天鳳JSON設定
        self.tenhou_config = self.config.get("tenhou_json", {})
        self.formatter = TenhouJsonFormatter()

    def save_results(
        self, game_data: Any, output_path: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        処理結果を保存

        Args:
            game_data: ゲームデータ
            output_path: 出力パス
            metadata: メタデータ（オプション）
        """
        try:
            # 天鳳形式に変換
            if hasattr(game_data, "to_tenhou_format"):
                tenhou_data = game_data.to_tenhou_format()
            else:
                # TenhouGameDataオブジェクトに変換
                # デフォルト値で初期化
                from ..models.tenhou_game_data import (
                    TenhouGameRule,
                    TenhouGameType,
                    TenhouPlayerState,
                )

                default_players = [TenhouPlayerState(i, f"プレイヤー{i + 1}") for i in range(4)]
                default_rule = TenhouGameRule(game_type=TenhouGameType.TONPUU)
                tenhou_game = TenhouGameData(
                    title="ゲーム", players=default_players, rule=default_rule
                )
                self._populate_tenhou_data(tenhou_game, game_data)
                # format_game_dataはJSON文字列を返すので、辞書に変換
                tenhou_json_str = self.formatter.format_game_data(tenhou_game)
                import json

                tenhou_data = json.loads(tenhou_json_str)

            # 既に辞書形式の場合の処理
            if isinstance(game_data, dict) and not hasattr(game_data, "to_tenhou_format"):
                tenhou_data = game_data

            # メタデータを追加（辞書形式の場合のみ）
            if metadata and isinstance(tenhou_data, dict):
                tenhou_data["metadata"] = metadata

            # 最適化処理（辞書形式の場合のみ）
            if isinstance(tenhou_data, dict):
                tenhou_data = self._optimize_tenhou_data(tenhou_data)

            # ファイル保存
            pretty_print = self.tenhou_config.get("pretty_print", True)
            FileIOHelper.save_json(tenhou_data, output_path, pretty=pretty_print)

            self.logger.info(f"結果を保存しました: {output_path}")

        except Exception as e:
            self.logger.error(f"結果の保存に失敗しました: {e}", exc_info=True)
            raise

    def _populate_tenhou_data(self, tenhou_game: TenhouGameData, game_data: Any) -> None:
        """
        ゲームデータをTenhouGameDataに変換

        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            game_data: 元のゲームデータ
        """
        # ゲーム情報の設定
        if hasattr(game_data, "game_info"):
            info = game_data.game_info
            # titleを更新
            if "room_name" in info:
                tenhou_game.title = (
                    f"{info['room_name']} {tenhou_game.timestamp.strftime('%Y%m%d-%H%M%S')}"
                )

        # プレイヤー情報の設定
        if hasattr(game_data, "players") and game_data.players:
            from ..models.tenhou_game_data import TenhouPlayerState

            new_players = []
            for i, player in enumerate(game_data.players):
                if isinstance(player, dict):
                    new_players.append(
                        TenhouPlayerState(
                            player_id=i,
                            name=player.get("name", f"Player{i + 1}"),
                            score=player.get("initial_score", 25000),
                        )
                    )
            if new_players:
                tenhou_game.players = new_players

        # ラウンド情報の追加
        if hasattr(game_data, "rounds"):
            for round_data in game_data.rounds:
                self._add_round_data(tenhou_game, round_data)

    def _add_round_data(self, tenhou_game: TenhouGameData, round_data: dict[str, Any]) -> None:
        """
        ラウンドデータを追加

        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            round_data: ラウンドデータ
        """
        # アクションの追加
        for action in round_data.get("actions", []):
            self._add_action(tenhou_game, action)

    def _add_action(self, tenhou_game: TenhouGameData, action: dict[str, Any]) -> None:
        """
        アクションを追加

        Args:
            tenhou_game: 天鳳ゲームデータオブジェクト
            action: アクションデータ
        """
        from ..models.tenhou_game_data import (
            TenhouCallAction,
            TenhouCallType,
            TenhouDiscardAction,
            TenhouDrawAction,
            TenhouTile,
        )

        action_type = action.get("type")
        player = action.get("player", 0)

        if action_type == "draw":
            tile_str = action.get("tile", "1m")
            draw_action = TenhouDrawAction(player=player, tile=TenhouTile(tile_str))
            tenhou_game.add_action(draw_action)
        elif action_type == "discard":
            tile_str = action.get("tile", "1m")
            discard_action = TenhouDiscardAction(
                player=player,
                tile=TenhouTile(tile_str),
                is_riichi=action.get("is_riichi", False),
                is_tsumogiri=action.get("is_tsumogiri", False),
            )
            tenhou_game.add_action(discard_action)
        elif action_type in ["chi", "pon", "kan"]:
            tiles_str = action.get("tiles", [])
            tiles = [TenhouTile(t) for t in tiles_str]
            call_type_map = {
                "chi": TenhouCallType.CHI,
                "pon": TenhouCallType.PON,
                "kan": TenhouCallType.KAN,
            }
            call_action = TenhouCallAction(
                player=player,
                call_type=call_type_map.get(action_type, TenhouCallType.PON),
                tiles=tiles,
                from_player=action.get("from_player", (player + 3) % 4),
            )
            tenhou_game.add_action(call_action)

    def _optimize_tenhou_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        天鳳データを最適化

        Args:
            data: 天鳳形式のデータ

        Returns:
            最適化されたデータ
        """
        optimization = self.tenhou_config.get("optimization", {})

        if optimization.get("remove_empty_fields", True):
            data = self._remove_empty_fields(data)

        if optimization.get("compress_redundant_data", True):
            data = self._compress_redundant_data(data)

        return data

    def _remove_empty_fields(self, data: Any) -> Any:
        """
        空のフィールドを削除

        Args:
            data: 処理対象のデータ

        Returns:
            空フィールドを削除したデータ
        """
        if isinstance(data, dict):
            return {
                k: self._remove_empty_fields(v)
                for k, v in data.items()
                if v is not None and v != "" and v != [] and v != {}
            }
        elif isinstance(data, list):
            return [self._remove_empty_fields(item) for item in data]
        else:
            return data

    def _compress_redundant_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        冗長なデータを圧縮

        Args:
            data: 処理対象のデータ

        Returns:
            圧縮されたデータ
        """
        # 連続する同じアクションをまとめる
        if "rounds" in data:
            for round_data in data["rounds"]:
                if "actions" in round_data:
                    round_data["actions"] = self._compress_actions(round_data["actions"])

        return data

    def _compress_actions(self, actions: list[Any]) -> list[Any]:
        """
        アクションリストを圧縮

        Args:
            actions: アクションリスト

        Returns:
            圧縮されたアクションリスト
        """
        if not actions:
            return actions

        compressed = []
        current_group = None

        for action in actions:
            # 同じ種類の連続したアクションをグループ化
            if (
                current_group and action[0] == current_group["type"] and len(action) == 2
            ):  # シンプルなアクションのみ
                current_group["actions"].append(action)
            else:
                if current_group and len(current_group["actions"]) > 1:
                    # グループ化されたアクションを追加
                    compressed.append([current_group["type"], current_group["actions"]])
                elif current_group:
                    # 単一のアクションはそのまま追加
                    compressed.extend(current_group["actions"])

                # 新しいグループを開始
                if len(action) == 2 and isinstance(action[0], str):
                    current_group = {"type": action[0], "actions": [action]}
                else:
                    compressed.append(action)
                    current_group = None

        # 最後のグループを処理
        if current_group:
            if len(current_group["actions"]) > 1:
                compressed.append([current_group["type"], current_group["actions"]])
            else:
                compressed.extend(current_group["actions"])

        return compressed

    def export_statistics(self, statistics: dict[str, Any], output_path: str) -> None:
        """
        統計情報をエクスポート

        Args:
            statistics: 統計情報
            output_path: 出力パス
        """
        stats_path = Path(output_path).with_suffix(".stats.json")
        FileIOHelper.save_json(statistics, stats_path, pretty=True)
        self.logger.info(f"統計情報を保存しました: {stats_path}")
