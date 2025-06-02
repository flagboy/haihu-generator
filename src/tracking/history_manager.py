"""
履歴管理クラス
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..game.player import PlayerPosition
from ..game.turn import Action, ActionType
from ..utils.tile_definitions import TileDefinitions


class RecordFormat(Enum):
    """牌譜形式"""

    TENHOU = "tenhou"  # 天鳳形式


@dataclass
class GameRecord:
    """ゲーム記録"""

    game_id: str
    start_time: datetime
    end_time: datetime | None = None
    players: dict[PlayerPosition, str] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    final_scores: dict[PlayerPosition, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初期化後の処理"""
        if not self.players:
            self.players = {pos: f"Player{pos.value + 1}" for pos in PlayerPosition}
        if not self.rounds:
            self.rounds = []
        if not self.final_scores:
            self.final_scores = dict.fromkeys(PlayerPosition, 25000)
        if not self.metadata:
            self.metadata = {}


@dataclass
class RoundRecord:
    """局記録"""

    round_number: int
    round_name: str
    dealer: PlayerPosition
    actions: list[Action] = field(default_factory=list)
    result: dict[str, Any] | None = None
    scores: dict[PlayerPosition, int] = field(default_factory=dict)
    duration: float = 0.0

    def __post_init__(self):
        """初期化後の処理"""
        if not self.actions:
            self.actions = []
        if not self.scores:
            self.scores = dict.fromkeys(PlayerPosition, 25000)


class HistoryManager:
    """ゲーム履歴管理クラス"""

    def __init__(self):
        """履歴管理クラスを初期化"""
        self.tile_definitions = TileDefinitions()

        # 現在のゲーム記録
        self.current_game: GameRecord | None = None
        self.current_round: RoundRecord | None = None

        # 履歴データ
        self.completed_games: list[GameRecord] = []
        self.action_history: list[Action] = []

        # 統計情報
        self.total_actions = 0
        self.total_rounds = 0
        self.total_games = 0

    def start_new_game(self, game_id: str, players: dict[PlayerPosition, str]) -> GameRecord:
        """
        新しいゲームを開始

        Args:
            game_id: ゲームID
            players: プレイヤー情報

        Returns:
            GameRecord: 新しいゲーム記録
        """
        # 前のゲームを完了
        if self.current_game:
            self.complete_current_game()

        # 新しいゲーム記録を作成
        self.current_game = GameRecord(
            game_id=game_id,
            start_time=datetime.now(),
            players=players.copy(),
            metadata={"video_source": "unknown", "detection_method": "ai_analysis"},
        )

        self.total_games += 1
        return self.current_game

    def start_new_round(
        self, round_number: int, round_name: str, dealer: PlayerPosition
    ) -> RoundRecord:
        """
        新しい局を開始

        Args:
            round_number: 局番号
            round_name: 局名（例：東1局）
            dealer: 親のプレイヤー

        Returns:
            RoundRecord: 新しい局記録
        """
        if not self.current_game:
            raise ValueError("No active game to start round")

        # 前の局を完了
        if self.current_round:
            self.complete_current_round()

        # 新しい局記録を作成
        self.current_round = RoundRecord(
            round_number=round_number, round_name=round_name, dealer=dealer
        )

        self.total_rounds += 1
        return self.current_round

    def add_action(self, action: Action) -> bool:
        """
        行動を履歴に追加

        Args:
            action: 追加する行動

        Returns:
            bool: 追加に成功したかどうか
        """
        if not self.current_round:
            return False

        # 行動を現在の局に追加
        self.current_round.actions.append(action)

        # 全体の行動履歴にも追加
        self.action_history.append(action)
        self.total_actions += 1

        return True

    def complete_current_round(
        self, result: dict[str, Any] | None = None, scores: dict[PlayerPosition, int] | None = None
    ):
        """
        現在の局を完了

        Args:
            result: 局の結果
            scores: 各プレイヤーの点数
        """
        if not self.current_round or not self.current_game:
            return

        # 局の結果を設定
        if result:
            self.current_round.result = result

        if scores:
            self.current_round.scores = scores

        # 局の所要時間を計算
        if self.current_round.actions:
            start_time = min(action.timestamp for action in self.current_round.actions)
            end_time = max(action.timestamp for action in self.current_round.actions)
            self.current_round.duration = end_time - start_time

        # ゲーム記録に局を追加
        round_data = self._convert_round_to_dict(self.current_round)
        self.current_game.rounds.append(round_data)

        self.current_round = None

    def complete_current_game(self, final_scores: dict[PlayerPosition, int] | None = None):
        """
        現在のゲームを完了

        Args:
            final_scores: 最終点数
        """
        if not self.current_game:
            return

        # 現在の局があれば完了
        if self.current_round:
            self.complete_current_round()

        # 最終点数を設定
        if final_scores:
            self.current_game.final_scores = final_scores

        # ゲーム終了時刻を設定
        self.current_game.end_time = datetime.now()

        # 完了したゲームリストに追加
        self.completed_games.append(self.current_game)
        self.current_game = None

    def _convert_round_to_dict(self, round_record: RoundRecord) -> dict[str, Any]:
        """局記録を辞書形式に変換"""
        return {
            "round_number": round_record.round_number,
            "round_name": round_record.round_name,
            "dealer": round_record.dealer.value,
            "actions": [self._convert_action_to_dict(action) for action in round_record.actions],
            "result": round_record.result,
            "scores": {pos.value: score for pos, score in round_record.scores.items()},
            "duration": round_record.duration,
        }

    def _convert_action_to_dict(self, action: Action) -> dict[str, Any]:
        """行動を辞書形式に変換"""
        return {
            "action_type": action.action_type.value,
            "player": action.player.value,
            "tile": action.tile,
            "tiles": action.tiles,
            "from_player": action.from_player.value if action.from_player else None,
            "timestamp": action.timestamp,
            "frame_number": action.frame_number,
            "confidence": action.confidence,
            "detected_by": action.detected_by,
            "metadata": action.metadata,
        }

    def export_to_tenhou_format(self, game_record: GameRecord | None = None) -> str:
        """
        天鳳形式で牌譜をエクスポート

        Args:
            game_record: エクスポートするゲーム記録（Noneの場合は現在のゲーム）

        Returns:
            str: 天鳳形式の牌譜
        """
        if game_record is None:
            game_record = self.current_game

        if not game_record:
            return ""

        # 天鳳JSON形式専用のため、XML出力は削除
        self.logger.warning("XML出力は天鳳JSON特化により削除されました")
        return ""

    def export_to_tenhou_json_format(self, game_record: GameRecord | None = None) -> dict[str, Any]:
        """
        天鳳JSON形式で牌譜をエクスポート（最適化版）

        Args:
            game_record: エクスポートするゲーム記録（Noneの場合は現在のゲーム）

        Returns:
            Dict[str, Any]: 天鳳JSON形式の牌譜データ
        """
        if game_record is None:
            game_record = self.current_game

        if not game_record:
            return {}

        # 天鳳JSON形式のデータ構造
        tenhou_data = {
            "format": "tenhou_json",
            "version": "1.0",
            "game_info": {
                "game_id": game_record.game_id,
                "rule": "東南戦",  # デフォルト
                "players": self._format_players_for_tenhou(game_record.players),
                "start_time": game_record.start_time.isoformat(),
                "end_time": game_record.end_time.isoformat() if game_record.end_time else None,
                "final_scores": self._format_scores_for_tenhou(game_record.final_scores),
            },
            "rounds": [],
        }

        # 各局のデータを天鳳形式に変換
        for round_data in game_record.rounds:
            tenhou_round = self._convert_round_to_tenhou_json(round_data)
            tenhou_data["rounds"].append(tenhou_round)

        # メタデータを追加
        tenhou_data["metadata"] = {
            "total_rounds": len(game_record.rounds),
            "total_actions": sum(len(round_data["actions"]) for round_data in game_record.rounds),
            "export_timestamp": datetime.now().isoformat(),
            "source_metadata": game_record.metadata,
        }

        return tenhou_data

    def _format_players_for_tenhou(
        self, players: dict[PlayerPosition, str]
    ) -> list[dict[str, Any]]:
        """プレイヤー情報を天鳳JSON形式にフォーマット"""
        formatted_players = []
        for pos in PlayerPosition:
            player_name = players.get(pos, f"Player{pos.value + 1}")
            formatted_players.append(
                {
                    "position": pos.value,
                    "name": player_name,
                    "seat": pos.name.lower(),  # east, south, west, north
                }
            )
        return formatted_players

    def _format_scores_for_tenhou(self, scores: dict[PlayerPosition, int]) -> list[int]:
        """点数を天鳳JSON形式にフォーマット"""
        return [scores.get(pos, 25000) for pos in PlayerPosition]

    def _convert_round_to_tenhou_json(self, round_data: dict[str, Any]) -> dict[str, Any]:
        """局データを天鳳JSON形式に変換"""
        tenhou_round = {
            "round_info": {
                "round_number": round_data["round_number"],
                "round_name": round_data["round_name"],
                "dealer": round_data["dealer"],
                "honba": 0,  # 本場数（デフォルト）
                "riichi_sticks": 0,  # リーチ棒数（デフォルト）
            },
            "initial_state": {
                "dora_indicators": [],  # ドラ表示牌
                "scores": [25000, 25000, 25000, 25000],  # 各プレイヤーの開始点数
            },
            "actions": [],
            "result": round_data.get("result"),
            "final_scores": list(round_data["scores"].values())
            if round_data.get("scores")
            else [25000, 25000, 25000, 25000],
        }

        # 行動データを天鳳JSON形式に変換
        for action_data in round_data["actions"]:
            tenhou_action = self._convert_action_to_tenhou_json(action_data)
            if tenhou_action:
                tenhou_round["actions"].append(tenhou_action)

        return tenhou_round

    def _convert_action_to_tenhou_json(self, action_data: dict[str, Any]) -> dict[str, Any] | None:
        """行動を天鳳JSON形式に変換"""
        action_type = action_data["action_type"]
        player = action_data["player"]
        tile = action_data.get("tile", "")
        tiles = action_data.get("tiles", [])

        # 基本的な行動情報
        tenhou_action = {
            "type": action_type,
            "player": player,
            "timestamp": action_data.get("timestamp", 0),
            "frame_number": action_data.get("frame_number", 0),
            "confidence": action_data.get("confidence", 0.0),
        }

        # 行動タイプ別の詳細情報
        if action_type == "draw":
            tenhou_action["tile"] = self._convert_tile_to_tenhou_format(tile)
        elif action_type == "discard":
            tenhou_action["tile"] = self._convert_tile_to_tenhou_format(tile)
            tenhou_action["riichi"] = action_data.get("metadata", {}).get("riichi", False)
        elif action_type == "call":
            tenhou_action["call_type"] = action_data.get("metadata", {}).get("call_type", "unknown")
            tenhou_action["tiles"] = [self._convert_tile_to_tenhou_format(t) for t in tiles]
            tenhou_action["from_player"] = action_data.get("from_player")
        elif action_type == "riichi":
            tenhou_action["riichi_tile"] = self._convert_tile_to_tenhou_format(tile)
        elif action_type == "tsumo" or action_type == "ron":
            tenhou_action["winning_tile"] = self._convert_tile_to_tenhou_format(tile)
            tenhou_action["hand"] = [self._convert_tile_to_tenhou_format(t) for t in tiles]

        return tenhou_action

    def _convert_tile_to_tenhou_format(self, tile: str) -> str:
        """牌を天鳳形式に変換"""
        if not tile:
            return ""

        # 既に天鳳形式の場合はそのまま返す
        if tile.endswith(("m", "p", "s", "z")):
            return tile

        # 他の形式から天鳳形式に変換（簡略化）
        tile_id = self.tile_definitions.get_tile_id(tile)
        if tile_id >= 0:
            return self.tile_definitions.get_tenhou_notation(tile_id)

        return tile

    def get_action_statistics(self) -> dict[str, Any]:
        """行動統計を取得"""
        action_counts = {}
        player_action_counts = dict.fromkeys(PlayerPosition, 0)

        for action in self.action_history:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            player_action_counts[action.player] += 1

        return {
            "total_actions": self.total_actions,
            "action_types": action_counts,
            "player_actions": {pos.name: count for pos, count in player_action_counts.items()},
            "average_actions_per_round": self.total_actions / max(1, self.total_rounds),
        }

    def get_game_summary(self) -> dict[str, Any]:
        """ゲーム概要を取得"""
        return {
            "total_games": self.total_games,
            "total_rounds": self.total_rounds,
            "total_actions": self.total_actions,
            "completed_games": len(self.completed_games),
            "current_game_active": self.current_game is not None,
            "current_round_active": self.current_round is not None,
        }

    def search_actions(
        self,
        action_type: ActionType | None = None,
        player: PlayerPosition | None = None,
        tile: str | None = None,
    ) -> list[Action]:
        """
        条件に基づいて行動を検索

        Args:
            action_type: 行動タイプ
            player: プレイヤー
            tile: 牌

        Returns:
            List[Action]: 条件に一致する行動のリスト
        """
        results = []

        for action in self.action_history:
            if action_type and action.action_type != action_type:
                continue
            if player and action.player != player:
                continue
            if tile and action.tile != tile:
                continue

            results.append(action)

        return results

    def clear_history(self):
        """履歴をクリア"""
        self.current_game = None
        self.current_round = None
        self.completed_games = []
        self.action_history = []

        self.total_actions = 0
        self.total_rounds = 0
        self.total_games = 0

    def __str__(self) -> str:
        """文字列表現"""
        return (
            f"HistoryManager(Games: {self.total_games}, "
            f"Rounds: {self.total_rounds}, Actions: {self.total_actions})"
        )

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"HistoryManager(games={self.total_games}, rounds={self.total_rounds}, "
            f"actions={self.total_actions}, active_game={self.current_game is not None})"
        )
