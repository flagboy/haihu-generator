"""
天鳳形式データモデル
天鳳記法との完全互換性を持つ型安全なデータ構造を定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..utils.tile_definitions import TileDefinitions


class TenhouActionType(Enum):
    """天鳳アクション種別"""

    DRAW = "T"  # ツモ
    DISCARD = "D"  # 打牌
    CALL = "N"  # 鳴き
    RIICHI = "REACH"  # リーチ
    AGARI = "AGARI"  # 和了
    RYUUKYOKU = "RYUU"  # 流局


class TenhouCallType(Enum):
    """天鳳鳴き種別"""

    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    ANKAN = "ankan"


class TenhouGameType(Enum):
    """天鳳ゲーム種別"""

    TONPUU = 0  # 東風戦
    HANCHAN = 1  # 半荘戦


@dataclass
class TenhouTile:
    """天鳳牌データ"""

    notation: str  # 天鳳記法（例: "1m", "5z", "0p"）
    original: str = ""  # 元の記法
    is_red_dora: bool = False

    def __post_init__(self):
        """初期化後処理"""
        if not self.original:
            self.original = self.notation
        self.is_red_dora = self.notation.startswith("0")

    def to_standard_notation(self) -> str:
        """標準記法に変換"""
        tile_def = TileDefinitions()
        return tile_def.convert_from_tenhou_notation(self.notation)

    def is_valid(self) -> bool:
        """有効な牌かどうか判定"""
        tile_def = TileDefinitions()
        return tile_def.is_tenhou_notation(self.notation)


class TenhouAction:
    """天鳳アクション基底クラス"""

    def __init__(
        self, action_type: TenhouActionType, player: int, timestamp: datetime | None = None
    ):
        self.action_type = action_type
        self.player = player
        self.timestamp = timestamp

    def to_tenhou_format(self) -> list[Any]:
        """天鳳形式に変換"""
        raise NotImplementedError("サブクラスで実装してください")


@dataclass
class TenhouDrawAction(TenhouAction):
    """ツモアクション"""

    player: int
    tile: TenhouTile
    timestamp: datetime | None = None

    def __post_init__(self):
        super().__init__(TenhouActionType.DRAW, self.player, self.timestamp)

    def to_tenhou_format(self) -> list[Any]:
        return [f"T{self.player}", self.tile.notation]


@dataclass
class TenhouDiscardAction(TenhouAction):
    """打牌アクション"""

    player: int
    tile: TenhouTile
    is_riichi: bool = False
    is_tsumogiri: bool = False
    timestamp: datetime | None = None

    def __post_init__(self):
        super().__init__(TenhouActionType.DISCARD, self.player, self.timestamp)

    def to_tenhou_format(self) -> list[Any]:
        result = [f"D{self.player}", self.tile.notation]
        if self.is_riichi:
            result.append("r")
        if self.is_tsumogiri:
            result.append("t")
        return result


@dataclass
class TenhouCallAction(TenhouAction):
    """鳴きアクション"""

    player: int
    call_type: TenhouCallType
    tiles: list[TenhouTile]
    from_player: int
    timestamp: datetime | None = None

    def __post_init__(self):
        super().__init__(TenhouActionType.CALL, self.player, self.timestamp)

    def to_tenhou_format(self) -> list[Any]:
        tile_notations = [tile.notation for tile in self.tiles]
        return [f"N{self.player}", self.call_type.value, tile_notations, self.from_player]


@dataclass
class TenhouRiichiAction(TenhouAction):
    """リーチアクション"""

    player: int
    step: int = 1  # 1: リーチ宣言, 2: リーチ成立
    timestamp: datetime | None = None

    def __post_init__(self):
        super().__init__(TenhouActionType.RIICHI, self.player, self.timestamp)

    def to_tenhou_format(self) -> list[Any]:
        return [f"REACH{self.player}", self.step]


@dataclass
class TenhouAgariAction(TenhouAction):
    """和了アクション"""

    player: int
    is_tsumo: bool
    target_player: int | None = None  # ロンの場合の放銃者
    han: int = 0
    fu: int = 0
    score: int = 0
    yaku: list[str] = field(default_factory=list)
    winning_tile: TenhouTile | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        super().__init__(TenhouActionType.AGARI, self.player, self.timestamp)

    def to_tenhou_format(self) -> list[Any]:
        if self.is_tsumo:
            return [f"AGARI{self.player}", "tsumo", self.han, self.fu, self.score]
        else:
            return [
                f"AGARI{self.player}",
                f"ron{self.target_player}",
                self.han,
                self.fu,
                self.score,
            ]


@dataclass
class TenhouPlayerState:
    """プレイヤー状態"""

    player_id: int
    name: str
    hand: list[TenhouTile] = field(default_factory=list)
    discards: list[TenhouTile] = field(default_factory=list)
    calls: list[TenhouCallAction] = field(default_factory=list)
    score: int = 25000
    is_riichi: bool = False
    riichi_turn: int | None = None

    def add_tile(self, tile: TenhouTile) -> None:
        """手牌に牌を追加"""
        self.hand.append(tile)

    def remove_tile(self, tile_notation: str) -> TenhouTile | None:
        """手牌から牌を削除"""
        for i, tile in enumerate(self.hand):
            if tile.notation == tile_notation:
                return self.hand.pop(i)
        return None

    def add_discard(self, tile: TenhouTile) -> None:
        """捨て牌に追加"""
        self.discards.append(tile)

    def declare_riichi(self, turn: int) -> None:
        """リーチ宣言"""
        self.is_riichi = True
        self.riichi_turn = turn

    def get_hand_count(self) -> int:
        """手牌枚数を取得"""
        return len(self.hand)


@dataclass
class TenhouGameRule:
    """天鳳ゲームルール"""

    game_type: TenhouGameType = TenhouGameType.TONPUU
    red_dora: bool = True
    kuitan: bool = True
    display_name: str = "四麻東風戦"

    def to_tenhou_format(self) -> dict[str, Any]:
        """天鳳形式に変換"""
        return {
            "disp": self.display_name,
            "aka": 1 if self.red_dora else 0,
            "kuitan": 1 if self.kuitan else 0,
            "tonnan": self.game_type.value,
        }


@dataclass
class TenhouGameResult:
    """ゲーム結果"""

    rankings: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    final_scores: list[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    uma: list[int] = field(default_factory=lambda: [15, 5, -5, -15])

    def to_tenhou_format(self) -> dict[str, Any]:
        """天鳳形式に変換"""
        return {"順位": self.rankings, "得点": self.final_scores, "ウマ": self.uma}


@dataclass
class TenhouGameData:
    """天鳳ゲームデータ"""

    title: str
    players: list[TenhouPlayerState]
    rule: TenhouGameRule
    actions: list[TenhouAction] = field(default_factory=list)
    result: TenhouGameResult | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    game_id: str = ""

    def __post_init__(self):
        """初期化後処理"""
        if not self.game_id:
            self.game_id = f"game_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

        # プレイヤーが4人未満の場合は補完
        while len(self.players) < 4:
            player_id = len(self.players)
            self.players.append(
                TenhouPlayerState(player_id=player_id, name=f"プレイヤー{player_id + 1}")
            )

    def add_action(self, action: TenhouAction) -> None:
        """アクションを追加"""
        self.actions.append(action)

    def get_player(self, player_id: int) -> TenhouPlayerState | None:
        """プレイヤー状態を取得"""
        if 0 <= player_id < len(self.players):
            return self.players[player_id]
        return None

    def get_current_turn(self) -> int:
        """現在のターン数を取得"""
        return len(self.actions)

    def to_tenhou_format(self) -> dict[str, Any]:
        """天鳳形式に変換"""
        return {
            "title": self.title,
            "name": [player.name for player in self.players],
            "rule": self.rule.to_tenhou_format(),
            "log": [action.to_tenhou_format() for action in self.actions],
            "sc": [player.score for player in self.players],
            "owari": self.result.to_tenhou_format() if self.result else {},
        }

    def validate_structure(self) -> tuple[bool, list[str]]:
        """データ構造の妥当性を検証"""
        errors = []

        # プレイヤー数チェック
        if len(self.players) != 4:
            errors.append(f"プレイヤー数が不正です: {len(self.players)}")

        # アクションの妥当性チェック
        for i, action in enumerate(self.actions):
            # プレイヤーIDの妥当性チェック
            if action.player < 0 or action.player >= 4:
                errors.append(f"アクション{i}のプレイヤーIDが不正です: {action.player}")

        # タイトルチェック
        if not self.title or not isinstance(self.title, str):
            errors.append("タイトルが設定されていません")

        return len(errors) == 0, errors

    def get_statistics(self) -> dict[str, Any]:
        """ゲーム統計を取得"""
        action_counts: dict[str, int] = {}
        for action in self.actions:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        return {
            "total_actions": len(self.actions),
            "action_breakdown": action_counts,
            "game_duration": self.get_current_turn(),
            "players": len(self.players),
        }


class TenhouGameDataBuilder:
    """天鳳ゲームデータビルダー"""

    def __init__(self):
        self.reset()

    def reset(self) -> "TenhouGameDataBuilder":
        """ビルダーをリセット"""
        self._title = ""
        self._players: list[TenhouPlayerState] = []
        self._rule = TenhouGameRule()
        self._actions: list[TenhouAction] = []
        self._result: TenhouGameResult | None = None
        return self

    def set_title(self, title: str) -> "TenhouGameDataBuilder":
        """タイトルを設定"""
        self._title = title
        return self

    def add_player(self, name: str, score: int = 25000) -> "TenhouGameDataBuilder":
        """プレイヤーを追加"""
        player_id = len(self._players)
        player = TenhouPlayerState(player_id=player_id, name=name, score=score)
        self._players.append(player)
        return self

    def set_rule(self, rule: TenhouGameRule) -> "TenhouGameDataBuilder":
        """ルールを設定"""
        self._rule = rule
        return self

    def add_action(self, action: TenhouAction) -> "TenhouGameDataBuilder":
        """アクションを追加"""
        self._actions.append(action)
        return self

    def set_result(self, result: TenhouGameResult) -> "TenhouGameDataBuilder":
        """結果を設定"""
        self._result = result
        return self

    def build(self) -> TenhouGameData:
        """ゲームデータを構築"""
        return TenhouGameData(
            title=self._title,
            players=self._players,
            rule=self._rule,
            actions=self._actions,
            result=self._result,
        )
