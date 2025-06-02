"""
プレイヤー状態管理クラス
"""

import copy
from dataclasses import dataclass, field
from enum import Enum

from ..utils.tile_definitions import TileDefinitions


class PlayerPosition(Enum):
    """プレイヤーの座席位置"""

    EAST = 0  # 東家
    SOUTH = 1  # 南家
    WEST = 2  # 西家
    NORTH = 3  # 北家


class CallType(Enum):
    """鳴きの種類"""

    CHI = "chi"  # チー
    PON = "pon"  # ポン
    KAN = "kan"  # カン
    ANKAN = "ankan"  # 暗カン


@dataclass
class Call:
    """鳴きの情報"""

    call_type: CallType
    tiles: list[str]
    from_player: PlayerPosition | None = None
    called_tile: str | None = None
    timestamp: float | None = None


@dataclass
class PlayerState:
    """プレイヤーの状態情報"""

    position: PlayerPosition
    name: str = ""
    score: int = 25000
    hand_tiles: list[str] = field(default_factory=list)
    discarded_tiles: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    riichi: bool = False
    riichi_turn: int | None = None
    furiten: bool = False
    ippatsu: bool = False
    menzen: bool = True
    tenpai: bool = False

    def __post_init__(self):
        """初期化後の処理"""
        if not self.hand_tiles:
            self.hand_tiles = []
        if not self.discarded_tiles:
            self.discarded_tiles = []
        if not self.calls:
            self.calls = []


class Player:
    """プレイヤー管理クラス"""

    def __init__(self, position: PlayerPosition, name: str = "", initial_score: int = 25000):
        """
        プレイヤーを初期化

        Args:
            position: プレイヤーの座席位置
            name: プレイヤー名
            initial_score: 初期点数
        """
        self.tile_definitions = TileDefinitions()
        self.state = PlayerState(
            position=position, name=name or f"Player{position.value + 1}", score=initial_score
        )
        self._hand_tile_counts: dict[str, int] = {}
        self._update_hand_counts()

    def _update_hand_counts(self):
        """手牌の枚数カウントを更新"""
        self._hand_tile_counts = {}
        for tile in self.state.hand_tiles:
            self._hand_tile_counts[tile] = self._hand_tile_counts.get(tile, 0) + 1

    def add_tile_to_hand(self, tile: str) -> bool:
        """
        手牌に牌を追加

        Args:
            tile: 追加する牌

        Returns:
            bool: 追加に成功したかどうか
        """
        if not self.tile_definitions.is_valid_tile(tile):
            return False

        self.state.hand_tiles.append(tile)
        self._update_hand_counts()
        return True

    def remove_tile_from_hand(self, tile: str) -> bool:
        """
        手牌から牌を削除

        Args:
            tile: 削除する牌

        Returns:
            bool: 削除に成功したかどうか
        """
        if tile in self.state.hand_tiles:
            self.state.hand_tiles.remove(tile)
            self._update_hand_counts()
            return True
        return False

    def discard_tile(self, tile: str) -> bool:
        """
        牌を捨てる

        Args:
            tile: 捨てる牌

        Returns:
            bool: 捨牌に成功したかどうか
        """
        if self.remove_tile_from_hand(tile):
            self.state.discarded_tiles.append(tile)
            # リーチ後の捨牌でイッパツフラグをクリア
            if self.state.riichi:
                self.state.ippatsu = False
            return True
        return False

    def make_call(self, call: Call) -> bool:
        """
        鳴きを行う

        Args:
            call: 鳴きの情報

        Returns:
            bool: 鳴きに成功したかどうか
        """
        # 鳴きに必要な牌が手牌にあるかチェック
        required_tiles = call.tiles.copy()
        if call.called_tile and call.called_tile in required_tiles:
            required_tiles.remove(call.called_tile)

        temp_hand = self.state.hand_tiles.copy()
        for tile in required_tiles:
            if tile in temp_hand:
                temp_hand.remove(tile)
            else:
                return False

        # 鳴きを実行
        for tile in required_tiles:
            self.remove_tile_from_hand(tile)

        self.state.calls.append(call)
        self.state.menzen = False  # 鳴いたので門前ではなくなる
        self._update_hand_counts()
        return True

    def declare_riichi(self, turn: int) -> bool:
        """
        リーチを宣言

        Args:
            turn: リーチを宣言したターン

        Returns:
            bool: リーチ宣言に成功したかどうか
        """
        if not self.state.menzen or self.state.riichi:
            return False

        if self.state.score < 1000:
            return False  # 点数不足

        self.state.riichi = True
        self.state.riichi_turn = turn
        self.state.ippatsu = True
        self.state.score -= 1000  # リーチ棒
        return True

    def get_hand_size(self) -> int:
        """手牌の枚数を取得"""
        return len(self.state.hand_tiles)

    def get_tile_count_in_hand(self, tile: str) -> int:
        """手牌内の指定牌の枚数を取得"""
        return self._hand_tile_counts.get(tile, 0)

    def has_tile_in_hand(self, tile: str) -> bool:
        """手牌に指定牌があるかチェック"""
        return tile in self.state.hand_tiles

    def get_sorted_hand(self) -> list[str]:
        """ソートされた手牌を取得"""
        return sorted(self.state.hand_tiles, key=lambda x: self.tile_definitions.get_tile_id(x))

    def get_called_tiles(self) -> list[str]:
        """鳴いた牌をすべて取得"""
        called_tiles = []
        for call in self.state.calls:
            called_tiles.extend(call.tiles)
        return called_tiles

    def get_all_tiles(self) -> list[str]:
        """手牌と鳴いた牌をすべて取得"""
        return self.state.hand_tiles + self.get_called_tiles()

    def is_furiten(self) -> bool:
        """フリテン状態かどうかを判定"""
        return self.state.furiten

    def set_furiten(self, furiten: bool):
        """フリテン状態を設定"""
        self.state.furiten = furiten

    def is_tenpai(self) -> bool:
        """テンパイ状態かどうかを判定"""
        return self.state.tenpai

    def set_tenpai(self, tenpai: bool):
        """テンパイ状態を設定"""
        self.state.tenpai = tenpai

    def reset_for_new_round(self):
        """新しい局のためにリセット"""
        self.state.hand_tiles = []
        self.state.discarded_tiles = []
        self.state.calls = []
        self.state.riichi = False
        self.state.riichi_turn = None
        self.state.furiten = False
        self.state.ippatsu = False
        self.state.menzen = True
        self.state.tenpai = False
        self._hand_tile_counts = {}

    def get_state_copy(self) -> PlayerState:
        """プレイヤー状態のコピーを取得"""
        return copy.deepcopy(self.state)

    def restore_state(self, state: PlayerState):
        """プレイヤー状態を復元"""
        self.state = copy.deepcopy(state)
        self._update_hand_counts()

    def __str__(self) -> str:
        """文字列表現"""
        return f"Player({self.state.position.name}, {self.state.name}, Score: {self.state.score})"

    def __repr__(self) -> str:
        """詳細な文字列表現"""
        return (
            f"Player(position={self.state.position}, name='{self.state.name}', "
            f"score={self.state.score}, hand_size={self.get_hand_size()}, "
            f"calls={len(self.state.calls)}, riichi={self.state.riichi})"
        )
