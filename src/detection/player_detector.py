"""
プレイヤー位置・手番検出モジュール

4人のプレイヤーの位置（東南西北）と
現在の手番を検出する
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ..utils.logger import LoggerMixin


class PlayerPosition(Enum):
    """プレイヤー位置"""

    EAST = "east"  # 東家（親）
    SOUTH = "south"  # 南家
    WEST = "west"  # 西家
    NORTH = "north"  # 北家


class TurnIndicator(Enum):
    """手番インジケーターのタイプ"""

    HIGHLIGHT = "highlight"  # ハイライト表示
    ARROW = "arrow"  # 矢印表示
    TIMER = "timer"  # タイマー表示
    NAME_COLOR = "name_color"  # 名前の色変化


@dataclass
class PlayerInfo:
    """プレイヤー情報"""

    position: PlayerPosition
    name: str | None
    is_dealer: bool  # 親かどうか
    is_active: bool  # 現在の手番かどうか
    bbox: tuple[int, int, int, int]  # プレイヤー情報の表示領域
    confidence: float


@dataclass
class PlayerDetectionResult:
    """プレイヤー検出結果"""

    frame_number: int
    timestamp: float
    players: list[PlayerInfo]
    active_position: PlayerPosition | None  # 現在の手番
    dealer_position: PlayerPosition  # 親の位置
    round_wind: str  # 場風（東、南、西、北）

    def get_player_by_position(self, position: PlayerPosition) -> PlayerInfo | None:
        """位置指定でプレイヤー情報を取得"""
        for player in self.players:
            if player.position == position:
                return player
        return None

    def get_active_player(self) -> PlayerInfo | None:
        """現在の手番のプレイヤーを取得"""
        for player in self.players:
            if player.is_active:
                return player
        return None


class PlayerDetector(LoggerMixin):
    """プレイヤー検出クラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}

        # プレイヤー情報の想定表示位置
        self.player_regions = self.config.get(
            "player_regions",
            {
                PlayerPosition.EAST: {"x": 0.8, "y": 0.45, "w": 0.15, "h": 0.1},
                PlayerPosition.SOUTH: {"x": 0.4, "y": 0.8, "w": 0.2, "h": 0.1},
                PlayerPosition.WEST: {"x": 0.05, "y": 0.45, "w": 0.15, "h": 0.1},
                PlayerPosition.NORTH: {"x": 0.4, "y": 0.05, "w": 0.2, "h": 0.1},
            },
        )

        # 手番検出の設定
        self.turn_indicators = self.config.get(
            "turn_indicators", [TurnIndicator.HIGHLIGHT, TurnIndicator.TIMER]
        )

        # 色の閾値（HSV）
        self.active_color_ranges = self.config.get(
            "active_color_ranges",
            {
                "yellow": [(20, 100, 100), (30, 255, 255)],  # 黄色ハイライト
                "green": [(40, 100, 100), (80, 255, 255)],  # 緑色ハイライト
                "red": [(0, 100, 100), (10, 255, 255)],  # 赤色ハイライト
            },
        )

        # 親マークの検出設定
        self.dealer_mark_template = None  # 親マークのテンプレート画像

        # 前回の検出結果（安定化用）
        self.prev_result: PlayerDetectionResult | None = None

        self.logger.info("PlayerDetector初期化完了")

    def detect_players(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> PlayerDetectionResult:
        """
        フレームからプレイヤー情報を検出

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            プレイヤー検出結果
        """
        players = []
        active_position = None
        dealer_position = PlayerPosition.EAST  # デフォルト

        # 各プレイヤー位置を検査
        for position, region_config in self.player_regions.items():
            player_info = self._detect_player_in_region(frame, position, region_config)

            if player_info:
                players.append(player_info)

                if player_info.is_active:
                    active_position = position

                if player_info.is_dealer:
                    dealer_position = position

        # 場風を推定
        round_wind = self._estimate_round_wind(dealer_position)

        result = PlayerDetectionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            players=players,
            active_position=active_position,
            dealer_position=dealer_position,
            round_wind=round_wind,
        )

        # 結果の安定化
        result = self._stabilize_result(result)

        # 手番変化の検出
        self._detect_turn_change(result)

        self.prev_result = result

        return result

    def _detect_player_in_region(
        self, frame: np.ndarray, position: PlayerPosition, region_config: dict
    ) -> PlayerInfo | None:
        """特定領域のプレイヤー情報を検出"""
        height, width = frame.shape[:2]

        # 領域を切り出し
        x1 = int(region_config["x"] * width)
        y1 = int(region_config["y"] * height)
        x2 = x1 + int(region_config["w"] * width)
        y2 = y1 + int(region_config["h"] * height)

        region = frame[y1:y2, x1:x2]

        # プレイヤー名の検出（OCRまたはテンプレート）
        player_name = self._detect_player_name(region)

        # 親マークの検出
        is_dealer = self._detect_dealer_mark(region)

        # 手番インジケーターの検出
        is_active, confidence = self._detect_turn_indicator(region, frame, (x1, y1, x2, y2))

        return PlayerInfo(
            position=position,
            name=player_name,
            is_dealer=is_dealer,
            is_active=is_active,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
        )

    def _detect_player_name(self, region: np.ndarray) -> str | None:
        """プレイヤー名を検出"""
        # TODO: OCRまたはテンプレートマッチングで実装
        # 現在は位置ベースの仮名を返す
        return None

    def _detect_dealer_mark(self, region: np.ndarray) -> bool:
        """親マークを検出"""
        # 赤い円形の親マークを探す
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 赤色の範囲
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 円形の検出
        circles = cv2.HoughCircles(
            red_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30,
        )

        if circles is not None:
            return True

        # テンプレートマッチングも試行
        if self.dealer_mark_template is not None:
            result = cv2.matchTemplate(region, self.dealer_mark_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > 0.7:
                return True

        return False

    def _detect_turn_indicator(
        self, region: np.ndarray, full_frame: np.ndarray, bbox: tuple
    ) -> tuple[bool, float]:
        """手番インジケーターを検出"""
        indicators_found = []

        # ハイライト検出
        if TurnIndicator.HIGHLIGHT in self.turn_indicators:
            is_highlighted, conf = self._detect_highlight(region)
            if is_highlighted:
                indicators_found.append(("highlight", conf))

        # タイマー検出
        if TurnIndicator.TIMER in self.turn_indicators:
            has_timer, conf = self._detect_timer(region)
            if has_timer:
                indicators_found.append(("timer", conf))

        # 矢印検出
        if TurnIndicator.ARROW in self.turn_indicators:
            has_arrow, conf = self._detect_arrow_pointing_to(full_frame, bbox)
            if has_arrow:
                indicators_found.append(("arrow", conf))

        # 最も信頼度の高いインジケーターを採用
        if indicators_found:
            indicators_found.sort(key=lambda x: x[1], reverse=True)
            return True, indicators_found[0][1]

        return False, 0.0

    def _detect_highlight(self, region: np.ndarray) -> tuple[bool, float]:
        """ハイライト表示を検出"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        max_ratio = 0.0

        for _color_name, (lower, upper) in self.active_color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = np.sum(mask > 0) / mask.size

            if ratio > max_ratio:
                max_ratio = ratio

        # 閾値以上の色が含まれていればハイライトとみなす
        threshold = 0.1
        is_highlighted = max_ratio > threshold
        confidence = min(max_ratio / threshold, 1.0) if is_highlighted else 0.0

        return is_highlighted, confidence

    def _detect_timer(self, region: np.ndarray) -> tuple[bool, float]:
        """タイマー表示を検出"""
        # 円形または数字の動的な表示を探す
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # 円形の検出
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50,
        )

        if circles is not None and len(circles[0]) > 0:
            # タイマーらしい円が見つかった
            return True, 0.8

        # 数字の変化を検出（簡易版）
        # TODO: より精密な実装

        return False, 0.0

    def _detect_arrow_pointing_to(
        self, frame: np.ndarray, target_bbox: tuple
    ) -> tuple[bool, float]:
        """対象領域を指す矢印を検出"""
        # 矢印の形状を探す
        # TODO: 実装
        return False, 0.0

    def _estimate_round_wind(self, dealer_position: PlayerPosition) -> str:
        """場風を推定"""
        # 親の位置から場風を推定
        # 実際のゲームでは画面表示から読み取る必要がある
        wind_map = {
            PlayerPosition.EAST: "東",
            PlayerPosition.SOUTH: "南",
            PlayerPosition.WEST: "西",
            PlayerPosition.NORTH: "北",
        }
        return wind_map.get(dealer_position, "東")

    def _stabilize_result(self, result: PlayerDetectionResult) -> PlayerDetectionResult:
        """検出結果を安定化"""
        if self.prev_result is None:
            return result

        # 手番が頻繁に変わらないように安定化
        if result.active_position != self.prev_result.active_position:
            # 信頼度が低い場合は前回の結果を維持
            current_player = result.get_active_player()
            if current_player and current_player.confidence < 0.7:
                result.active_position = self.prev_result.active_position
                for player in result.players:
                    player.is_active = player.position == result.active_position

        return result

    def _detect_turn_change(self, result: PlayerDetectionResult):
        """手番変化を検出してログ出力"""
        if self.prev_result is None:
            return

        if result.active_position != self.prev_result.active_position:
            self.logger.info(
                f"手番変化: {self.prev_result.active_position.value if self.prev_result.active_position else 'None'} "
                f"→ {result.active_position.value if result.active_position else 'None'}"
            )

        if result.dealer_position != self.prev_result.dealer_position:
            self.logger.info(
                f"親変化: {self.prev_result.dealer_position.value} → {result.dealer_position.value}"
            )
