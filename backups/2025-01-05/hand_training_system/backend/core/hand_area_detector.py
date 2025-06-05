"""
手牌領域検出モジュール
"""

import json
from pathlib import Path

import cv2
import numpy as np


class HandAreaDetector:
    """手牌領域を検出・管理するクラス"""

    # デフォルトの手牌領域位置（画面比率）
    DEFAULT_REGIONS = {
        "bottom": {"x": 0.15, "y": 0.75, "w": 0.7, "h": 0.15},  # 自分
        "top": {"x": 0.15, "y": 0.1, "w": 0.7, "h": 0.15},  # 対面
        "left": {"x": 0.05, "y": 0.3, "w": 0.15, "h": 0.4},  # 左
        "right": {"x": 0.8, "y": 0.3, "w": 0.15, "h": 0.4},  # 右
    }

    def __init__(self, config_file: str | None = None):
        """
        初期化

        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = Path(config_file) if config_file else None
        self.regions = {}
        self.frame_size = None

        # 設定を読み込み
        if self.config_file and self.config_file.exists():
            self.load_config()
        else:
            self.reset_to_default()

    def reset_to_default(self):
        """デフォルト設定にリセット"""
        self.regions = self.DEFAULT_REGIONS.copy()

    def set_frame_size(self, width: int, height: int):
        """
        フレームサイズを設定

        Args:
            width: 幅
            height: 高さ
        """
        self.frame_size = (width, height)

    def get_absolute_regions(self) -> dict[str, tuple[int, int, int, int]]:
        """
        絶対座標での領域を取得

        Returns:
            各プレイヤーの領域 {プレイヤー名: (x, y, w, h)}
        """
        if not self.frame_size:
            raise ValueError("フレームサイズが設定されていません")

        width, height = self.frame_size
        absolute_regions = {}

        for player, region in self.regions.items():
            x = int(region["x"] * width)
            y = int(region["y"] * height)
            w = int(region["w"] * width)
            h = int(region["h"] * height)
            absolute_regions[player] = (x, y, w, h)

        return absolute_regions

    def set_region(self, player: str, x: float, y: float, w: float, h: float):
        """
        手牌領域を設定（比率で指定）

        Args:
            player: プレイヤー名（"bottom", "top", "left", "right"）
            x, y, w, h: 領域の位置とサイズ（0.0～1.0の比率）
        """
        if player not in ["bottom", "top", "left", "right"]:
            raise ValueError(f"不正なプレイヤー名: {player}")

        self.regions[player] = {"x": x, "y": y, "w": w, "h": h}

    def detect_hand_areas(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        フレームから手牌領域を自動検出（試験的機能）

        Args:
            frame: 入力フレーム

        Returns:
            検出された領域のリスト [(x, y, w, h), ...]
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)

        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 手牌領域の候補を抽出
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 手牌領域の条件（横長の矩形）
            if w > h * 2 and w > frame.shape[1] * 0.3:
                candidates.append((x, y, w, h))

        # 候補を位置でソート
        candidates.sort(key=lambda r: r[1])  # y座標でソート

        return candidates[:4]  # 最大4つまで

    def extract_hand_region(self, frame: np.ndarray, player: str) -> np.ndarray | None:
        """
        フレームから指定プレイヤーの手牌領域を抽出

        Args:
            frame: 入力フレーム
            player: プレイヤー名

        Returns:
            手牌領域の画像
        """
        if player not in self.regions:
            return None

        # フレームサイズを自動設定
        if not self.frame_size:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        regions = self.get_absolute_regions()
        x, y, w, h = regions[player]

        # 領域を切り出し
        return frame[y : y + h, x : x + w]

    def draw_regions(
        self, frame: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        フレームに手牌領域を描画

        Args:
            frame: 入力フレーム
            color: 描画色 (B, G, R)

        Returns:
            領域を描画したフレーム
        """
        result = frame.copy()

        # フレームサイズを自動設定
        if not self.frame_size:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        regions = self.get_absolute_regions()

        for player, (x, y, w, h) in regions.items():
            # 矩形を描画
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # ラベルを描画
            label = f"{player}"
            cv2.putText(result, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result

    def save_config(self, filepath: str | None = None):
        """
        設定を保存

        Args:
            filepath: 保存先パス
        """
        save_path = filepath or self.config_file
        if not save_path:
            raise ValueError("保存先が指定されていません")

        config = {"regions": self.regions, "frame_size": self.frame_size}

        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, filepath: str | None = None):
        """
        設定を読み込み

        Args:
            filepath: 読み込み元パス
        """
        load_path = filepath or self.config_file
        if not load_path or not Path(load_path).exists():
            raise ValueError("設定ファイルが存在しません")

        with open(load_path) as f:
            config = json.load(f)

        self.regions = config.get("regions", {})
        self.frame_size = config.get("frame_size")

    def validate_regions(self) -> bool:
        """
        設定された領域の妥当性を検証

        Returns:
            妥当な場合True
        """
        # 必須プレイヤーが設定されているか
        required_players = ["bottom", "top", "left", "right"]
        for player in required_players:
            if player not in self.regions:
                return False

        # 領域が画面内に収まっているか
        for player, region in self.regions.items():
            if not (0 <= region["x"] <= 1 and 0 <= region["y"] <= 1):
                return False
            if not (0 < region["w"] <= 1 and 0 < region["h"] <= 1):
                return False
            if region["x"] + region["w"] > 1 or region["y"] + region["h"] > 1:
                return False

        return True
