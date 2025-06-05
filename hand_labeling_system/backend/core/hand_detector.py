"""
手牌検出モジュール
動画フレームから手牌領域を検出する
"""

import json
from pathlib import Path

import cv2
import numpy as np


class HandDetector:
    """手牌領域検出クラス"""

    def __init__(self, config_file: str | None = None):
        """
        初期化

        Args:
            config_file: 設定ファイルのパス
        """
        self.default_regions = {
            "player1": {"x": 0.2, "y": 0.75, "w": 0.6, "h": 0.15},  # 下側プレイヤー
            "player2": {"x": 0.75, "y": 0.2, "w": 0.15, "h": 0.6},  # 右側プレイヤー
            "player3": {"x": 0.2, "y": 0.05, "w": 0.6, "h": 0.15},  # 上側プレイヤー
            "player4": {"x": 0.05, "y": 0.2, "w": 0.15, "h": 0.6},  # 左側プレイヤー
        }

        # 設定ファイルから読み込み
        if config_file and Path(config_file).exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
                self.default_regions = config.get("hand_regions", self.default_regions)

        # テンプレートマッチング用の設定
        self.templates = []
        self.color_ranges = {
            "white": {"lower": np.array([0, 0, 200]), "upper": np.array([180, 30, 255])},
            "green": {"lower": np.array([40, 40, 40]), "upper": np.array([80, 255, 255])},
        }

    def detect_hand_regions(self, frame: np.ndarray) -> dict[str, dict[str, int]]:
        """
        フレームから手牌領域を自動検出

        Args:
            frame: 入力フレーム

        Returns:
            検出された手牌領域 {"player1": {"x": x, "y": y, "w": w, "h": h}, ...}
        """
        height, width = frame.shape[:2]
        regions = {}

        # まずデフォルト領域を絶対座標に変換
        for player, region in self.default_regions.items():
            regions[player] = {
                "x": int(region["x"] * width),
                "y": int(region["y"] * height),
                "w": int(region["w"] * width),
                "h": int(region["h"] * height),
            }

        # 自動検出を試みる
        detected_regions = self._auto_detect_regions(frame)

        # 検出された領域でデフォルトを更新
        for player, region in detected_regions.items():
            if self._validate_region(region, width, height):
                regions[player] = region

        return regions

    def _auto_detect_regions(self, frame: np.ndarray) -> dict[str, dict[str, int]]:
        """
        画像処理による自動検出

        Args:
            frame: 入力フレーム

        Returns:
            検出された領域
        """
        detected = {}

        # HSV変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 牌の白色を検出
        mask_white = cv2.inRange(
            hsv, self.color_ranges["white"]["lower"], self.color_ranges["white"]["upper"]
        )

        # モルフォロジー処理
        kernel = np.ones((5, 5), np.uint8)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

        # 輪郭検出
        contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 手牌領域の候補を抽出
        hand_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 最小面積フィルタ
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # 手牌は横長の領域
                if aspect_ratio > 2.0:
                    hand_candidates.append(
                        {"x": x, "y": y, "w": w, "h": h, "area": area, "position": "horizontal"}
                    )
                # 縦向きの手牌
                elif aspect_ratio < 0.5 and aspect_ratio > 0:
                    hand_candidates.append(
                        {"x": x, "y": y, "w": w, "h": h, "area": area, "position": "vertical"}
                    )

        # 位置に基づいてプレイヤーを割り当て
        height, width = frame.shape[:2]

        for candidate in hand_candidates:
            cx = candidate["x"] + candidate["w"] // 2
            cy = candidate["y"] + candidate["h"] // 2

            # 下側プレイヤー（player1）
            if cy > height * 0.7 and candidate["position"] == "horizontal":
                if "player1" not in detected or candidate["area"] > detected["player1"]["area"]:
                    detected["player1"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 上側プレイヤー（player3）
            elif cy < height * 0.3 and candidate["position"] == "horizontal":
                if "player3" not in detected or candidate["area"] > detected["player3"]["area"]:
                    detected["player3"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 右側プレイヤー（player2）
            elif cx > width * 0.7 and candidate["position"] == "vertical":
                if "player2" not in detected or candidate["area"] > detected["player2"]["area"]:
                    detected["player2"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 左側プレイヤー（player4）
            elif cx < width * 0.3 and candidate["position"] == "vertical":
                if "player4" not in detected or candidate["area"] > detected["player4"]["area"]:
                    detected["player4"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

        # areaフィールドを削除
        for player in detected:
            if "area" in detected[player]:
                del detected[player]["area"]

        return detected

    def _validate_region(self, region: dict[str, int], width: int, height: int) -> bool:
        """
        領域の妥当性を検証

        Args:
            region: 検証する領域
            width: フレーム幅
            height: フレーム高さ

        Returns:
            妥当な場合True
        """
        # 境界チェック
        if region["x"] < 0 or region["y"] < 0:
            return False
        if region["x"] + region["w"] > width:
            return False
        if region["y"] + region["h"] > height:
            return False

        # サイズチェック
        if region["w"] < 50 or region["h"] < 30:
            return False

        return True

    def refine_regions(
        self, frame: np.ndarray, regions: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """
        検出された領域を微調整

        Args:
            frame: 入力フレーム
            regions: 初期領域

        Returns:
            調整された領域
        """
        refined = {}

        for player, region in regions.items():
            # 領域を切り出し
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            roi = frame[y : y + h, x : x + w]

            # エッジ検出で正確な境界を見つける
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # 投影による境界検出
            # 水平方向の手牌
            if w > h * 1.5:
                h_proj = np.sum(edges, axis=1)
                v_proj = np.sum(edges, axis=0)

                # 上下の境界を調整
                top = self._find_boundary(h_proj, 0, len(h_proj) // 2, True)
                bottom = self._find_boundary(h_proj, len(h_proj) // 2, len(h_proj), False)

                # 左右の境界を調整
                left = self._find_boundary(v_proj, 0, len(v_proj) // 2, True)
                right = self._find_boundary(v_proj, len(v_proj) // 2, len(v_proj), False)

            # 垂直方向の手牌
            else:
                h_proj = np.sum(edges, axis=1)
                v_proj = np.sum(edges, axis=0)

                # 境界を調整
                left = self._find_boundary(v_proj, 0, len(v_proj) // 2, True)
                right = self._find_boundary(v_proj, len(v_proj) // 2, len(v_proj), False)
                top = self._find_boundary(h_proj, 0, len(h_proj) // 2, True)
                bottom = self._find_boundary(h_proj, len(h_proj) // 2, len(h_proj), False)

            # 調整された領域を保存
            refined[player] = {"x": x + left, "y": y + top, "w": right - left, "h": bottom - top}

        return refined

    def _find_boundary(
        self, projection: np.ndarray, start: int, end: int, forward: bool = True
    ) -> int:
        """
        投影データから境界を検出

        Args:
            projection: 投影データ
            start: 開始位置
            end: 終了位置
            forward: 前方向に探索するか

        Returns:
            境界位置
        """
        threshold = np.mean(projection) * 0.3

        if forward:
            for i in range(start, end):
                if projection[i] > threshold:
                    return i
            return start
        else:
            for i in range(end - 1, start - 1, -1):
                if projection[i] > threshold:
                    return i
            return end - 1

    def draw_regions(self, frame: np.ndarray, regions: dict[str, dict[str, int]]) -> np.ndarray:
        """
        検出された領域を描画

        Args:
            frame: 入力フレーム
            regions: 描画する領域

        Returns:
            描画されたフレーム
        """
        result = frame.copy()
        colors = {
            "player1": (0, 255, 0),  # 緑
            "player2": (255, 0, 0),  # 青
            "player3": (0, 0, 255),  # 赤
            "player4": (255, 255, 0),  # シアン
        }

        for player, region in regions.items():
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            color = colors.get(player, (255, 255, 255))

            # 矩形を描画
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # ラベルを描画
            label = player.replace("player", "P")
            cv2.putText(result, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result

    def save_config(
        self, regions: dict[str, dict[str, int]], config_file: str, frame_size: tuple[int, int]
    ):
        """
        検出結果を設定ファイルに保存

        Args:
            regions: 保存する領域
            config_file: 設定ファイルのパス
            frame_size: フレームサイズ (width, height)
        """
        width, height = frame_size

        # 相対座標に変換
        relative_regions = {}
        for player, region in regions.items():
            relative_regions[player] = {
                "x": region["x"] / width,
                "y": region["y"] / height,
                "w": region["w"] / width,
                "h": region["h"] / height,
            }

        # 設定を保存
        config = {
            "hand_regions": relative_regions,
            "frame_size": {"width": width, "height": height},
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def extract_hand_image(self, frame: np.ndarray, region: dict[str, int]) -> np.ndarray:
        """
        フレームから手牌領域を切り出し

        Args:
            frame: 入力フレーム
            region: 切り出す領域

        Returns:
            切り出された画像
        """
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        return frame[y : y + h, x : x + w].copy()
