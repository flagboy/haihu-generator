"""
統合された手牌領域検出モジュール
hand_labeling_systemとhand_training_systemの機能を統合
"""

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class UnifiedHandAreaDetector:
    """統合された手牌領域検出クラス"""

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

        # 色範囲による検出設定（hand_detector.pyから移植）
        self.color_ranges = {
            "white": {"lower": np.array([0, 0, 200]), "upper": np.array([180, 30, 255])},
            "green": {"lower": np.array([40, 40, 40]), "upper": np.array([80, 255, 255])},
        }

        # 設定を読み込み
        if self.config_file and self.config_file.exists():
            self.load_config()
        else:
            self.reset_to_default()

    def reset_to_default(self):
        """デフォルト設定にリセット"""
        self.regions = self.DEFAULT_REGIONS.copy()
        logger.info("手牌領域をデフォルト設定にリセットしました")

    def set_frame_size(self, width: int, height: int):
        """フレームサイズを設定"""
        self.frame_size = (width, height)
        logger.debug(f"フレームサイズを設定: {width}x{height}")

    def detect_areas(
        self, frame: np.ndarray, auto_detect: bool = True
    ) -> dict[str, dict[str, int]]:
        """
        自動検出機能（hand_detector.pyから移植）

        Args:
            frame: 入力フレーム
            auto_detect: 自動検出を行うか

        Returns:
            検出された手牌領域（絶対座標）
        """
        height, width = frame.shape[:2]

        # フレームサイズを設定
        if not self.frame_size:
            self.set_frame_size(width, height)

        # デフォルト領域を絶対座標に変換
        regions = self._get_absolute_regions()

        # 自動検出を試みる
        if auto_detect:
            detected_regions = self._auto_detect_regions(frame)

            # 検出された領域でデフォルトを更新
            for player, region in detected_regions.items():
                if self._validate_region(region, width, height):
                    regions[player] = region
                    logger.debug(f"{player}の手牌領域を自動検出しました")

        return regions

    def _auto_detect_regions(self, frame: np.ndarray) -> dict[str, dict[str, int]]:
        """
        画像処理による自動検出（hand_detector.pyから移植）
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

            # 下側プレイヤー（bottom）
            if cy > height * 0.7 and candidate["position"] == "horizontal":
                if "bottom" not in detected or candidate["area"] > detected["bottom"]["area"]:
                    detected["bottom"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 上側プレイヤー（top）
            elif cy < height * 0.3 and candidate["position"] == "horizontal":
                if "top" not in detected or candidate["area"] > detected["top"]["area"]:
                    detected["top"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 右側プレイヤー（right）
            elif cx > width * 0.7 and candidate["position"] == "vertical":
                if "right" not in detected or candidate["area"] > detected["right"]["area"]:
                    detected["right"] = {
                        "x": candidate["x"],
                        "y": candidate["y"],
                        "w": candidate["w"],
                        "h": candidate["h"],
                        "area": candidate["area"],
                    }

            # 左側プレイヤー（left）
            elif (
                cx < width * 0.3
                and candidate["position"] == "vertical"
                and ("left" not in detected or candidate["area"] > detected["left"]["area"])
            ):
                detected["left"] = {
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

    def set_manual_area(self, player: str, area: dict[str, float]):
        """
        手動設定機能（hand_area_detector.pyから移植）

        Args:
            player: プレイヤー名（"bottom", "top", "left", "right"）
            area: 領域情報（比率）{"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.2}
        """
        if player not in ["bottom", "top", "left", "right"]:
            raise ValueError(f"不正なプレイヤー名: {player}")

        # 値の検証
        if not self._validate_relative_region(area):
            raise ValueError("不正な領域値です")

        self.regions[player] = area
        logger.info(f"{player}の手牌領域を手動設定しました")

    def refine_regions(
        self, frame: np.ndarray, regions: dict[str, dict[str, int]]
    ) -> dict[str, dict[str, int]]:
        """
        検出された領域を微調整（hand_detector.pyから移植）
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
        """投影データから境界を検出"""
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

    def _validate_region(self, region: dict[str, int], width: int, height: int) -> bool:
        """領域の妥当性を検証（絶対座標）"""
        # 境界チェック
        if region["x"] < 0 or region["y"] < 0:
            return False
        if region["x"] + region["w"] > width:
            return False
        if region["y"] + region["h"] > height:
            return False

        # サイズチェック
        return region["w"] >= 50 and region["h"] >= 30

    def _validate_relative_region(self, region: dict[str, float]) -> bool:
        """領域の妥当性を検証（相対座標）"""
        if not (0 <= region["x"] <= 1 and 0 <= region["y"] <= 1):
            return False
        if not (0 < region["w"] <= 1 and 0 < region["h"] <= 1):
            return False
        return not (region["x"] + region["w"] > 1 or region["y"] + region["h"] > 1)

    def _get_absolute_regions(self) -> dict[str, dict[str, int]]:
        """相対座標から絶対座標への変換"""
        if not self.frame_size:
            raise ValueError("フレームサイズが設定されていません")

        width, height = self.frame_size
        absolute_regions = {}

        for player, region in self.regions.items():
            absolute_regions[player] = {
                "x": int(region["x"] * width),
                "y": int(region["y"] * height),
                "w": int(region["w"] * width),
                "h": int(region["h"] * height),
            }

        return absolute_regions

    def extract_hand_region(self, frame: np.ndarray, player: str) -> np.ndarray | None:
        """フレームから指定プレイヤーの手牌領域を抽出"""
        if player not in self.regions:
            return None

        # フレームサイズを自動設定
        if not self.frame_size:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        regions = self._get_absolute_regions()
        region = regions[player]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        # 領域を切り出し
        return frame[y : y + h, x : x + w].copy()

    def draw_regions(self, frame: np.ndarray) -> np.ndarray:
        """検出された領域を描画"""
        result = frame.copy()

        # フレームサイズを自動設定
        if not self.frame_size:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        regions = self._get_absolute_regions()

        colors = {
            "bottom": (0, 255, 0),  # 緑
            "right": (255, 0, 0),  # 青
            "top": (0, 0, 255),  # 赤
            "left": (255, 255, 0),  # シアン
        }

        player_names = {"bottom": "自分", "top": "対面", "left": "左", "right": "右"}

        for player, region in regions.items():
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            color = colors.get(player, (255, 255, 255))

            # 矩形を描画
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # ラベルを描画
            label = player_names.get(player, player)
            cv2.putText(result, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return result

    def save_config(self, filepath: str | None = None):
        """設定を保存"""
        save_path = filepath or self.config_file
        if not save_path:
            raise ValueError("保存先が指定されていません")

        config = {"regions": self.regions, "frame_size": self.frame_size}

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"手牌領域設定を保存しました: {save_path}")

    def load_config(self, filepath: str | None = None):
        """設定を読み込み"""
        load_path = filepath or self.config_file
        if not load_path or not Path(load_path).exists():
            raise ValueError("設定ファイルが存在しません")

        with open(load_path, encoding="utf-8") as f:
            config = json.load(f)

        self.regions = config.get("regions", {})
        self.frame_size = config.get("frame_size")

        logger.info(f"手牌領域設定を読み込みました: {load_path}")

    def validate_regions(self) -> bool:
        """設定された領域の妥当性を検証"""
        # 必須プレイヤーが設定されているか
        required_players = ["bottom", "top", "left", "right"]
        for player in required_players:
            if player not in self.regions:
                logger.warning(f"必須プレイヤー{player}が設定されていません")
                return False

        # 各領域の妥当性チェック
        for player, region in self.regions.items():
            if not self._validate_relative_region(region):
                logger.warning(f"{player}の領域設定が不正です")
                return False

        return True
