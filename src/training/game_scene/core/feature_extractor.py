"""
対局画面の特徴量抽出器

画像から対局画面特有の特徴を抽出するモジュール
"""

import cv2
import numpy as np

from ....utils.logger import LoggerMixin


class FeatureExtractor(LoggerMixin):
    """対局画面の特徴量抽出器"""

    def __init__(self):
        """初期化"""
        # 麻雀卓の典型的な色範囲（HSV）
        self.table_color_lower = np.array([40, 40, 40])  # 緑色の下限
        self.table_color_upper = np.array([80, 255, 255])  # 緑色の上限

        # エッジ検出パラメータ
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

        self.logger.info("FeatureExtractor初期化完了")

    def extract_all_features(self, frame: np.ndarray) -> dict[str, float]:
        """
        フレームから全ての特徴量を抽出

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            特徴量の辞書
        """
        features = {}

        # 色特徴
        color_features = self.extract_color_features(frame)
        features.update(color_features)

        # 構造特徴
        structure_features = self.extract_structure_features(frame)
        features.update(structure_features)

        # テクスチャ特徴
        texture_features = self.extract_texture_features(frame)
        features.update(texture_features)

        # UI要素特徴
        ui_features = self.extract_ui_features(frame)
        features.update(ui_features)

        return features

    def extract_color_features(self, frame: np.ndarray) -> dict[str, float]:
        """
        色に基づく特徴量を抽出

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            色特徴量の辞書
        """
        features = {}

        # HSV変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 緑色（麻雀卓）の割合
        table_mask = cv2.inRange(hsv, self.table_color_lower, self.table_color_upper)
        table_ratio = np.sum(table_mask > 0) / (frame.shape[0] * frame.shape[1])
        features["green_ratio"] = table_ratio

        # 色の分散（対局画面は色が均一になりやすい）
        features["color_variance"] = np.std(frame)

        # 各チャンネルのヒストグラム統計
        for i, color in enumerate(["blue", "green", "red"]):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()

            # エントロピー
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features[f"{color}_entropy"] = entropy

            # ピーク位置
            peak_idx = np.argmax(hist)
            features[f"{color}_peak"] = peak_idx / 255.0

        return features

    def extract_structure_features(self, frame: np.ndarray) -> dict[str, float]:
        """
        構造的特徴量を抽出

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            構造特徴量の辞書
        """
        features = {}

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        edge_ratio = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        features["edge_ratio"] = edge_ratio

        # 直線検出（対局画面は牌や卓の直線が多い）
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
        )

        if lines is not None:
            features["line_count"] = len(lines)

            # 水平・垂直線の割合
            horizontal_lines = 0
            vertical_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 10 or angle > 170:  # 水平
                    horizontal_lines += 1
                elif 80 < angle < 100:  # 垂直
                    vertical_lines += 1

            features["horizontal_line_ratio"] = (
                horizontal_lines / len(lines) if lines is not None else 0
            )
            features["vertical_line_ratio"] = (
                vertical_lines / len(lines) if lines is not None else 0
            )
        else:
            features["line_count"] = 0
            features["horizontal_line_ratio"] = 0
            features["vertical_line_ratio"] = 0

        # コーナー検出
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        features["corner_count"] = len(corners) if corners is not None else 0

        return features

    def extract_texture_features(self, frame: np.ndarray) -> dict[str, float]:
        """
        テクスチャ特徴量を抽出

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            テクスチャ特徴量の辞書
        """
        features = {}

        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ラプラシアンによるシャープネス
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        features["sharpness"] = sharpness

        # ローカルバイナリパターン（簡易版）
        def simple_lbp(image):
            rows, cols = image.shape
            lbp = np.zeros_like(image)

            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    center = image[i, j]
                    code = 0

                    # 8近傍
                    neighbors = [
                        image[i - 1, j - 1],
                        image[i - 1, j],
                        image[i - 1, j + 1],
                        image[i, j + 1],
                        image[i + 1, j + 1],
                        image[i + 1, j],
                        image[i + 1, j - 1],
                        image[i, j - 1],
                    ]

                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= 1 << k

                    lbp[i, j] = code

            return lbp

        # LBPヒストグラム
        lbp = simple_lbp(gray)
        lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()

        # LBPエントロピー
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        features["lbp_entropy"] = lbp_entropy

        return features

    def extract_ui_features(self, frame: np.ndarray) -> dict[str, float]:
        """
        UI要素の特徴量を抽出

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            UI特徴量の辞書
        """
        features = {}

        height, width = frame.shape[:2]

        # 画面の特定領域の分析（点数表示エリアなど）
        # 上部20%（プレイヤー情報が表示される領域）
        top_region = frame[: int(height * 0.2), :]
        top_mean = np.mean(top_region)
        features["top_region_brightness"] = top_mean / 255.0

        # 下部20%（自分の手牌が表示される領域）
        bottom_region = frame[int(height * 0.8) :, :]
        bottom_mean = np.mean(bottom_region)
        features["bottom_region_brightness"] = bottom_mean / 255.0

        # 中央領域（卓が表示される領域）
        center_y1 = int(height * 0.3)
        center_y2 = int(height * 0.7)
        center_x1 = int(width * 0.2)
        center_x2 = int(width * 0.8)
        center_region = frame[center_y1:center_y2, center_x1:center_x2]

        # 中央領域の緑色割合
        center_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
        center_green_mask = cv2.inRange(center_hsv, self.table_color_lower, self.table_color_upper)
        center_green_ratio = np.sum(center_green_mask > 0) / center_green_mask.size
        features["center_green_ratio"] = center_green_ratio

        # テキスト領域の検出（簡易版）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 二値化
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 白い領域の割合（テキストや数字が含まれる可能性）
        white_ratio = np.sum(binary > 0) / binary.size
        features["white_area_ratio"] = white_ratio

        # 矩形領域の検出（UIパネルなど）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 小さすぎる領域は無視
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # 矩形らしい形状（アスペクト比が極端でない）
                if 0.2 < aspect_ratio < 5.0:
                    rect_count += 1

        features["rect_region_count"] = rect_count

        return features

    def get_feature_vector(self, frame: np.ndarray) -> np.ndarray:
        """
        特徴量を数値ベクトルとして取得

        Args:
            frame: 入力フレーム（BGR）

        Returns:
            特徴量ベクトル
        """
        features = self.extract_all_features(frame)

        # 特徴量を固定順序でベクトル化
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])

        return feature_vector
