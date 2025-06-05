"""
牌分割モジュール
"""

from pathlib import Path

import cv2
import numpy as np


class TileSplitter:
    """手牌領域から個々の牌を分割するクラス"""

    def __init__(self, tile_width_ratio: float = 0.07, tile_height_ratio: float = 0.9):
        """
        初期化

        Args:
            tile_width_ratio: 牌の幅の画像幅に対する比率
            tile_height_ratio: 牌の高さの領域高さに対する比率
        """
        self.tile_width_ratio = tile_width_ratio
        self.tile_height_ratio = tile_height_ratio

    def split_hand(self, hand_image: np.ndarray, num_tiles: int = 13) -> list[np.ndarray]:
        """
        手牌領域から個々の牌を分割

        Args:
            hand_image: 手牌領域の画像
            num_tiles: 牌の数（通常13枚または14枚）

        Returns:
            分割された牌の画像リスト
        """
        tiles = []
        h, w = hand_image.shape[:2]

        # 牌のサイズを推定
        tile_width = int(w * self.tile_width_ratio)
        tile_height = int(h * self.tile_height_ratio)

        # 牌の間隔を計算
        total_tiles_width = tile_width * num_tiles
        if total_tiles_width > w:
            # 牌が重なっている場合
            tile_width = w // num_tiles
            spacing = 0
        else:
            spacing = (w - total_tiles_width) // (num_tiles - 1) if num_tiles > 1 else 0

        # 各牌を切り出し
        y_offset = int((h - tile_height) / 2)

        for i in range(num_tiles):
            x_offset = i * (tile_width + spacing)

            # 領域をクリップ
            x_end = min(x_offset + tile_width, w)
            y_end = min(y_offset + tile_height, h)

            # 牌を切り出し
            tile = hand_image[y_offset:y_end, x_offset:x_end]

            if tile.size > 0:
                tiles.append(tile)

        return tiles

    def split_hand_auto(self, hand_image: np.ndarray) -> list[np.ndarray]:
        """
        手牌領域から牌を自動検出して分割

        Args:
            hand_image: 手牌領域の画像

        Returns:
            分割された牌の画像リスト
        """
        # 前処理
        gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)

        # 垂直方向の投影
        vertical_proj = np.sum(edges, axis=0)

        # ピークを検出（牌の境界）
        boundaries = self._find_tile_boundaries(vertical_proj)

        # 牌を切り出し
        tiles = []
        h = hand_image.shape[0]

        for i in range(len(boundaries) - 1):
            x_start = boundaries[i]
            x_end = boundaries[i + 1]

            # 牌を切り出し
            tile = hand_image[:, x_start:x_end]

            # 最小幅チェック
            if x_end - x_start > 20:
                tiles.append(tile)

        return tiles

    def _find_tile_boundaries(self, projection: np.ndarray, min_gap: int = 10) -> list[int]:
        """
        投影データから牌の境界を検出

        Args:
            projection: 垂直投影データ
            min_gap: 最小間隔

        Returns:
            境界位置のリスト
        """
        # しきい値を設定（平均値の半分）
        threshold = np.mean(projection) * 0.5

        # しきい値以上の領域を検出
        above_threshold = projection > threshold

        boundaries = [0]
        in_tile = False
        last_boundary = 0

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_tile:
                # 牌の開始
                if i - last_boundary > min_gap:
                    boundaries.append(i)
                    last_boundary = i
                in_tile = True
            elif not is_above and in_tile:
                # 牌の終了
                in_tile = False

        boundaries.append(len(projection) - 1)

        return boundaries

    def enhance_tile_image(self, tile_image: np.ndarray) -> np.ndarray:
        """
        牌画像を補正・強調

        Args:
            tile_image: 牌の画像

        Returns:
            補正された画像
        """
        # リサイズ（統一サイズに）
        target_size = (64, 96)  # 幅x高さ
        resized = cv2.resize(tile_image, target_size)

        # 明度補正
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE（Contrast Limited Adaptive Histogram Equalization）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def save_tiles(
        self, tiles: list[np.ndarray], output_dir: str, prefix: str = "tile", frame_number: int = 0
    ) -> list[str]:
        """
        分割した牌を保存

        Args:
            tiles: 牌画像のリスト
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
            frame_number: フレーム番号

        Returns:
            保存したファイルパスのリスト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, tile in enumerate(tiles):
            # ファイル名を生成
            filename = f"{prefix}_frame{frame_number:08d}_tile{i:02d}.jpg"
            filepath = output_path / filename

            # 画像を補正
            enhanced = self.enhance_tile_image(tile)

            # 保存
            cv2.imwrite(str(filepath), enhanced)
            saved_paths.append(str(filepath))

        return saved_paths

    def estimate_tile_count(self, hand_image: np.ndarray) -> int:
        """
        手牌の枚数を推定

        Args:
            hand_image: 手牌領域の画像

        Returns:
            推定された牌の枚数
        """
        # 自動分割を実行
        tiles = self.split_hand_auto(hand_image)

        # 13枚または14枚に近い値を返す
        count = len(tiles)
        if count <= 11:
            return 13
        elif count >= 15:
            return 14
        else:
            return count
