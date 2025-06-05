"""
牌分割モジュール
hand_training_systemから移植
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


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

        # 標準的な牌のサイズ
        self.standard_tile_size = (64, 96)  # 幅x高さ

        logger.info("牌分割モジュールを初期化しました")

    def split_tiles(self, hand_image: np.ndarray, num_tiles: int | None = None) -> list[np.ndarray]:
        """
        手牌領域から個々の牌を分割（メインメソッド）

        Args:
            hand_image: 手牌領域の画像
            num_tiles: 牌の数（Noneの場合は自動検出）

        Returns:
            分割された牌の画像リスト
        """
        if hand_image is None or hand_image.size == 0:
            logger.warning("空の手牌画像が渡されました")
            return []

        # 牌数が指定されていない場合は自動検出
        if num_tiles is None:
            tiles = self.split_hand_auto(hand_image)
            logger.info(f"自動検出で{len(tiles)}枚の牌を分割しました")
        else:
            tiles = self.split_hand_fixed(hand_image, num_tiles)
            logger.info(f"固定分割で{len(tiles)}枚の牌を分割しました")

        return tiles

    def split_hand_fixed(self, hand_image: np.ndarray, num_tiles: int = 13) -> list[np.ndarray]:
        """
        手牌領域から固定数の牌を分割

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
        if tile_image.size == 0:
            return tile_image

        # リサイズ（統一サイズに）
        resized = cv2.resize(tile_image, self.standard_tile_size)

        # 明度補正
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE（Contrast Limited Adaptive Histogram Equalization）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def adjust_boundaries(
        self, boundaries: list[tuple[int, int]], hand_image: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        牌の境界を調整（新機能）

        Args:
            boundaries: 初期境界リスト [(x_start, x_end), ...]
            hand_image: 手牌領域の画像

        Returns:
            調整された境界リスト
        """
        adjusted = []

        for x_start, x_end in boundaries:
            # 境界付近のエッジを再検出
            roi = hand_image[:, max(0, x_start - 10) : min(hand_image.shape[1], x_end + 10)]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges_roi = cv2.Canny(gray_roi, 50, 150)

            # 左右の境界を微調整
            left_proj = np.sum(edges_roi[:, :20], axis=0)
            right_proj = np.sum(edges_roi[:, -20:], axis=0)

            # 最適な境界を見つける
            left_adjust = np.argmax(left_proj) - 10
            right_adjust = np.argmax(right_proj) - 10

            adjusted.append(
                (max(0, x_start + left_adjust), min(hand_image.shape[1], x_end + right_adjust))
            )

        return adjusted

    def detect_horizontal_tiles(self, hand_image: np.ndarray) -> list[int]:
        """
        横向き牌の検出（新機能）

        Args:
            hand_image: 手牌領域の画像

        Returns:
            横向き牌のインデックスリスト
        """
        tiles = self.split_hand_auto(hand_image)
        horizontal_indices = []

        for i, tile in enumerate(tiles):
            h, w = tile.shape[:2]
            aspect_ratio = w / h if h > 0 else 0

            # 横向きの判定（通常の牌よりも横長）
            if aspect_ratio > 1.2:
                horizontal_indices.append(i)
                logger.debug(f"横向き牌を検出: インデックス {i}")

        return horizontal_indices

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

        logger.info(f"{len(saved_paths)}枚の牌画像を保存しました: {output_dir}")
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

    def get_tile_positions(
        self, tiles: list[np.ndarray], hand_image_width: int
    ) -> list[dict[str, int]]:
        """
        各牌の位置情報を取得（新機能）

        Args:
            tiles: 分割された牌のリスト
            hand_image_width: 手牌領域の幅

        Returns:
            各牌の位置情報 [{"x": x, "y": y, "w": w, "h": h}, ...]
        """
        positions = []
        current_x = 0

        for tile in tiles:
            h, w = tile.shape[:2]
            positions.append({"x": current_x, "y": 0, "w": w, "h": h})
            current_x += w

        return positions
