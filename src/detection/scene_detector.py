"""
麻雀ゲームシーン検出モジュール

動画から対局の開始・終了、局の切り替わり、
重要なゲームイベントを検出する
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

from ..utils.logger import LoggerMixin


class SceneType(Enum):
    """シーンタイプの定義"""

    UNKNOWN = "unknown"
    MENU = "menu"  # メニュー画面
    GAME_START = "game_start"  # 対局開始
    GAME_PLAY = "game_play"  # 対局中
    ROUND_START = "round_start"  # 局開始（東1局など）
    ROUND_END = "round_end"  # 局終了
    GAME_END = "game_end"  # 対局終了
    RESULT = "result"  # 結果表示
    DORA_INDICATOR = "dora_indicator"  # ドラ表示
    RIICHI = "riichi"  # リーチ宣言
    TSUMO = "tsumo"  # ツモ
    RON = "ron"  # ロン
    DRAW = "draw"  # 流局


@dataclass
class SceneDetectionResult:
    """シーン検出結果"""

    scene_type: SceneType
    confidence: float
    frame_number: int
    timestamp: float
    metadata: dict[str, Any]

    def is_game_boundary(self) -> bool:
        """ゲームの境界（開始・終了）かどうか"""
        return self.scene_type in [
            SceneType.GAME_START,
            SceneType.GAME_END,
            SceneType.ROUND_START,
            SceneType.ROUND_END,
        ]


class SceneDetector(LoggerMixin):
    """シーン検出クラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}

        # 検出パラメータ
        self.scene_change_threshold = self.config.get("scene_change_threshold", 0.3)
        self.text_detection_confidence = self.config.get("text_detection_confidence", 0.8)

        # テンプレートマッチング用の画像を保存
        self.templates: dict[SceneType, list[np.ndarray]] = {}

        # 前フレームの情報を保持
        self.prev_frame: np.ndarray | None = None
        self.prev_scene_type = SceneType.UNKNOWN

        # OCR設定（将来的に点数読み取りと統合）
        self.enable_ocr = self.config.get("enable_ocr", False)

        self.logger.info("SceneDetector初期化完了")

    def detect_scene(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> SceneDetectionResult:
        """
        フレームからシーンを検出

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            シーン検出結果
        """
        # シーン変化の検出
        scene_changed = self._detect_scene_change(frame)

        # 各種シーン検出手法を実行
        scene_type = SceneType.UNKNOWN
        confidence = 0.0
        metadata: dict[str, Any] = {}

        # 1. 色ヒストグラムによる判定
        hist_result = self._detect_by_histogram(frame)

        # 2. テキスト検出による判定
        text_result = self._detect_by_text(frame)

        # 3. 特徴的なUIパターンの検出
        ui_result = self._detect_ui_patterns(frame)

        # 4. 牌の配置パターンによる判定
        tile_result = self._detect_by_tile_arrangement(frame)

        # 結果の統合
        scene_type, confidence, metadata = self._integrate_results(
            hist_result, text_result, ui_result, tile_result, scene_changed
        )

        # 前フレーム情報の更新
        self.prev_frame = frame.copy()
        self.prev_scene_type = scene_type

        return SceneDetectionResult(
            scene_type=scene_type,
            confidence=confidence,
            frame_number=frame_number,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _detect_scene_change(self, frame: np.ndarray) -> bool:
        """シーン変化を検出"""
        if self.prev_frame is None:
            return True

        # フレーム差分を計算
        diff = cv2.absdiff(frame, self.prev_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 差分の平均値で判定
        mean_diff = np.mean(diff_gray) / 255.0

        return mean_diff > self.scene_change_threshold

    def _detect_by_histogram(self, frame: np.ndarray) -> tuple[SceneType, float, dict]:
        """色ヒストグラムによるシーン検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ヒストグラムを計算
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        # 正規化
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)

        # 特徴的な色パターンの検出
        # 緑色が支配的 → ゲームプレイ中
        green_ratio = np.sum(hist_h[30:90])  # 緑色の範囲
        if green_ratio > 0.6:
            return SceneType.GAME_PLAY, green_ratio, {"green_ratio": green_ratio}

        # 暗い画面 → メニューまたは結果画面
        dark_ratio = np.sum(hist_v[:50])
        if dark_ratio > 0.7:
            return SceneType.MENU, dark_ratio, {"dark_ratio": dark_ratio}

        return SceneType.UNKNOWN, 0.0, {}

    def _detect_by_text(self, frame: np.ndarray) -> tuple[SceneType, float, dict]:
        """テキスト検出によるシーン判定"""
        if not self.enable_ocr:
            return SceneType.UNKNOWN, 0.0, {}

        # TODO: OCR実装（将来的に点数読み取りモジュールと統合）
        # 現在は簡易的なパターンマッチングで代替

        # 特定のテキストパターンを探す
        # "東1局"、"南2局"などの文字列を検出

        return SceneType.UNKNOWN, 0.0, {}

    def _detect_ui_patterns(self, frame: np.ndarray) -> tuple[SceneType, float, dict]:
        """UI要素のパターン検出"""
        height, width = frame.shape[:2]

        # 画面を領域に分割
        regions = {
            "top": frame[0 : height // 4, :],
            "bottom": frame[3 * height // 4 : height, :],
            "center": frame[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4],
        }

        # 各領域の特徴を分析
        features = {}

        # 上部に情報表示がある → ゲーム中
        top_edges = self._detect_edges(regions["top"])
        features["top_edge_density"] = np.mean(top_edges) / 255.0

        # 下部に手牌がある → ゲーム中
        bottom_edges = self._detect_edges(regions["bottom"])
        features["bottom_edge_density"] = np.mean(bottom_edges) / 255.0

        # 中央に大きなテキストや画像 → メニューまたは結果
        center_edges = self._detect_edges(regions["center"])
        features["center_edge_density"] = np.mean(center_edges) / 255.0

        # パターンに基づいて判定
        if features["top_edge_density"] > 0.1 and features["bottom_edge_density"] > 0.2:
            return SceneType.GAME_PLAY, 0.7, features

        if features["center_edge_density"] > 0.3:
            return SceneType.MENU, 0.6, features

        return SceneType.UNKNOWN, 0.0, features

    def _detect_edges(self, region: np.ndarray) -> np.ndarray:
        """エッジ検出"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges

    def _detect_by_tile_arrangement(self, frame: np.ndarray) -> tuple[SceneType, float, dict]:
        """牌の配置パターンによる検出"""
        # TODO: TileDetectorと連携して牌の配置を分析
        # - 14枚の手牌が整列 → ROUND_START
        # - 牌が散らばっている → GAME_PLAY
        # - 牌がない → MENU/RESULT

        return SceneType.UNKNOWN, 0.0, {}

    def _integrate_results(
        self,
        hist_result: tuple,
        text_result: tuple,
        ui_result: tuple,
        tile_result: tuple,
        scene_changed: bool,
    ) -> tuple[SceneType, float, dict]:
        """複数の検出結果を統合"""
        results = [hist_result, text_result, ui_result, tile_result]

        # 最も信頼度の高い結果を採用
        best_result = max(results, key=lambda x: x[1])
        scene_type, confidence, metadata = best_result

        # シーン変化があった場合は信頼度を調整
        if scene_changed:
            confidence *= 1.2  # シーン変化時は信頼度を上げる
        else:
            # 前回と同じシーンタイプなら信頼度を上げる
            if scene_type == self.prev_scene_type:
                confidence *= 1.1

        confidence = min(confidence, 1.0)  # 最大値を1.0に制限

        # メタデータを統合
        all_metadata = {"scene_changed": scene_changed}
        for result in results:
            all_metadata.update(result[2])

        return scene_type, confidence, all_metadata

    def detect_game_boundaries(
        self, video_path: str, sample_interval: int = 30
    ) -> list[SceneDetectionResult]:
        """
        動画全体からゲームの境界を検出

        Args:
            video_path: 動画ファイルパス
            sample_interval: サンプリング間隔（フレーム数）

        Returns:
            ゲーム境界の検出結果リスト
        """
        boundaries: list[SceneDetectionResult] = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"動画を開けません: {video_path}")
            return boundaries

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_number = 0
            while frame_number < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    break

                timestamp = frame_number / fps
                result = self.detect_scene(frame, frame_number, timestamp)

                if result.is_game_boundary():
                    boundaries.append(result)
                    self.logger.info(
                        f"ゲーム境界を検出: {result.scene_type.value} "
                        f"at frame {frame_number} ({timestamp:.2f}s)"
                    )

                frame_number += sample_interval

        finally:
            cap.release()

        return boundaries

    def reset(self):
        """検出器の状態をリセット"""
        self.prev_frame = None
        self.prev_scene_type = SceneType.UNKNOWN
        self.logger.debug("SceneDetectorをリセットしました")
