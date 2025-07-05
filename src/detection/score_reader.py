"""
麻雀ゲーム点数読み取りモジュール

画面上の点数表示をOCRで読み取り、
各プレイヤーの点数を追跡する
"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    import pytesseract
    from PIL import Image

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    Image = None

from ..utils.logger import LoggerMixin


@dataclass
class PlayerScore:
    """プレイヤーの点数情報"""

    player_position: str  # "east", "south", "west", "north"
    score: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class ScoreReadingResult:
    """点数読み取り結果"""

    frame_number: int
    timestamp: float
    scores: list[PlayerScore]
    total_confidence: float

    def get_score_by_position(self, position: str) -> PlayerScore | None:
        """位置指定でプレイヤーの点数を取得"""
        for score in self.scores:
            if score.player_position == position:
                return score
        return None

    def is_valid(self, min_confidence: float = 0.7) -> bool:
        """読み取り結果が有効かどうか"""
        return (
            len(self.scores) == 4
            and self.total_confidence >= min_confidence
            and sum(s.score for s in self.scores) == 100000  # 4人麻雀の初期点数合計
        )


class ScoreReader(LoggerMixin):
    """点数読み取りクラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}

        # OCR設定
        self.ocr_lang = self.config.get("ocr_lang", "eng")  # 数字は英語でOK
        self.preprocessing = self.config.get("preprocessing", True)

        # 点数表示の想定位置（画面サイズに対する相対位置）
        self.score_regions = self.config.get(
            "score_regions",
            {
                "east": {"x": 0.85, "y": 0.5, "w": 0.1, "h": 0.05},  # 右
                "south": {"x": 0.45, "y": 0.85, "w": 0.1, "h": 0.05},  # 下
                "west": {"x": 0.05, "y": 0.5, "w": 0.1, "h": 0.05},  # 左
                "north": {"x": 0.45, "y": 0.05, "w": 0.1, "h": 0.05},  # 上
            },
        )

        # 数字認識の設定
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.valid_score_range = (0, 500000)  # 有効な点数範囲

        # 前回の読み取り結果（変化検出用）
        self.prev_scores: dict[str, int] | None = None

        if not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseractが利用できません。OCR機能は制限されます")

        self.logger.info("ScoreReader初期化完了")

    def read_scores(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        custom_regions: dict | None = None,
    ) -> ScoreReadingResult:
        """
        フレームから点数を読み取る

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ
            custom_regions: カスタム領域設定

        Returns:
            点数読み取り結果
        """
        height, width = frame.shape[:2]
        regions = custom_regions or self.score_regions

        scores = []
        total_confidence = 0.0

        for position, region_config in regions.items():
            # 領域を切り出し
            x1 = int(region_config["x"] * width)
            y1 = int(region_config["y"] * height)
            x2 = x1 + int(region_config["w"] * width)
            y2 = y1 + int(region_config["h"] * height)

            region = frame[y1:y2, x1:x2]

            # 点数を読み取り
            score_value, confidence = self._read_score_from_region(region)

            if score_value is not None:
                scores.append(
                    PlayerScore(
                        player_position=position,
                        score=score_value,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
                total_confidence += confidence

        # 平均信頼度を計算
        if scores:
            total_confidence /= len(scores)

        result = ScoreReadingResult(
            frame_number=frame_number,
            timestamp=timestamp,
            scores=scores,
            total_confidence=total_confidence,
        )

        # 変化検出とログ出力
        self._detect_score_changes(result)

        return result

    def _read_score_from_region(self, region: np.ndarray) -> tuple[int | None, float]:
        """
        領域から点数を読み取る

        Args:
            region: 点数表示領域

        Returns:
            (点数, 信頼度)のタプル
        """
        if not TESSERACT_AVAILABLE:
            # OCRが利用できない場合はダミー値を返す
            return self._detect_by_template(region)

        # 前処理
        processed = self._preprocess_for_ocr(region) if self.preprocessing else region

        try:
            # OCR実行
            custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,"
            text = pytesseract.image_to_string(processed, config=custom_config)

            # データ付きで取得
            data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT, config=custom_config
            )

            # 数値を抽出
            score_value = self._extract_score_from_text(text)
            confidence = self._calculate_confidence(data)

            # 有効性チェック
            if score_value is not None and self._is_valid_score(score_value):
                return score_value, confidence

        except Exception as e:
            self.logger.debug(f"OCRエラー: {e}")

        # OCRが失敗した場合はテンプレートマッチングを試行
        return self._detect_by_template(region)

    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """OCR用の前処理"""
        # グレースケール変換
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region.copy()

        # リサイズ（OCRの精度向上のため）
        scale = 3
        width = gray.shape[1] * scale
        height = gray.shape[0] * scale
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # ノイズ除去
        denoised = cv2.fastNlMeansDenoising(resized)

        # 二値化
        # 適応的閾値処理
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 反転（白背景に黒文字にする）
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # モルフォロジー処理
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return morphed

    def _extract_score_from_text(self, text: str) -> int | None:
        """テキストから点数を抽出"""
        # 数字以外を除去
        numbers = "".join(c for c in text if c.isdigit() or c == ",")
        numbers = numbers.replace(",", "")

        if numbers:
            try:
                score = int(numbers)
                # 100の倍数に丸める（麻雀の点数は通常100点単位）
                score = (score // 100) * 100
                return score
            except ValueError:
                pass

        return None

    def _calculate_confidence(self, ocr_data: dict) -> float:
        """OCR結果の信頼度を計算"""
        confidences = []

        for _i, conf in enumerate(ocr_data["conf"]):
            if conf > 0:  # -1は無効な値
                confidences.append(conf / 100.0)

        if confidences:
            return np.mean(confidences)
        return 0.0

    def _detect_by_template(self, region: np.ndarray) -> tuple[int | None, float]:
        """
        テンプレートマッチングによる数字検出
        （OCRのフォールバック）
        """
        # TODO: 事前に準備した数字テンプレートとマッチング
        # 現在は簡易的な実装

        # 領域の平均輝度から推定（デモ用）
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        mean_val = np.mean(gray)

        # 仮の点数（実際の実装では適切なテンプレートマッチングを行う）
        if mean_val > 200 or mean_val > 150:  # 明るい = 高得点
            return 25000, 0.3
        else:
            return 25000, 0.3

    def _is_valid_score(self, score: int) -> bool:
        """点数の妥当性チェック"""
        return (
            self.valid_score_range[0] <= score <= self.valid_score_range[1]
            and score % 100 == 0  # 100点単位
        )

    def _detect_score_changes(self, result: ScoreReadingResult):
        """点数変化を検出してログ出力"""
        if not result.scores:
            return

        current_scores = {s.player_position: s.score for s in result.scores}

        if self.prev_scores is not None:
            for position, score in current_scores.items():
                if position in self.prev_scores:
                    diff = score - self.prev_scores[position]
                    if abs(diff) >= 100:  # 100点以上の変化
                        self.logger.info(
                            f"点数変化検出: {position} "
                            f"{self.prev_scores[position]:,} → {score:,} "
                            f"({diff:+,})"
                        )

        self.prev_scores = current_scores

    def calibrate_regions(self, frame: np.ndarray, known_scores: dict[str, int]) -> dict[str, dict]:
        """
        既知の点数を使って領域を自動調整

        Args:
            frame: キャリブレーション用フレーム
            known_scores: 既知の点数 {"east": 25000, ...}

        Returns:
            調整された領域設定
        """
        height, width = frame.shape[:2]
        calibrated_regions = {}

        # 各位置について最適な領域を探索
        for position, expected_score in known_scores.items():
            best_region = None
            best_confidence = 0.0

            # 現在の設定を基準に周辺を探索
            base_region = self.score_regions.get(position, {})
            if not base_region:
                continue

            # グリッドサーチ
            for dx in [-0.02, 0, 0.02]:
                for dy in [-0.02, 0, 0.02]:
                    test_region = {
                        "x": base_region["x"] + dx,
                        "y": base_region["y"] + dy,
                        "w": base_region["w"],
                        "h": base_region["h"],
                    }

                    # 領域を切り出してテスト
                    x1 = int(test_region["x"] * width)
                    y1 = int(test_region["y"] * height)
                    x2 = x1 + int(test_region["w"] * width)
                    y2 = y1 + int(test_region["h"] * height)

                    if 0 <= x1 < x2 < width and 0 <= y1 < y2 < height:
                        region_img = frame[y1:y2, x1:x2]
                        score, confidence = self._read_score_from_region(region_img)

                        if score == expected_score and confidence > best_confidence:
                            best_region = test_region
                            best_confidence = confidence

            if best_region:
                calibrated_regions[position] = best_region
                self.logger.info(f"{position}の領域を調整: 信頼度 {best_confidence:.2f}")

        return calibrated_regions
