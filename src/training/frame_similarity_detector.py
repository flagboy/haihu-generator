"""
フレーム類似度検出システム

教師データ作成済みのフレームと類似したフレームを検出し、
効率的にスキップするための機能を提供
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..utils.logger import LoggerMixin


@dataclass
class SimilarityResult:
    """類似度検出結果"""

    is_similar: bool
    similarity_score: float
    matched_frame_id: str | None
    similarity_type: str  # "identical", "very_similar", "similar", "different"
    metadata: dict[str, Any]


class FrameSimilarityDetector(LoggerMixin):
    """フレーム類似度検出クラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}

        # 閾値設定
        self.identical_threshold = self.config.get("identical_threshold", 0.99)
        self.very_similar_threshold = self.config.get("very_similar_threshold", 0.95)
        self.similar_threshold = self.config.get("similar_threshold", 0.85)

        # キャッシュ設定
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_size = self.config.get("cache_size", 1000)

        # キャッシュ
        self.hash_cache: dict[str, str] = {}  # frame_path -> hash
        self.feature_cache: dict[str, np.ndarray] = {}  # frame_path -> features
        self.similarity_cache: dict[tuple[str, str], float] = {}  # (frame1, frame2) -> score

        # 比較手法
        self.use_hash = self.config.get("use_hash", True)
        self.use_histogram = self.config.get("use_histogram", True)
        self.use_ssim = self.config.get("use_ssim", True)
        self.use_orb = self.config.get("use_orb", False)  # 計算コストが高い

        self.logger.info("FrameSimilarityDetector初期化完了")

    def check_similarity(
        self,
        current_frame: np.ndarray,
        reference_frames: list[dict[str, Any]],
        quick_check: bool = True,
    ) -> SimilarityResult:
        """
        現在のフレームと参照フレームの類似度をチェック

        Args:
            current_frame: 現在のフレーム
            reference_frames: 参照フレームのリスト
            quick_check: 高速チェックモード

        Returns:
            類似度検出結果
        """
        if not reference_frames:
            return SimilarityResult(
                is_similar=False,
                similarity_score=0.0,
                matched_frame_id=None,
                similarity_type="different",
                metadata={},
            )

        # 高速チェック（ハッシュベース）
        if quick_check and self.use_hash:
            current_hash = self._compute_hash(current_frame)

            for ref_frame in reference_frames:
                ref_hash = ref_frame.get("hash") or self._get_frame_hash(ref_frame["path"])

                if current_hash == ref_hash:
                    return SimilarityResult(
                        is_similar=True,
                        similarity_score=1.0,
                        matched_frame_id=ref_frame["id"],
                        similarity_type="identical",
                        metadata={"method": "hash"},
                    )

        # 詳細な類似度チェック
        best_score = 0.0
        best_match = None
        similarity_scores = {}

        for ref_frame in reference_frames:
            score = self._compute_similarity(current_frame, ref_frame)
            similarity_scores[ref_frame["id"]] = score

            if score > best_score:
                best_score = score
                best_match = ref_frame

        # 類似度判定
        if best_score >= self.identical_threshold:
            similarity_type = "identical"
            is_similar = True
        elif best_score >= self.very_similar_threshold:
            similarity_type = "very_similar"
            is_similar = True
        elif best_score >= self.similar_threshold:
            similarity_type = "similar"
            is_similar = True
        else:
            similarity_type = "different"
            is_similar = False

        return SimilarityResult(
            is_similar=is_similar,
            similarity_score=best_score,
            matched_frame_id=best_match["id"] if best_match else None,
            similarity_type=similarity_type,
            metadata={"all_scores": similarity_scores, "threshold_used": self.similar_threshold},
        )

    def _compute_hash(self, frame: np.ndarray) -> str:
        """フレームのハッシュを計算"""
        # 画像を縮小してハッシュ計算を高速化
        small_frame = cv2.resize(frame, (64, 64))

        # グレースケール変換
        if len(small_frame.shape) == 3:
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # ハッシュ計算
        return hashlib.md5(small_frame.tobytes()).hexdigest()

    def _get_frame_hash(self, frame_path: str) -> str:
        """フレームパスからハッシュを取得（キャッシュ付き）"""
        if self.enable_cache and frame_path in self.hash_cache:
            return self.hash_cache[frame_path]

        # フレーム読み込み
        frame = cv2.imread(frame_path)
        if frame is None:
            return ""

        hash_value = self._compute_hash(frame)

        # キャッシュに保存
        if self.enable_cache:
            self._update_cache(self.hash_cache, frame_path, hash_value)

        return hash_value

    def _compute_similarity(self, frame1: np.ndarray, ref_frame: dict[str, Any]) -> float:
        """2つのフレーム間の類似度を計算"""
        scores = []
        weights = []

        # 参照フレームを読み込み
        frame2 = cv2.imread(ref_frame["path"])
        if frame2 is None:
            return 0.0

        # サイズを揃える
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # ヒストグラム比較
        if self.use_histogram:
            hist_score = self._compare_histograms(frame1, frame2)
            scores.append(hist_score)
            weights.append(0.3)

        # SSIM（構造的類似性）
        if self.use_ssim:
            ssim_score = self._compute_ssim(frame1, frame2)
            scores.append(ssim_score)
            weights.append(0.5)

        # ORB特徴量比較（オプション）
        if self.use_orb:
            orb_score = self._compare_orb_features(frame1, frame2)
            scores.append(orb_score)
            weights.append(0.2)

        # 重み付き平均
        if scores:
            total_weight = sum(weights)
            weighted_score = (
                sum(s * w for s, w in zip(scores, weights, strict=False)) / total_weight
            )
            return weighted_score

        return 0.0

    def _compare_histograms(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """ヒストグラム比較による類似度計算"""
        # HSV色空間に変換
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # ヒストグラム計算
        hist_size = [50, 60]
        hist_range = [0, 180, 0, 256]

        hist1 = cv2.calcHist([hsv1], [0, 1], None, hist_size, hist_range)
        hist2 = cv2.calcHist([hsv2], [0, 1], None, hist_size, hist_range)

        # 正規化
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # 相関係数を計算
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # 0-1の範囲に正規化
        return (correlation + 1) / 2

    def _compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """SSIM（構造的類似性指標）を計算"""
        # グレースケールに変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # SSIMを計算
        score, _ = ssim(gray1, gray2, full=True)

        return score

    def _compare_orb_features(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """ORB特徴量による類似度計算"""
        # ORB検出器
        orb = cv2.ORB_create()

        # 特徴点検出
        kp1, des1 = orb.detectAndCompute(frame1, None)
        kp2, des2 = orb.detectAndCompute(frame2, None)

        if des1 is None or des2 is None:
            return 0.0

        # マッチング
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)

        # 良いマッチの割合を計算
        if len(kp1) > 0 and len(kp2) > 0:
            match_ratio = len(matches) / min(len(kp1), len(kp2))
            return min(match_ratio, 1.0)

        return 0.0

    def _update_cache(self, cache: dict, key: Any, value: Any):
        """キャッシュを更新（サイズ制限付き）"""
        if len(cache) >= self.cache_size:
            # 最も古いエントリを削除（簡易的なLRU）
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        cache[key] = value

    def find_similar_frames_in_sequence(
        self, frames: list[np.ndarray], min_difference_threshold: float = 0.1
    ) -> list[int]:
        """
        フレームシーケンス内で類似フレームを検出

        Args:
            frames: フレームのリスト
            min_difference_threshold: 最小差分閾値

        Returns:
            スキップ可能なフレームインデックスのリスト
        """
        skip_indices = []

        if len(frames) < 2:
            return skip_indices

        # 前のフレームと比較
        for i in range(1, len(frames)):
            similarity = self._compute_frame_difference(frames[i - 1], frames[i])

            if similarity < min_difference_threshold:
                skip_indices.append(i)
                self.logger.debug(f"フレーム {i} は前フレームと類似 (差分: {similarity:.3f})")

        return skip_indices

    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """フレーム間の差分を計算"""
        # グレースケール変換
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 差分計算
        diff = cv2.absdiff(gray1, gray2)

        # 平均差分を正規化
        mean_diff = np.mean(diff) / 255.0

        return mean_diff

    def get_key_frames(
        self, frames: list[np.ndarray], max_frames: int = 100, diversity_weight: float = 0.7
    ) -> list[int]:
        """
        多様性を考慮してキーフレームを選択

        Args:
            frames: フレームのリスト
            max_frames: 最大フレーム数
            diversity_weight: 多様性の重み

        Returns:
            選択されたフレームインデックスのリスト
        """
        if len(frames) <= max_frames:
            return list(range(len(frames)))

        # 初期選択（等間隔）
        step = len(frames) / max_frames
        selected_indices = [int(i * step) for i in range(max_frames)]

        # 多様性最適化
        if diversity_weight > 0:
            selected_indices = self._optimize_frame_selection(
                frames, selected_indices, diversity_weight
            )

        return sorted(selected_indices)

    def _optimize_frame_selection(
        self, frames: list[np.ndarray], initial_indices: list[int], diversity_weight: float
    ) -> list[int]:
        """フレーム選択を多様性の観点から最適化"""
        # 簡易的な実装：類似度が高いフレームを置き換え
        optimized_indices = initial_indices.copy()

        for _ in range(5):  # 5回の最適化イテレーション
            improved = False

            for i in range(1, len(optimized_indices) - 1):
                # 前後のフレームとの類似度を計算
                prev_sim = self._compute_frame_difference(
                    frames[optimized_indices[i - 1]], frames[optimized_indices[i]]
                )
                next_sim = self._compute_frame_difference(
                    frames[optimized_indices[i]], frames[optimized_indices[i + 1]]
                )

                # 類似度が高い場合、別のフレームを探す
                if prev_sim < 0.1 or next_sim < 0.1:
                    # 近隣のフレームから最も異なるものを選択
                    start = optimized_indices[i - 1]
                    end = optimized_indices[i + 1]

                    best_idx = optimized_indices[i]
                    best_diff = 0

                    for j in range(start + 1, end):
                        if j not in optimized_indices:
                            diff1 = self._compute_frame_difference(frames[start], frames[j])
                            diff2 = self._compute_frame_difference(frames[j], frames[end])
                            total_diff = diff1 + diff2

                            if total_diff > best_diff:
                                best_diff = total_diff
                                best_idx = j
                                improved = True

                    optimized_indices[i] = best_idx

            if not improved:
                break

        return optimized_indices
