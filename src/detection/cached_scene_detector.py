"""
キャッシュ機能付きシーン検出モジュール

パフォーマンス最適化のためのキャッシュ機能を追加
"""

import hashlib

import numpy as np

from .scene_detector import SceneDetectionResult, SceneDetector


class CachedSceneDetector(SceneDetector):
    """キャッシュ機能付きシーン検出クラス"""

    def __init__(self, config: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        super().__init__(config)

        # キャッシュ設定
        cache_config = self.config.get("cache", {})
        self.enable_cache = cache_config.get("enabled", True)
        self.cache_size = cache_config.get("size", 100)

        # フレームハッシュのキャッシュ
        self._frame_hash_cache = {}

        self.logger.info(f"CachedSceneDetector初期化完了 (cache_size: {self.cache_size})")

    def detect_scene(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> SceneDetectionResult:
        """
        フレームからシーンを検出（キャッシュ付き）

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            timestamp: タイムスタンプ

        Returns:
            シーン検出結果
        """
        if not self.enable_cache:
            return super().detect_scene(frame, frame_number, timestamp)

        # フレームのハッシュを計算
        frame_hash = self._compute_frame_hash(frame)

        # キャッシュから結果を取得
        cached_result = self._get_cached_result(frame_hash, frame_number, timestamp)
        if cached_result is not None:
            self.logger.debug(f"キャッシュヒット: frame {frame_number}")
            return cached_result

        # キャッシュミスの場合は通常の検出を実行
        result = super().detect_scene(frame, frame_number, timestamp)

        # 結果をキャッシュに保存
        self._cache_result(frame_hash, result)

        return result

    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """フレームのハッシュを計算"""
        # 高速化のため、フレームをダウンサンプリング
        small_frame = frame[::8, ::8, :]  # 8x8ピクセルごとにサンプリング

        # ハッシュ計算
        frame_bytes = small_frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()

    def _get_cached_result(
        self, frame_hash: str, frame_number: int, timestamp: float
    ) -> SceneDetectionResult | None:
        """キャッシュから結果を取得"""
        if frame_hash in self._frame_hash_cache:
            cached_data = self._frame_hash_cache[frame_hash]
            # フレーム番号とタイムスタンプを更新して返す
            return SceneDetectionResult(
                scene_type=cached_data["scene_type"],
                confidence=cached_data["confidence"],
                frame_number=frame_number,
                timestamp=timestamp,
                metadata=cached_data["metadata"],
            )
        return None

    def _cache_result(self, frame_hash: str, result: SceneDetectionResult):
        """結果をキャッシュに保存"""
        # キャッシュサイズの制限
        if len(self._frame_hash_cache) >= self.cache_size:
            # 最も古いエントリを削除（簡易的なLRU）
            oldest_key = next(iter(self._frame_hash_cache))
            del self._frame_hash_cache[oldest_key]

        # 結果を保存
        self._frame_hash_cache[frame_hash] = {
            "scene_type": result.scene_type,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }

    def clear_cache(self):
        """キャッシュをクリア"""
        self._frame_hash_cache.clear()
        self.logger.info("キャッシュをクリアしました")
