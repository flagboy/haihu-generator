"""
VideoCaptureのリソース管理用ユーティリティ

OpenCVのVideoCaptureを適切に管理し、リソースリークを防ぐ
"""

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2

from .logger import get_logger


class VideoCaptureContext:
    """VideoCaptureのコンテキストマネージャー"""

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)
        self.cap: cv2.VideoCapture | None = None
        self.logger = get_logger(__name__)

    def __enter__(self) -> cv2.VideoCapture:
        """コンテキスト開始時の処理"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        return self.cap

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """コンテキスト終了時の処理"""
        if self.cap is not None:
            self.cap.release()
            self.logger.debug(f"Released VideoCapture for: {self.video_path}")


class VideoCaptureCache:
    """VideoCaptureのキャッシュ管理"""

    def __init__(self, max_size: int = 5):
        """
        初期化

        Args:
            max_size: キャッシュする最大数
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)

    def get(self, video_path: str | Path) -> cv2.VideoCapture:
        """
        VideoCaptureを取得（キャッシュから or 新規作成）

        Args:
            video_path: 動画ファイルパス

        Returns:
            VideoCapture オブジェクト
        """
        video_path = str(video_path)

        with self._lock:
            # キャッシュに存在する場合
            if video_path in self._cache:
                # 最近使用したものとして末尾に移動
                self._cache.move_to_end(video_path)
                self.logger.debug(f"Cache hit for: {video_path}")
                return self._cache[video_path]

            # 新規作成
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")

            # キャッシュに追加
            self._cache[video_path] = cap
            self.logger.debug(f"Created new VideoCapture for: {video_path}")

            # キャッシュサイズ制限をチェック
            if len(self._cache) > self.max_size:
                # 最も古いものを削除
                oldest_path, oldest_cap = self._cache.popitem(last=False)
                oldest_cap.release()
                self.logger.debug(f"Evicted from cache: {oldest_path}")

            return cap

    def release(self, video_path: str | Path) -> None:
        """
        特定のVideoCaptureを解放

        Args:
            video_path: 動画ファイルパス
        """
        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                cap = self._cache.pop(video_path)
                cap.release()
                self.logger.debug(f"Released VideoCapture for: {video_path}")

    def release_all(self) -> None:
        """すべてのVideoCaptureを解放"""
        with self._lock:
            for video_path, cap in self._cache.items():
                cap.release()
                self.logger.debug(f"Released VideoCapture for: {video_path}")
            self._cache.clear()

    def __del__(self) -> None:
        """デストラクタ"""
        self.release_all()


# グローバルキャッシュインスタンス
_global_cache = VideoCaptureCache(max_size=10)


def get_video_capture(video_path: str | Path, use_cache: bool = True) -> cv2.VideoCapture:
    """
    VideoCaptureを取得

    Args:
        video_path: 動画ファイルパス
        use_cache: キャッシュを使用するか

    Returns:
        VideoCapture オブジェクト
    """
    if use_cache:
        return _global_cache.get(video_path)
    else:
        return cv2.VideoCapture(str(video_path))


def release_video_capture(video_path: str | Path) -> None:
    """
    VideoCaptureを解放

    Args:
        video_path: 動画ファイルパス
    """
    _global_cache.release(video_path)


def release_all_captures() -> None:
    """すべてのVideoCaptureを解放"""
    _global_cache.release_all()
