"""
最適化されたデータローダー
効率的なデータ読み込みとプリフェッチ機能
"""

import asyncio
import contextlib
import queue
import threading
from collections.abc import AsyncGenerator, Callable, Generator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoFrameDataset(Dataset):
    """動画フレームデータセット（メモリ効率的）"""

    def __init__(
        self,
        video_path: str | Path,
        frame_indices: list[int] | None = None,
        transform: Callable | None = None,
        cache_size: int = 100,
    ):
        """
        Args:
            video_path: 動画ファイルのパス
            frame_indices: 読み込むフレームインデックス（Noneの場合は全フレーム）
            transform: フレーム変換関数
            cache_size: キャッシュサイズ
        """
        self.video_path = Path(video_path)
        self.transform = transform

        # 動画情報を取得
        self._init_video_info()

        # フレームインデックス
        if frame_indices is None:
            self.frame_indices = list(range(self.total_frames))
        else:
            self.frame_indices = frame_indices

        # LRUキャッシュ
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self.cache_size = cache_size

    def _init_video_info(self) -> None:
        """動画情報を初期化"""
        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray]:
        """フレームを取得"""
        frame_idx = self.frame_indices[idx]

        # キャッシュチェック
        if frame_idx in self._cache:
            # キャッシュヒット
            self._update_cache_order(frame_idx)
            frame = self._cache[frame_idx]
        else:
            # キャッシュミス - フレームを読み込み
            frame = self._read_frame(frame_idx)
            self._add_to_cache(frame_idx, frame)

        # 変換を適用
        if self.transform:
            frame = self.transform(frame)

        return frame_idx, frame

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """フレームを読み込み"""
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.warning(f"フレーム {frame_idx} の読み込みに失敗")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return frame

    def _add_to_cache(self, frame_idx: int, frame: np.ndarray) -> None:
        """キャッシュに追加"""
        if len(self._cache) >= self.cache_size:
            # 最も古いエントリを削除
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]

        self._cache[frame_idx] = frame
        self._cache_order.append(frame_idx)

    def _update_cache_order(self, frame_idx: int) -> None:
        """キャッシュ順序を更新"""
        self._cache_order.remove(frame_idx)
        self._cache_order.append(frame_idx)


class StreamingVideoLoader:
    """ストリーミング動画ローダー（リアルタイム処理用）"""

    def __init__(
        self,
        video_path: str | Path,
        batch_size: int = 32,
        buffer_size: int = 100,
        num_workers: int = 2,
    ):
        """
        Args:
            video_path: 動画ファイルのパス
            batch_size: バッチサイズ
            buffer_size: バッファサイズ
            num_workers: ワーカースレッド数
        """
        self.video_path = Path(video_path)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        # バッファキュー
        self.frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.batch_queue: queue.Queue = queue.Queue(maxsize=10)

        # 制御フラグ
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []

    def start(self) -> None:
        """ローダーを開始"""
        self._stop_event.clear()

        # フレーム読み込みワーカー
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._frame_reader_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)

        # バッチ作成ワーカー
        batch_worker = threading.Thread(target=self._batch_creator_worker)
        batch_worker.daemon = True
        batch_worker.start()
        self._workers.append(batch_worker)

        logger.info(f"StreamingVideoLoader開始: workers={self.num_workers}")

    def stop(self) -> None:
        """ローダーを停止"""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._workers.clear()
        logger.info("StreamingVideoLoader停止")

    def _frame_reader_worker(self, worker_id: int) -> None:
        """フレーム読み込みワーカー"""
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ワーカーごとに読み込み範囲を分割
        frames_per_worker = total_frames // self.num_workers
        start_frame = worker_id * frames_per_worker
        end_frame = (
            start_frame + frames_per_worker if worker_id < self.num_workers - 1 else total_frames
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while not self._stop_event.is_set() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                self.frame_queue.put((frame_idx, frame), timeout=1.0)
                frame_idx += 1
            except queue.Full:
                continue

        cap.release()

    def _batch_creator_worker(self) -> None:
        """バッチ作成ワーカー"""
        batch_frames = []
        batch_indices = []

        while not self._stop_event.is_set():
            try:
                # フレームを取得
                frame_idx, frame = self.frame_queue.get(timeout=1.0)
                batch_frames.append(frame)
                batch_indices.append(frame_idx)

                # バッチサイズに達したら送信
                if len(batch_frames) >= self.batch_size:
                    self.batch_queue.put((batch_indices.copy(), batch_frames.copy()))
                    batch_frames.clear()
                    batch_indices.clear()

            except queue.Empty:
                # 残りのフレームを送信
                if batch_frames:
                    self.batch_queue.put((batch_indices.copy(), batch_frames.copy()))
                    batch_frames.clear()
                    batch_indices.clear()

    def get_batches(self) -> Generator[tuple[list[int], list[np.ndarray]], None, None]:
        """バッチを取得するジェネレータ"""
        while not self._stop_event.is_set():
            try:
                batch = self.batch_queue.get(timeout=1.0)
                yield batch
            except queue.Empty:
                if self.frame_queue.empty() and self.batch_queue.empty():
                    break


class OptimizedVideoDataLoader:
    """最適化された動画データローダー（PyTorch DataLoader統合）"""

    def __init__(
        self,
        video_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        device: str | None = None,
    ):
        """
        Args:
            video_path: 動画ファイルのパス
            batch_size: バッチサイズ
            num_workers: DataLoaderワーカー数
            prefetch_factor: プリフェッチ係数
            pin_memory: ピンメモリを使用するか
            device: デバイス
        """
        self.video_path = Path(video_path)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # データセット作成
        self.dataset = VideoFrameDataset(video_path)

        # DataLoader設定
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """バッチイテレータ"""
        for batch_indices, batch_frames in self.dataloader:
            # NumPy配列をTensorに変換してデバイスに転送
            if isinstance(batch_frames, np.ndarray):
                batch_frames = torch.from_numpy(batch_frames)

            # デバイスに転送（非同期）
            batch_frames = batch_frames.to(self.device, non_blocking=True)
            batch_indices = batch_indices.to(self.device, non_blocking=True)

            yield batch_indices, batch_frames

    def __len__(self) -> int:
        return len(self.dataloader)


class AsyncVideoLoader:
    """非同期動画ローダー（asyncio対応）"""

    def __init__(self, video_path: str | Path, batch_size: int = 32):
        """
        Args:
            video_path: 動画ファイルのパス
            batch_size: バッチサイズ
        """
        self.video_path = Path(video_path)
        self.batch_size = batch_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._reader_task = None

    async def start(self) -> None:
        """非同期読み込みを開始"""
        self._reader_task = asyncio.create_task(self._read_frames())

    async def stop(self) -> None:
        """非同期読み込みを停止"""
        if self._reader_task:
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

    async def _read_frames(self) -> None:
        """フレームを非同期で読み込み"""
        cap = cv2.VideoCapture(str(self.video_path))
        batch_frames = []
        batch_indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                # 最後のバッチを送信
                if batch_frames:
                    await self._queue.put((batch_indices, batch_frames))
                break

            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            frame_idx += 1

            if len(batch_frames) >= self.batch_size:
                await self._queue.put((batch_indices.copy(), batch_frames.copy()))
                batch_frames.clear()
                batch_indices.clear()

            # CPUに処理時間を譲る
            await asyncio.sleep(0)

        cap.release()

    async def get_batches(self) -> AsyncGenerator[tuple[list[int], list[np.ndarray]], None]:
        """バッチを非同期で取得"""
        while True:
            try:
                batch = await asyncio.wait_for(self._queue.get(), timeout=5.0)
                yield batch
            except TimeoutError:
                if self._queue.empty():
                    break
