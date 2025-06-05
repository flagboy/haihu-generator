"""
統合された動画処理モジュール
hand_labeling_systemとhand_training_systemの機能を統合
"""

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


class EnhancedVideoProcessor:
    """統合された動画処理クラス"""

    def __init__(self, video_path: str, cache_dir: str = "data/training/frames"):
        """
        初期化

        Args:
            video_path: 動画ファイルのパス
            cache_dir: フレームキャッシュディレクトリ
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 動画情報を取得
        self.cap = cv2.VideoCapture(str(self.video_path))
        self._load_video_info()

        # メタデータファイル
        self.metadata_file = self.cache_dir / f"{self.video_hash}_metadata.json"
        self._load_or_create_metadata()

        # 進捗管理
        self.progress = {
            "total_frames": self.frame_count,
            "processed_frames": len(self.metadata.get("processed_frames", [])),
            "cached_frames": len(self.metadata.get("extracted_frames", {})),
        }

        logger.info(f"動画処理を初期化: {self.video_path.name}")
        logger.info(
            f"フレーム数: {self.frame_count}, FPS: {self.fps}, 解像度: {self.width}x{self.height}"
        )

    def _load_video_info(self):
        """動画の基本情報を読み込み"""
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # 動画のハッシュを計算（最初の1MBから）
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        hash_md5 = hashlib.md5()
        for _ in range(30):  # 最初の30フレーム
            ret, frame = self.cap.read()
            if not ret:
                break
            hash_md5.update(frame.tobytes())
        self.video_hash = hash_md5.hexdigest()[:16]

        # 読み取り位置をリセット
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _load_or_create_metadata(self):
        """メタデータの読み込みまたは作成"""
        if self.metadata_file.exists():
            with open(self.metadata_file, encoding="utf-8") as f:
                self.metadata = json.load(f)
                logger.info("既存のメタデータを読み込みました")
        else:
            self.metadata = {
                "video_info": {
                    "path": str(self.video_path),
                    "hash": self.video_hash,
                    "fps": self.fps,
                    "frame_count": self.frame_count,
                    "width": self.width,
                    "height": self.height,
                    "duration": self.duration,
                    "created_at": datetime.now().isoformat(),
                },
                "extracted_frames": {},
                "hand_regions": {},
                "processed_frames": [],
                "scene_changes": [],
            }
            self._save_metadata()
            logger.info("新しいメタデータを作成しました")

    def _save_metadata(self):
        """メタデータを保存"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def get_frame(self, frame_number: int, use_cache: bool = True) -> np.ndarray | None:
        """
        指定したフレーム番号の画像を取得

        Args:
            frame_number: フレーム番号
            use_cache: キャッシュを使用するか

        Returns:
            フレーム画像（BGR形式）
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            logger.warning(f"無効なフレーム番号: {frame_number}")
            return None

        # キャッシュをチェック
        if use_cache:
            cache_path = self._get_cache_path(frame_number)
            if cache_path.exists():
                logger.debug(f"キャッシュからフレーム読み込み: {frame_number}")
                return cv2.imread(str(cache_path))

        # 動画から読み込み
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            # キャッシュに保存
            if use_cache:
                self._save_frame_cache(frame_number, frame)
            return frame

        logger.error(f"フレームの読み込みに失敗: {frame_number}")
        return None

    def extract_frames(
        self,
        interval: float = 1.0,
        start_time: float = 0.0,
        end_time: float | None = None,
        progress_callback: Callable | None = None,
    ) -> list[dict]:
        """
        指定間隔でフレームを抽出（frame_extractor.pyから統合）

        Args:
            interval: 抽出間隔（秒）
            start_time: 開始時間（秒）
            end_time: 終了時間（秒）
            progress_callback: 進捗コールバック関数

        Returns:
            抽出されたフレーム情報のリスト
        """
        if end_time is None:
            end_time = self.duration

        extracted_frames = []

        # 開始・終了フレーム番号を計算
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        interval_frames = int(interval * self.fps)

        # プログレスバーを表示
        total_frames = (end_frame - start_frame) // interval_frames

        with tqdm(total=total_frames, desc="フレーム抽出中") as pbar:
            for frame_num in range(start_frame, end_frame, interval_frames):
                frame = self.get_frame(frame_num)
                if frame is not None:
                    frame_info = {
                        "frame_number": frame_num,
                        "timestamp": frame_num / self.fps,
                        "cache_path": str(self._get_cache_path(frame_num)),
                    }
                    extracted_frames.append(frame_info)

                    # メタデータ更新
                    self.metadata["extracted_frames"][str(frame_num)] = frame_info

                    # 進捗更新
                    pbar.update(1)
                    if progress_callback:
                        progress_callback(len(extracted_frames) / total_frames)

        self._save_metadata()
        logger.info(f"{len(extracted_frames)}フレームを抽出しました")
        return extracted_frames

    def detect_scene_changes(self, threshold: float = 30.0) -> list[int]:
        """
        シーン変化を検出（video_processor.pyから移植）

        Args:
            threshold: 変化検出の閾値

        Returns:
            シーン変化のフレーム番号リスト
        """
        logger.info("シーン変化検出を開始")
        scene_changes = []
        prev_frame = None

        with tqdm(total=self.frame_count, desc="シーン変化検出中") as pbar:
            for i in range(0, self.frame_count, int(self.fps)):  # 1秒ごとにチェック
                frame = self.get_frame(i, use_cache=False)
                if frame is None:
                    continue

                if prev_frame is not None:
                    # フレーム差分を計算
                    diff = cv2.absdiff(prev_frame, frame)
                    diff_mean = np.mean(diff)

                    if diff_mean > threshold:
                        scene_changes.append(i)
                        logger.debug(f"シーン変化検出: フレーム {i} (差分: {diff_mean:.2f})")

                prev_frame = frame
                pbar.update(int(self.fps))

        # メタデータに保存
        self.metadata["scene_changes"] = scene_changes
        self._save_metadata()

        logger.info(f"{len(scene_changes)}個のシーン変化を検出しました")
        return scene_changes

    def detect_hand_changes(
        self, hand_regions: dict[str, dict[str, int]], threshold: float = 20.0
    ) -> list[int]:
        """
        手牌変化を検出（frame_extractor.pyから統合）

        Args:
            hand_regions: 手牌領域の定義
            threshold: 変化検出の閾値

        Returns:
            手牌変化のフレーム番号リスト
        """
        logger.info("手牌変化検出を開始")
        hand_changes = []
        prev_hands = {}

        with tqdm(total=self.frame_count, desc="手牌変化検出中") as pbar:
            for i in range(0, self.frame_count, int(self.fps / 2)):  # 0.5秒ごとにチェック
                frame = self.get_frame(i)
                if frame is None:
                    continue

                current_hands = {}
                for player, region in hand_regions.items():
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    hand_roi = frame[y : y + h, x : x + w]
                    current_hands[player] = hand_roi

                if prev_hands:
                    for player in hand_regions:
                        if player in prev_hands:
                            diff = cv2.absdiff(prev_hands[player], current_hands[player])
                            diff_mean = np.mean(diff)

                            if diff_mean > threshold:
                                hand_changes.append(i)
                                logger.debug(f"{player}の手牌変化検出: フレーム {i}")
                                break

                prev_hands = current_hands
                pbar.update(int(self.fps / 2))

        logger.info(f"{len(hand_changes)}個の手牌変化を検出しました")
        return hand_changes

    def _get_cache_path(self, frame_number: int) -> Path:
        """キャッシュパスを取得"""
        return self.cache_dir / f"{self.video_hash}_frame_{frame_number:06d}.jpg"

    def _save_frame_cache(self, frame_number: int, frame: np.ndarray):
        """フレームをキャッシュに保存"""
        cache_path = self._get_cache_path(frame_number)
        cv2.imwrite(str(cache_path), frame)
        logger.debug(f"フレームをキャッシュに保存: {frame_number}")

    def get_progress(self) -> dict[str, int]:
        """処理進捗を取得"""
        self.progress["processed_frames"] = len(self.metadata.get("processed_frames", []))
        self.progress["cached_frames"] = len(self.metadata.get("extracted_frames", {}))
        return self.progress

    def seek_to_time(self, time_seconds: float) -> bool:
        """
        指定時間へシーク

        Args:
            time_seconds: シーク先の時間（秒）

        Returns:
            成功した場合True
        """
        frame_number = int(time_seconds * self.fps)
        if 0 <= frame_number < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            return True
        return False

    def get_current_position(self) -> tuple[int, float]:
        """
        現在の再生位置を取得

        Returns:
            (フレーム番号, 時間（秒）)
        """
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_seconds = frame_number / self.fps if self.fps > 0 else 0
        return frame_number, time_seconds

    def generate_thumbnail(
        self, frame_number: int, size: tuple[int, int] = (320, 240)
    ) -> np.ndarray | None:
        """
        サムネイル画像を生成

        Args:
            frame_number: フレーム番号
            size: サムネイルサイズ

        Returns:
            サムネイル画像
        """
        frame = self.get_frame(frame_number)
        if frame is not None:
            return cv2.resize(frame, size)
        return None

    def cleanup_cache(self, keep_recent: int = 1000):
        """
        古いキャッシュをクリーンアップ

        Args:
            keep_recent: 保持する最近のフレーム数
        """
        cache_files = list(self.cache_dir.glob(f"{self.video_hash}_frame_*.jpg"))
        if len(cache_files) > keep_recent:
            # 古いファイルから削除
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for cache_file in cache_files[:-keep_recent]:
                cache_file.unlink()
                logger.debug(f"キャッシュを削除: {cache_file.name}")

    def __del__(self):
        """クリーンアップ"""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            logger.debug("動画キャプチャをリリースしました")
