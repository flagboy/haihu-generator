"""
動画処理モジュール
動画からフレームを抽出し、効率的に処理する
"""

import hashlib
import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class VideoProcessor:
    """動画処理クラス"""

    def __init__(self, video_path: str, cache_dir: str = "data/frames"):
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

    def _load_or_create_metadata(self):
        """メタデータの読み込みまたは作成"""
        if self.metadata_file.exists():
            with open(self.metadata_file, encoding="utf-8") as f:
                self.metadata = json.load(f)
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
            }
            self._save_metadata()

    def _save_metadata(self):
        """メタデータを保存"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def get_frame(self, frame_number: int) -> np.ndarray | None:
        """
        指定したフレーム番号の画像を取得

        Args:
            frame_number: フレーム番号

        Returns:
            フレーム画像（BGR形式）
        """
        # キャッシュをチェック
        cache_path = self._get_frame_cache_path(frame_number)
        if cache_path.exists():
            return cv2.imread(str(cache_path))

        # 動画から読み込み
        if 0 <= frame_number < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # キャッシュに保存
                cv2.imwrite(str(cache_path), frame)
                return frame
        return None

    def _get_frame_cache_path(self, frame_number: int) -> Path:
        """フレームキャッシュのパスを生成"""
        return self.cache_dir / f"{self.video_hash}_frame_{frame_number:08d}.jpg"

    def extract_frames_interval(
        self, interval_seconds: float = 1.0, start_time: float = 0.0, end_time: float | None = None
    ) -> list[int]:
        """
        指定間隔でフレームを抽出

        Args:
            interval_seconds: 抽出間隔（秒）
            start_time: 開始時間（秒）
            end_time: 終了時間（秒）

        Returns:
            抽出したフレーム番号のリスト
        """
        if end_time is None:
            end_time = self.duration

        frame_numbers = []
        current_time = start_time

        while current_time <= end_time:
            frame_number = int(current_time * self.fps)
            if frame_number >= self.frame_count:
                break

            frame_numbers.append(frame_number)
            current_time += interval_seconds

        # メタデータに記録
        self.metadata["extracted_frames"] = {
            str(fn): {"timestamp": fn / self.fps, "extracted_at": datetime.now().isoformat()}
            for fn in frame_numbers
        }
        self._save_metadata()

        return frame_numbers

    def detect_scene_changes(self, threshold: float = 30.0) -> list[int]:
        """
        シーン変化を検出（手牌の変化など）

        Args:
            threshold: 変化検出の閾値

        Returns:
            変化が検出されたフレーム番号のリスト
        """
        changes = []
        prev_frame = None

        # 1秒ごとにチェック
        for i in range(0, self.frame_count, int(self.fps)):
            frame = self.get_frame(i)
            if frame is None:
                continue

            if prev_frame is not None:
                # フレーム差分を計算
                diff = cv2.absdiff(prev_frame, frame)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                mean_diff = np.mean(diff_gray)

                if mean_diff > threshold:
                    changes.append(i)

            prev_frame = frame

        return changes

    def get_frame_iterator(
        self, frame_numbers: list[int] | None = None
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        フレームのイテレータを取得

        Args:
            frame_numbers: 処理するフレーム番号のリスト（Noneの場合は全フレーム）

        Yields:
            (フレーム番号, フレーム画像)のタプル
        """
        if frame_numbers is None:
            frame_numbers = range(self.frame_count)

        for frame_number in frame_numbers:
            frame = self.get_frame(frame_number)
            if frame is not None:
                yield frame_number, frame

    def set_hand_regions(self, regions: dict):
        """
        手牌領域を設定

        Args:
            regions: プレイヤーごとの手牌領域
                    {"player1": {"x": 100, "y": 200, "w": 300, "h": 100}, ...}
        """
        self.metadata["hand_regions"] = regions
        self._save_metadata()

    def get_hand_regions(self) -> dict:
        """設定された手牌領域を取得"""
        return self.metadata.get("hand_regions", {})

    def mark_frame_processed(self, frame_number: int):
        """フレームを処理済みとしてマーク"""
        if frame_number not in self.metadata["processed_frames"]:
            self.metadata["processed_frames"].append(frame_number)
            self._save_metadata()

    def get_progress(self) -> dict:
        """処理進捗を取得"""
        total_extracted = len(self.metadata.get("extracted_frames", {}))
        total_processed = len(self.metadata.get("processed_frames", []))

        return {
            "total_frames": self.frame_count,
            "extracted_frames": total_extracted,
            "processed_frames": total_processed,
            "progress_percentage": (total_processed / total_extracted * 100)
            if total_extracted > 0
            else 0,
        }

    def close(self):
        """リソースを解放"""
        if hasattr(self, "cap"):
            self.cap.release()

    def __del__(self):
        """デストラクタ"""
        self.close()
