"""
動画フレーム抽出モジュール
"""

import json
from pathlib import Path

import cv2
import numpy as np


class FrameExtractor:
    """動画からフレームを抽出するクラス"""

    def __init__(self, video_path: str, output_dir: str):
        """
        初期化

        Args:
            video_path: 動画ファイルのパス
            output_dir: フレーム出力ディレクトリ
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 動画情報を取得
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # メタデータファイル
        self.metadata_file = self.output_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """既存のメタデータを読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "video_path": str(self.video_path),
                "fps": self.fps,
                "frame_count": self.frame_count,
                "width": self.width,
                "height": self.height,
                "extracted_frames": {},
            }
            self._save_metadata()

    def _save_metadata(self):
        """メタデータを保存"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def extract_frame(self, frame_number: int) -> np.ndarray | None:
        """
        指定したフレーム番号の画像を抽出

        Args:
            frame_number: フレーム番号

        Returns:
            フレーム画像（numpy配列）
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            return frame
        return None

    def extract_frames_at_interval(
        self, interval_seconds: float = 1.0, start_time: float = 0.0, end_time: float | None = None
    ) -> list[tuple[int, str]]:
        """
        指定間隔でフレームを抽出

        Args:
            interval_seconds: 抽出間隔（秒）
            start_time: 開始時間（秒）
            end_time: 終了時間（秒）

        Returns:
            (フレーム番号、保存パス)のリスト
        """
        if end_time is None:
            end_time = self.frame_count / self.fps

        extracted_frames = []

        current_time = start_time
        while current_time <= end_time:
            frame_number = int(current_time * self.fps)
            if frame_number >= self.frame_count:
                break

            # フレーム抽出
            frame = self.extract_frame(frame_number)
            if frame is not None:
                # フレームを保存
                frame_path = self.save_frame(frame, frame_number)
                extracted_frames.append((frame_number, str(frame_path)))

            current_time += interval_seconds

        return extracted_frames

    def save_frame(self, frame: np.ndarray, frame_number: int) -> Path:
        """
        フレームを保存

        Args:
            frame: フレーム画像
            frame_number: フレーム番号

        Returns:
            保存したファイルのパス
        """
        # ファイル名を生成
        timestamp = frame_number / self.fps
        filename = f"frame_{frame_number:08d}_t{timestamp:.2f}s.jpg"
        filepath = self.output_dir / filename

        # 保存
        cv2.imwrite(str(filepath), frame)

        # メタデータに記録
        self.metadata["extracted_frames"][str(frame_number)] = {
            "timestamp": timestamp,
            "filename": filename,
            "path": str(filepath),
        }
        self._save_metadata()

        return filepath

    def detect_hand_changes(
        self, hand_regions: list[tuple[int, int, int, int]], threshold: float = 0.05
    ) -> list[int]:
        """
        手牌領域の変化を検出

        Args:
            hand_regions: 手牌領域のリスト [(x, y, w, h), ...]
            threshold: 変化検出の閾値

        Returns:
            変化が検出されたフレーム番号のリスト
        """
        changed_frames = []
        prev_hands = [None] * len(hand_regions)

        # 1秒ごとにチェック
        for frame_num in range(0, self.frame_count, int(self.fps)):
            frame = self.extract_frame(frame_num)
            if frame is None:
                continue

            # 各手牌領域をチェック
            for i, (x, y, w, h) in enumerate(hand_regions):
                # 領域を切り出し
                hand_area = frame[y : y + h, x : x + w]

                # グレースケール変換
                hand_gray = cv2.cvtColor(hand_area, cv2.COLOR_BGR2GRAY)

                if prev_hands[i] is not None:
                    # 前フレームとの差分を計算
                    diff = cv2.absdiff(prev_hands[i], hand_gray)
                    change_ratio = np.sum(diff > 30) / diff.size

                    if change_ratio > threshold:
                        changed_frames.append(frame_num)
                        break

                prev_hands[i] = hand_gray

        return changed_frames

    def get_frame_info(self, frame_number: int) -> dict | None:
        """
        フレーム情報を取得

        Args:
            frame_number: フレーム番号

        Returns:
            フレーム情報の辞書
        """
        if str(frame_number) in self.metadata["extracted_frames"]:
            return self.metadata["extracted_frames"][str(frame_number)]
        return None

    def __del__(self):
        """クリーンアップ"""
        if hasattr(self, "cap"):
            self.cap.release()
