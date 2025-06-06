"""
対局画面ラベリングセッション管理

効率的なラベリング作業のためのセッション管理
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ....utils.logger import LoggerMixin
from ..core.feature_extractor import FeatureExtractor
from ..core.game_scene_classifier import GameSceneClassifier


@dataclass
class FrameLabel:
    """フレームラベル"""

    video_id: str
    frame_number: int
    is_game_scene: bool
    confidence: float | None = None
    annotator: str = "manual"
    created_at: str | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class SceneLabelingSession(LoggerMixin):
    """対局画面ラベリングセッション"""

    def __init__(
        self,
        session_id: str,
        video_path: str,
        db_path: str = "data/training/game_scene_labels.db",
        classifier: GameSceneClassifier | None = None,
    ):
        """
        初期化

        Args:
            session_id: セッションID
            video_path: 動画ファイルパス
            db_path: データベースパス
            classifier: 対局画面分類器（自動推論用）
        """
        self.session_id = session_id
        self.video_path = video_path
        self.db_path = db_path
        self.classifier = classifier
        self.feature_extractor = FeatureExtractor()

        # 動画情報を取得
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # セッション情報
        self.current_frame = 0
        self.labels: dict[int, FrameLabel] = {}
        self.uncertainty_frames: list[int] = []

        # データベース初期化
        self._init_database()

        # 既存のラベルを読み込み
        self._load_existing_labels()

        self.logger.info(
            f"SceneLabelingSession初期化完了: "
            f"video={Path(video_path).name}, "
            f"frames={self.total_frames}"
        )

    def _init_database(self):
        """データベースを初期化"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ラベルテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_scene_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                is_game_scene BOOLEAN NOT NULL,
                confidence REAL,
                annotator TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id, frame_number)
            )
        """)

        # セグメントテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_scene_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                scene_type TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # セッションテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labeling_sessions (
                session_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                video_path TEXT NOT NULL,
                total_frames INTEGER NOT NULL,
                labeled_frames INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

        # セッション情報を保存
        self._save_session_info()

    def _save_session_info(self):
        """セッション情報を保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_id = Path(self.video_path).stem

        cursor.execute(
            """
            INSERT OR REPLACE INTO labeling_sessions
            (session_id, video_id, video_path, total_frames, labeled_frames, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                self.session_id,
                video_id,
                self.video_path,
                self.total_frames,
                len(self.labels),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def _load_existing_labels(self):
        """既存のラベルを読み込み"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_id = Path(self.video_path).stem

        cursor.execute(
            """
            SELECT frame_number, is_game_scene, confidence, annotator, created_at
            FROM game_scene_labels
            WHERE video_id = ?
        """,
            (video_id,),
        )

        for row in cursor.fetchall():
            frame_number, is_game_scene, confidence, annotator, created_at = row
            self.labels[frame_number] = FrameLabel(
                video_id=video_id,
                frame_number=frame_number,
                is_game_scene=bool(is_game_scene),
                confidence=confidence,
                annotator=annotator,
                created_at=created_at,
            )

        conn.close()

        self.logger.info(f"既存ラベル読み込み: {len(self.labels)}件")

    def get_frame(self, frame_number: int) -> np.ndarray | None:
        """
        指定フレームを取得

        Args:
            frame_number: フレーム番号

        Returns:
            フレーム画像
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame_number
            return frame
        else:
            return None

    def get_next_unlabeled_frame(self, start_from: int | None = None) -> int | None:
        """
        次のラベル未付与フレームを取得

        Args:
            start_from: 検索開始フレーム

        Returns:
            フレーム番号（ない場合はNone）
        """
        start = start_from if start_from is not None else self.current_frame

        for frame_num in range(start, self.total_frames):
            if frame_num not in self.labels:
                return frame_num

        # 最初から検索
        for frame_num in range(0, start):
            if frame_num not in self.labels:
                return frame_num

        return None

    def get_uncertainty_frame(self) -> int | None:
        """
        不確実性の高いフレームを取得

        Returns:
            フレーム番号（ない場合はNone）
        """
        if self.uncertainty_frames:
            return self.uncertainty_frames.pop(0)
        return None

    def label_frame(
        self, frame_number: int, is_game_scene: bool, annotator: str = "manual"
    ) -> FrameLabel:
        """
        フレームにラベルを付与

        Args:
            frame_number: フレーム番号
            is_game_scene: 対局画面かどうか
            annotator: アノテーター識別子

        Returns:
            付与したラベル
        """
        video_id = Path(self.video_path).stem

        label = FrameLabel(
            video_id=video_id,
            frame_number=frame_number,
            is_game_scene=is_game_scene,
            annotator=annotator,
        )

        self.labels[frame_number] = label
        self._save_label(label)

        return label

    def _save_label(self, label: FrameLabel):
        """ラベルをデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO game_scene_labels
            (video_id, frame_number, is_game_scene, confidence, annotator, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                label.video_id,
                label.frame_number,
                int(label.is_game_scene),
                label.confidence,
                label.annotator,
                label.created_at,
            ),
        )

        conn.commit()
        conn.close()

        # セッション情報を更新
        self._save_session_info()

    def auto_label_frame(self, frame_number: int) -> FrameLabel | None:
        """
        AIを使用してフレームを自動ラベリング

        Args:
            frame_number: フレーム番号

        Returns:
            付与したラベル（失敗時はNone）
        """
        if self.classifier is None:
            return None

        frame = self.get_frame(frame_number)
        if frame is None:
            return None

        # 分類
        is_game_scene, confidence = self.classifier.classify_frame(frame)

        # 不確実な場合は記録
        if 0.3 < confidence < 0.7 and frame_number not in self.uncertainty_frames:
            self.uncertainty_frames.append(frame_number)

        # ラベル作成
        video_id = Path(self.video_path).stem
        label = FrameLabel(
            video_id=video_id,
            frame_number=frame_number,
            is_game_scene=is_game_scene,
            confidence=confidence,
            annotator="auto",
        )

        self.labels[frame_number] = label
        self._save_label(label)

        return label

    def batch_label_frames(
        self, start_frame: int, end_frame: int, is_game_scene: bool, annotator: str = "manual"
    ) -> list[FrameLabel]:
        """
        複数フレームに一括でラベルを付与

        Args:
            start_frame: 開始フレーム
            end_frame: 終了フレーム（含む）
            is_game_scene: 対局画面かどうか
            annotator: アノテーター識別子

        Returns:
            付与したラベルのリスト
        """
        labels = []

        for frame_num in range(start_frame, min(end_frame + 1, self.total_frames)):
            label = self.label_frame(frame_num, is_game_scene, annotator)
            labels.append(label)

        return labels

    def get_statistics(self) -> dict[str, any]:
        """
        ラベリング統計を取得

        Returns:
            統計情報の辞書
        """
        total_labeled = len(self.labels)
        game_scenes = sum(1 for label in self.labels.values() if label.is_game_scene)
        non_game_scenes = total_labeled - game_scenes

        auto_labeled = sum(1 for label in self.labels.values() if label.annotator == "auto")
        manual_labeled = total_labeled - auto_labeled

        return {
            "total_frames": self.total_frames,
            "labeled_frames": total_labeled,
            "unlabeled_frames": self.total_frames - total_labeled,
            "progress": total_labeled / self.total_frames if self.total_frames > 0 else 0,
            "game_scenes": game_scenes,
            "non_game_scenes": non_game_scenes,
            "auto_labeled": auto_labeled,
            "manual_labeled": manual_labeled,
            "uncertainty_frames": len(self.uncertainty_frames),
        }

    def export_segments(self) -> list[dict[str, any]]:
        """
        連続するラベルからセグメントを生成してエクスポート

        Returns:
            セグメント情報のリスト
        """
        if not self.labels:
            return []

        # フレーム番号でソート
        sorted_frames = sorted(self.labels.keys())

        segments = []
        current_start = sorted_frames[0]
        current_is_game = self.labels[sorted_frames[0]].is_game_scene

        for i in range(1, len(sorted_frames)):
            frame_num = sorted_frames[i]
            label = self.labels[frame_num]

            # 連続性チェック（フレーム番号の差が大きい場合は別セグメント）
            if label.is_game_scene != current_is_game or frame_num - sorted_frames[i - 1] > 30:
                # セグメント終了
                segments.append(
                    {
                        "start_frame": current_start,
                        "end_frame": sorted_frames[i - 1],
                        "scene_type": "game" if current_is_game else "non_game",
                        "duration_frames": sorted_frames[i - 1] - current_start + 1,
                    }
                )

                # 新しいセグメント開始
                current_start = frame_num
                current_is_game = label.is_game_scene

        # 最後のセグメント
        segments.append(
            {
                "start_frame": current_start,
                "end_frame": sorted_frames[-1],
                "scene_type": "game" if current_is_game else "non_game",
                "duration_frames": sorted_frames[-1] - current_start + 1,
            }
        )

        # データベースに保存
        self._save_segments(segments)

        return segments

    def _save_segments(self, segments: list[dict[str, any]]):
        """セグメントをデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_id = Path(self.video_path).stem

        # 既存のセグメントを削除
        cursor.execute("DELETE FROM game_scene_segments WHERE video_id = ?", (video_id,))

        # 新しいセグメントを保存
        for segment in segments:
            cursor.execute(
                """
                INSERT INTO game_scene_segments
                (video_id, start_frame, end_frame, scene_type, confidence)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    video_id,
                    segment["start_frame"],
                    segment["end_frame"],
                    segment["scene_type"],
                    1.0,  # 手動ラベリングなので信頼度は最大
                ),
            )

        conn.commit()
        conn.close()

    def close(self):
        """リソースを解放"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
