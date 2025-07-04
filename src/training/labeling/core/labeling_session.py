"""
ラベリングセッション管理モジュール（新規）
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class LabelingSession:
    """ラベリングセッションを管理するクラス"""

    def __init__(self, session_id: str | None = None, data_dir: str = "data/training/sessions"):
        """
        初期化

        Args:
            session_id: セッションID（Noneの場合は新規作成）
            data_dir: セッションデータの保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if session_id:
            self.session_id = session_id
            self._load_session()
        else:
            self.session_id = str(uuid.uuid4())
            self._create_new_session()

    def _create_new_session(self):
        """新規セッションを作成"""
        self.session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "video_info": {},
            "hand_regions": {},
            "progress": {"total_frames": 0, "labeled_frames": 0, "current_frame": 0},
            "annotations": {},
            "metadata": {"labeler": "", "notes": "", "version": "1.0"},
        }
        self._save_session()
        logger.info(f"新規セッションを作成: {self.session_id}")

    def _load_session(self):
        """既存セッションを読み込み"""
        session_file = self._get_session_file()
        if session_file.exists():
            with open(session_file, encoding="utf-8") as f:
                self.session_data = json.load(f)
            logger.info(f"セッションを読み込み: {self.session_id}")
        else:
            logger.warning(f"セッションが見つかりません: {self.session_id}")
            self._create_new_session()

    def _save_session(self):
        """セッションを保存"""
        self.session_data["updated_at"] = datetime.now().isoformat()
        session_file = self._get_session_file()

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

    def _get_session_file(self) -> Path:
        """セッションファイルのパスを取得"""
        return self.data_dir / f"{self.session_id}.json"

    def set_video_info(self, video_info: dict[str, Any]):
        """動画情報を設定"""
        self.session_data["video_info"] = video_info
        self.session_data["progress"]["total_frames"] = video_info.get("frame_count", 0)
        self._save_session()
        logger.debug(f"動画情報を設定: {video_info.get('path', 'unknown')}")

    def set_hand_regions(self, regions: dict[str, dict[str, float]]):
        """手牌領域を設定"""
        self.session_data["hand_regions"] = regions
        self._save_session()
        logger.debug("手牌領域を設定しました")

    def add_annotation(self, frame_number: int, player: str, tiles: list[dict[str, Any]]):
        """
        アノテーションを追加

        Args:
            frame_number: フレーム番号
            player: プレイヤー名
            tiles: 牌のアノテーション情報
        """
        frame_key = str(frame_number)

        if frame_key not in self.session_data["annotations"]:
            self.session_data["annotations"][frame_key] = {
                "timestamp": frame_number / self.session_data["video_info"].get("fps", 30),
                "players": {},
            }

        self.session_data["annotations"][frame_key]["players"][player] = {
            "tiles": tiles,
            "labeled_at": datetime.now().isoformat(),
        }

        # 進捗を更新
        self._update_progress()
        self._save_session()

        logger.debug(f"アノテーションを追加: フレーム {frame_number}, {player}")

    def get_annotation(self, frame_number: int, player: str | None = None) -> dict | None:
        """
        アノテーションを取得

        Args:
            frame_number: フレーム番号
            player: プレイヤー名（Noneの場合は全プレイヤー）

        Returns:
            アノテーション情報
        """
        frame_key = str(frame_number)

        if frame_key not in self.session_data["annotations"]:
            return None

        frame_annotations = self.session_data["annotations"][frame_key]

        if player:
            return frame_annotations.get("players", {}).get(player)
        else:
            return frame_annotations

    def _update_progress(self):
        """進捗を更新"""
        # ラベル済みフレーム数を計算
        labeled_frames = len(self.session_data["annotations"])
        self.session_data["progress"]["labeled_frames"] = labeled_frames

        # 現在のフレームを更新
        if self.session_data["annotations"]:
            latest_frame = max(int(f) for f in self.session_data["annotations"])
            self.session_data["progress"]["current_frame"] = latest_frame

    def get_progress(self) -> dict[str, int]:
        """進捗情報を取得"""
        return self.session_data["progress"].copy()

    def set_current_frame(self, frame_number: int):
        """現在のフレーム番号を設定"""
        self.session_data["progress"]["current_frame"] = frame_number
        self._save_session()

    def export_annotations(self, format: str = "tenhou") -> dict:
        """
        アノテーションをエクスポート

        Args:
            format: エクスポート形式（"tenhou"）

        Returns:
            エクスポートされたデータ
        """
        if format == "tenhou":
            return self._export_tenhou_format()
        else:
            raise ValueError(f"未対応のフォーマット: {format}")

    def _export_tenhou_format(self) -> dict:
        """天鳳形式でエクスポート"""
        tenhou_data = {"session_id": self.session_id, "frames": []}

        for frame_str, frame_data in self.session_data["annotations"].items():
            frame_number = int(frame_str)
            frame_info = {"frame": frame_number, "timestamp": frame_data["timestamp"], "hands": {}}

            for player, player_data in frame_data.get("players", {}).items():
                tiles = [tile["label"] for tile in player_data.get("tiles", [])]
                frame_info["hands"][player] = tiles

            tenhou_data["frames"].append(frame_info)

        return tenhou_data

    def get_unlabeled_frames(self) -> list[int]:
        """未ラベルのフレーム番号リストを取得"""
        total_frames = self.session_data["progress"]["total_frames"]
        labeled_frames = {int(f) for f in self.session_data["annotations"]}

        unlabeled = []
        for i in range(total_frames):
            if i not in labeled_frames:
                unlabeled.append(i)

        return unlabeled

    def get_session_summary(self) -> dict:
        """セッションのサマリー情報を取得"""
        return {
            "session_id": self.session_id,
            "created_at": self.session_data["created_at"],
            "updated_at": self.session_data["updated_at"],
            "video_path": self.session_data["video_info"].get("path", ""),
            "progress": self.get_progress(),
            "total_annotations": sum(
                len(frame_data.get("players", {}))
                for frame_data in self.session_data["annotations"].values()
            ),
        }

    @classmethod
    def list_sessions(cls, data_dir: str = "data/training/sessions") -> list[dict]:
        """セッション一覧を取得"""
        data_path = Path(data_dir)
        sessions = []

        for session_file in data_path.glob("*.json"):
            try:
                session = cls(session_id=session_file.stem, data_dir=data_dir)
                sessions.append(session.get_session_summary())
            except Exception as e:
                logger.error(f"セッション読み込みエラー: {session_file.name} - {e}")

        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)
