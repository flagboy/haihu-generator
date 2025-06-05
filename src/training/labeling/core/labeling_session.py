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
            latest_frame = max(int(f) for f in self.session_data["annotations"].keys())
            self.session_data["progress"]["current_frame"] = latest_frame

    def get_progress(self) -> dict[str, int]:
        """進捗情報を取得"""
        return self.session_data["progress"].copy()

    def set_current_frame(self, frame_number: int):
        """現在のフレーム番号を設定"""
        self.session_data["progress"]["current_frame"] = frame_number
        self._save_session()

    def export_annotations(self, format: str = "coco") -> dict:
        """
        アノテーションをエクスポート

        Args:
            format: エクスポート形式（"coco", "yolo", "tenhou"）

        Returns:
            エクスポートされたデータ
        """
        if format == "coco":
            return self._export_coco_format()
        elif format == "yolo":
            return self._export_yolo_format()
        elif format == "tenhou":
            return self._export_tenhou_format()
        else:
            raise ValueError(f"未対応のフォーマット: {format}")

    def _export_coco_format(self) -> dict:
        """COCO形式でエクスポート"""
        coco_data = {
            "info": {
                "description": "麻雀牌ラベリングデータ",
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "session_id": self.session_id,
            },
            "images": [],
            "annotations": [],
            "categories": self._get_tile_categories(),
        }

        annotation_id = 1

        for frame_str, frame_data in self.session_data["annotations"].items():
            frame_number = int(frame_str)

            # 画像情報
            image_info = {
                "id": frame_number,
                "file_name": f"frame_{frame_number:08d}.jpg",
                "width": self.session_data["video_info"].get("width", 1920),
                "height": self.session_data["video_info"].get("height", 1080),
            }
            coco_data["images"].append(image_info)

            # アノテーション情報
            for player, player_data in frame_data.get("players", {}).items():
                for tile_idx, tile in enumerate(player_data.get("tiles", [])):
                    annotation = {
                        "id": annotation_id,
                        "image_id": frame_number,
                        "category_id": self._tile_to_category_id(tile["label"]),
                        "bbox": [tile["x"], tile["y"], tile["w"], tile["h"]],
                        "area": tile["w"] * tile["h"],
                        "iscrowd": 0,
                        "attributes": {
                            "player": player,
                            "tile_index": tile_idx,
                            "confidence": tile.get("confidence", 1.0),
                        },
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

        return coco_data

    def _export_yolo_format(self) -> dict:
        """YOLO形式でエクスポート"""
        yolo_data = {}

        for frame_str, frame_data in self.session_data["annotations"].items():
            frame_number = int(frame_str)
            annotations = []

            for player, player_data in frame_data.get("players", {}).items():
                for tile in player_data.get("tiles", []):
                    # YOLOフォーマット: class x_center y_center width height
                    class_id = self._tile_to_category_id(tile["label"]) - 1  # 0-indexed
                    x_center = (tile["x"] + tile["w"] / 2) / self.session_data["video_info"][
                        "width"
                    ]
                    y_center = (tile["y"] + tile["h"] / 2) / self.session_data["video_info"][
                        "height"
                    ]
                    width = tile["w"] / self.session_data["video_info"]["width"]
                    height = tile["h"] / self.session_data["video_info"]["height"]

                    annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

            yolo_data[f"frame_{frame_number:08d}.txt"] = "\n".join(annotations)

        return yolo_data

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

    def _get_tile_categories(self) -> list[dict]:
        """牌のカテゴリ一覧を取得"""
        categories = []
        tile_types = [
            # 萬子
            "1m",
            "2m",
            "3m",
            "4m",
            "5m",
            "6m",
            "7m",
            "8m",
            "9m",
            # 筒子
            "1p",
            "2p",
            "3p",
            "4p",
            "5p",
            "6p",
            "7p",
            "8p",
            "9p",
            # 索子
            "1s",
            "2s",
            "3s",
            "4s",
            "5s",
            "6s",
            "7s",
            "8s",
            "9s",
            # 字牌
            "1z",
            "2z",
            "3z",
            "4z",
            "5z",
            "6z",
            "7z",
        ]

        for i, tile_type in enumerate(tile_types, 1):
            categories.append({"id": i, "name": tile_type, "supercategory": "tile"})

        return categories

    def _tile_to_category_id(self, tile_label: str) -> int:
        """牌ラベルをカテゴリIDに変換"""
        tile_map = {
            # 萬子
            "1m": 1,
            "2m": 2,
            "3m": 3,
            "4m": 4,
            "5m": 5,
            "6m": 6,
            "7m": 7,
            "8m": 8,
            "9m": 9,
            # 筒子
            "1p": 10,
            "2p": 11,
            "3p": 12,
            "4p": 13,
            "5p": 14,
            "6p": 15,
            "7p": 16,
            "8p": 17,
            "9p": 18,
            # 索子
            "1s": 19,
            "2s": 20,
            "3s": 21,
            "4s": 22,
            "5s": 23,
            "6s": 24,
            "7s": 25,
            "8s": 26,
            "9s": 27,
            # 字牌
            "1z": 28,
            "2z": 29,
            "3z": 30,
            "4z": 31,
            "5z": 32,
            "6z": 33,
            "7z": 34,
        }
        return tile_map.get(tile_label, 0)

    def get_unlabeled_frames(self) -> list[int]:
        """未ラベルのフレーム番号リストを取得"""
        total_frames = self.session_data["progress"]["total_frames"]
        labeled_frames = set(int(f) for f in self.session_data["annotations"].keys())

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
