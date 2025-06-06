"""
対局画面学習用データセット管理

ラベリングされたデータを学習用に管理・提供
"""

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ....utils.logger import LoggerMixin


class SceneDataset(Dataset, LoggerMixin):
    """対局画面学習用データセット"""

    def __init__(
        self,
        db_path: str = "web_interface/data/training/game_scene_labels.db",
        cache_dir: str = "web_interface/data/training/game_scene_cache",
        transform: transforms.Compose | None = None,
        split: str = "train",  # train, val, test
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        """
        初期化

        Args:
            db_path: ラベルデータベースのパス
            cache_dir: 画像キャッシュディレクトリ
            transform: 画像変換
            split: データセット分割（train/val/test）
            split_ratio: 分割比率（train, val, test）
        """
        super().__init__()
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.split_ratio = split_ratio

        # デフォルトの変換
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

        # データを読み込み
        self._load_data()

        self.logger.info(f"SceneDataset初期化完了: {self.split} ({len(self.data)}サンプル)")

    def _load_data(self):
        """データベースからデータを読み込み"""
        if not Path(self.db_path).exists():
            self.logger.warning(f"データベースが見つかりません: {self.db_path}")
            self.data = []
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # まず、すべてのvideo_idと最新のvideo_pathを取得
        cursor.execute("""
            SELECT DISTINCT l.video_id,
                   (SELECT video_path FROM labeling_sessions
                    WHERE video_id = l.video_id
                    ORDER BY created_at DESC LIMIT 1) as video_path
            FROM game_scene_labels l
        """)
        video_info = {row[0]: row[1] for row in cursor.fetchall()}

        self.logger.info(f"動画情報: {video_info}")

        # ラベル付きデータを取得（シンプルなクエリ）
        cursor.execute("""
            SELECT video_id, frame_number, is_game_scene, confidence, annotator
            FROM game_scene_labels
            ORDER BY video_id, frame_number
        """)

        all_data = []
        rows = cursor.fetchall()
        self.logger.info(f"SQLクエリ結果: {len(rows)}行")

        for idx, row in enumerate(rows):
            # sqlite3.Rowオブジェクトから値を取得
            video_id = row["video_id"]
            frame_number = row["frame_number"]
            is_game_scene = row["is_game_scene"]
            confidence = row["confidence"]
            annotator = row["annotator"]
            video_path = row["video_path"]

            # デバッグ：最初の数行を出力
            if idx < 5:
                self.logger.debug(
                    f"Row {idx}: video_id={video_id}, frame={frame_number}, is_game={is_game_scene}, path={video_path}"
                )

            # video_pathがNoneの場合の処理
            if video_path is None:
                self.logger.warning(f"video_pathがNULL: video_id={video_id}, frame={frame_number}")
                continue

            all_data.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "frame_number": frame_number,
                    "label": int(is_game_scene),
                    "confidence": confidence,
                    "annotator": annotator,
                }
            )

        conn.close()

        # デバッグ：読み込みデータの統計
        total_game_scenes = sum(1 for item in all_data if item["label"] == 1)
        total_non_game_scenes = len(all_data) - total_game_scenes
        self.logger.info(
            f"データベースから読み込み: 総数={len(all_data)}, "
            f"対局画面={total_game_scenes}, 非対局画面={total_non_game_scenes}"
        )

        # データを分割
        self._split_data(all_data)

    def _split_data(self, all_data: list[dict]):
        """データを train/val/test に分割"""
        if not all_data:
            self.data = []
            return

        # 動画IDごとにグループ化（同じ動画のフレームは同じ分割に）
        video_groups = {}
        for item in all_data:
            video_id = item["video_id"]
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(item)

        # 動画IDをシャッフルして分割
        video_ids = list(video_groups.keys())
        np.random.seed(42)  # 再現性のため
        np.random.shuffle(video_ids)

        n_videos = len(video_ids)

        # 動画が少ない場合はフレームレベルで分割
        if n_videos <= 3:
            # 全フレームをシャッフルして分割
            np.random.seed(42)
            np.random.shuffle(all_data)

            n_frames = len(all_data)
            train_end = int(n_frames * self.split_ratio[0])
            val_end = train_end + int(n_frames * self.split_ratio[1])

            if self.split == "train":
                self.data = all_data[:train_end]
            elif self.split == "val":
                self.data = all_data[train_end:val_end]
            elif self.split == "test":
                self.data = all_data[val_end:]
            else:
                raise ValueError(f"不明な分割: {self.split}")

            # デバッグ情報を追加
            game_scenes = sum(1 for item in self.data if item["label"] == 1)
            non_game_scenes = len(self.data) - game_scenes
            self.logger.info(
                f"データ分割完了（フレームレベル）: {self.split} - {len(self.data)}フレーム "
                f"(対局画面: {game_scenes}, 非対局画面: {non_game_scenes})"
            )
            return

        # 動画が十分ある場合は動画レベルで分割
        train_end = int(n_videos * self.split_ratio[0])
        val_end = train_end + int(n_videos * self.split_ratio[1])

        train_videos = video_ids[:train_end]
        val_videos = video_ids[train_end:val_end]
        test_videos = video_ids[val_end:]

        # 分割に基づいてデータを選択
        if self.split == "train":
            selected_videos = train_videos
        elif self.split == "val":
            selected_videos = val_videos
        elif self.split == "test":
            selected_videos = test_videos
        else:
            raise ValueError(f"不明な分割: {self.split}")

        self.data = []
        for video_id in selected_videos:
            self.data.extend(video_groups[video_id])

        self.logger.info(
            f"データ分割完了: {self.split} - {len(selected_videos)}動画, {len(self.data)}フレーム"
        )

    def __len__(self) -> int:
        """データセットのサイズ"""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        指定インデックスのデータを取得

        Args:
            idx: インデックス

        Returns:
            (画像テンソル, ラベル)
        """
        item = self.data[idx]

        # キャッシュから画像を読み込み
        image = self._load_frame(item["video_path"], item["frame_number"], item["video_id"])

        if image is None:
            # エラー時はダミーデータを返す
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # 変換を適用
        if self.transform:
            image = self.transform(image)

        return image, item["label"]

    def _load_frame(self, video_path: str, frame_number: int, video_id: str) -> np.ndarray | None:
        """
        フレームを読み込み（キャッシュ優先）

        Args:
            video_path: 動画ファイルパス
            frame_number: フレーム番号
            video_id: 動画ID

        Returns:
            画像データ（BGR）
        """
        # キャッシュパスを生成
        cache_path = self.cache_dir / video_id / f"frame_{frame_number:06d}.jpg"

        # キャッシュから読み込み
        if cache_path.exists():
            image = cv2.imread(str(cache_path))
            if image is not None:
                return image

        # 動画から読み込み
        if not Path(video_path).exists():
            self.logger.warning(f"動画ファイルが見つかりません: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"動画を開けません: {video_path}")
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret and frame is not None:
                # キャッシュに保存
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cache_path), frame)
                return frame
            else:
                self.logger.error(f"フレーム読み込み失敗: {video_path} frame={frame_number}")
                return None

        finally:
            cap.release()

    def get_class_weights(self) -> torch.Tensor:
        """
        クラスの重みを計算（不均衡データ対策）

        Returns:
            クラスの重み
        """
        if not self.data:
            return torch.tensor([1.0, 1.0])

        # クラスごとのサンプル数を計算
        class_counts = [0, 0]
        for item in self.data:
            class_counts[item["label"]] += 1

        # 重みを計算（少数クラスに大きな重み）
        total = sum(class_counts)
        weights = []
        for count in class_counts:
            if count > 0:
                weights.append(total / (len(class_counts) * count))
            else:
                weights.append(1.0)

        return torch.tensor(weights, dtype=torch.float32)

    def get_statistics(self) -> dict[str, any]:
        """
        データセットの統計情報を取得

        Returns:
            統計情報
        """
        if not self.data:
            return {
                "total_samples": 0,
                "game_scenes": 0,
                "non_game_scenes": 0,
                "videos": 0,
                "annotators": {},
            }

        game_scenes = sum(1 for item in self.data if item["label"] == 1)
        non_game_scenes = len(self.data) - game_scenes

        videos = {item["video_id"] for item in self.data}

        annotators = {}
        for item in self.data:
            annotator = item["annotator"]
            if annotator not in annotators:
                annotators[annotator] = 0
            annotators[annotator] += 1

        return {
            "total_samples": len(self.data),
            "game_scenes": game_scenes,
            "non_game_scenes": non_game_scenes,
            "game_ratio": game_scenes / len(self.data) if self.data else 0,
            "videos": len(videos),
            "annotators": annotators,
            "split": self.split,
        }

    def export_split_info(self, output_path: str):
        """
        データ分割情報をエクスポート

        Args:
            output_path: 出力ファイルパス
        """
        split_info = {"split": self.split, "statistics": self.get_statistics(), "samples": []}

        for item in self.data:
            split_info["samples"].append(
                {
                    "video_id": item["video_id"],
                    "frame_number": item["frame_number"],
                    "label": item["label"],
                    "annotator": item["annotator"],
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        self.logger.info(f"分割情報をエクスポート: {output_path}")
