"""
対局画面学習用データセット管理

ラベリングされたデータを学習用に管理・提供
"""

import json
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ....utils.config import ConfigManager
from ....utils.logger import LoggerMixin


class SceneDataset(Dataset, LoggerMixin):
    """対局画面学習用データセット"""

    def __init__(
        self,
        db_path: str | None = None,
        cache_dir: str | None = None,
        transform: transforms.Compose | None = None,
        split: str = "train",  # train, val, test
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        config_manager: ConfigManager | None = None,
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

        # 設定管理を初期化
        self.config_manager = config_manager or ConfigManager()
        config = self.config_manager.get_config()

        # データベースパスの設定
        if db_path is None:
            db_path = config.get("directories", {}).get(
                "game_scene_db", "web_interface/data/training/game_scene_labels.db"
            )

        # キャッシュディレクトリの設定
        if cache_dir is None:
            cache_dir = config.get("directories", {}).get(
                "game_scene_cache", "web_interface/data/training/game_scene_cache"
            )

        # パスを絶対パスに変換
        if not os.path.isabs(db_path):
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.db_path = str(project_root / db_path)
        else:
            self.db_path = db_path

        # キャッシュディレクトリパスを絶対パスに変換
        if not os.path.isabs(cache_dir):
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.cache_dir = project_root / cache_dir
        else:
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

        # キャッシュ作成の並列化オプション
        self.use_cache_preload = True  # キャッシュ事前作成を有効化

        # VideoCapture の再利用（パフォーマンス最適化）
        self._video_cache = {}  # video_path -> VideoCapture のキャッシュ
        self._last_access_time = {}  # アクセス時間記録
        self._cache_max_size = 2  # 最大2つのVideoCapture を保持

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
        raw_video_info = cursor.fetchall()
        video_info = {}

        for row in raw_video_info:
            video_id = row[0]
            video_path = row[1]

            # パスが存在するかチェックし、存在しない場合は代替パスを探す
            if video_path and Path(video_path).exists():
                video_info[video_id] = video_path
                self.logger.info(f"動画パス確認済み: {video_id} -> {video_path}")
            else:
                # 代替パスを検索
                alternative_paths = [
                    f"web_interface/uploads/{video_id}.mp4",
                    f"uploads/{video_id}.mp4",
                    f"{video_id}.mp4",
                ]

                found_path = None
                project_root = Path(__file__).parent.parent.parent.parent.parent

                for alt_path in alternative_paths:
                    full_path = project_root / alt_path
                    if full_path.exists():
                        found_path = str(full_path)
                        break

                if found_path:
                    video_info[video_id] = found_path
                    self.logger.info(f"代替パス見つかりました: {video_id} -> {found_path}")
                else:
                    self.logger.warning(
                        f"動画ファイルが見つかりません: {video_id}, 元パス: {video_path}"
                    )
                    video_info[video_id] = video_path  # 元のパスを保持

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
            # タプルから値を取得（インデックスでアクセス）
            video_id = row[0]
            frame_number = row[1]
            is_game_scene = row[2]
            confidence = row[3]
            annotator = row[4]

            # video_infoから動画パスを取得
            video_path = video_info.get(video_id)

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

    def _get_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """
        VideoCapture を取得（キャッシュ優先）

        Args:
            video_path: 動画ファイルパス

        Returns:
            VideoCapture オブジェクト
        """
        import time

        # キャッシュから取得
        if video_path in self._video_cache:
            cap = self._video_cache[video_path]
            if cap.isOpened():
                self._last_access_time[video_path] = time.time()
                self.logger.debug(f"📹 VideoCapture キャッシュヒット: {video_path}")
                return cap
            else:
                # 無効なCapture を削除
                self.logger.debug(f"🗑️ 無効なVideoCapture削除: {video_path}")
                del self._video_cache[video_path]
                if video_path in self._last_access_time:
                    del self._last_access_time[video_path]

        # キャッシュサイズ制限
        if len(self._video_cache) >= self._cache_max_size:
            # 最も古いアクセスのものを削除
            oldest_path = min(
                self._last_access_time.keys(), key=lambda k: self._last_access_time[k]
            )
            old_cap = self._video_cache[oldest_path]
            old_cap.release()
            del self._video_cache[oldest_path]
            del self._last_access_time[oldest_path]
            self.logger.debug(f"🧹 古いVideoCapture削除: {oldest_path}")

        # 新しいVideoCapture を作成
        self.logger.debug(f"🆕 新しいVideoCapture作成: {video_path}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小化

        if cap.isOpened():
            self._video_cache[video_path] = cap
            self._last_access_time[video_path] = time.time()
            return cap
        else:
            self.logger.error(f"❌ VideoCapture作成失敗: {video_path}")
            cap.release()
            return None

    def _cleanup_video_cache(self):
        """VideoCapture キャッシュをクリーンアップ"""
        for _video_path, cap in self._video_cache.items():
            cap.release()
        self._video_cache.clear()
        self._last_access_time.clear()
        self.logger.debug("🧹 VideoCapture キャッシュをクリーンアップしました")

    def __del__(self):
        """デストラクタでVideoCapture を解放"""
        if hasattr(self, "_video_cache"):
            self._cleanup_video_cache()

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
        import time

        start_time = time.time()

        try:
            item = self.data[idx]
            self.logger.info(
                f"🔍 データ取得開始: idx={idx}, video={item['video_id']}, frame={item['frame_number']}"
            )

            # キャッシュから画像を読み込み
            load_start = time.time()
            image = self._load_frame(item["video_path"], item["frame_number"], item["video_id"])
            load_time = time.time() - load_start

            if image is None:
                # エラー時はダミーデータを返す
                self.logger.warning(
                    f"⚠️ ダミーフレーム使用: idx={idx}, frame={item['frame_number']}, load_time={load_time:.3f}s"
                )
                image = self._create_dummy_frame()
                if image is None:
                    # ダミーフレーム作成も失敗した場合の最終手段
                    self.logger.error(f"❌ ダミーフレーム作成失敗: idx={idx}")
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                self.logger.info(
                    f"✅ フレーム読み込み成功: idx={idx}, load_time={load_time:.3f}s, shape={image.shape}"
                )

            # 変換を適用
            transform_start = time.time()
            if self.transform:
                try:
                    image = self.transform(image)
                    transform_time = time.time() - transform_start
                    self.logger.debug(
                        f"🔄 画像変換完了: idx={idx}, transform_time={transform_time:.3f}s"
                    )
                except Exception as e:
                    transform_time = time.time() - transform_start
                    self.logger.error(
                        f"❌ 画像変換エラー: idx={idx}, transform_time={transform_time:.3f}s, error={e}"
                    )
                    # 変換エラー時は最小限の処理でテンソル化
                    image = torch.zeros((3, 224, 224), dtype=torch.float32)

            total_time = time.time() - start_time
            self.logger.info(
                f"⏱️ データ取得完了: idx={idx}, total_time={total_time:.3f}s (load: {load_time:.3f}s)"
            )

            return image, item["label"]

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(
                f"❌ データ取得エラー: idx={idx}, total_time={total_time:.3f}s, error={e}"
            )
            # 完全なエラー時は安全なダミーデータを返す
            dummy_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_label = 0  # デフォルトラベル
            return dummy_image, dummy_label

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
        import time

        # load_start = time.time()  # 未使用のため削除

        # キャッシュパスを生成
        cache_path = self.cache_dir / video_id / f"frame_{frame_number:06d}.jpg"
        self.logger.debug(f"📁 キャッシュパス確認: {cache_path}")

        # キャッシュから読み込み（優先）
        if cache_path.exists():
            try:
                cache_load_start = time.time()
                image = cv2.imread(str(cache_path))
                cache_load_time = time.time() - cache_load_start

                if image is not None and image.size > 0:
                    # キャッシュから正常に読み込めた場合
                    self.logger.debug(
                        f"💾 キャッシュ読み込み成功: frame={frame_number}, time={cache_load_time:.3f}s"
                    )
                    return image
                else:
                    # 破損したキャッシュファイルを削除
                    self.logger.warning(
                        f"💥 破損キャッシュファイル削除: {cache_path}, time={cache_load_time:.3f}s"
                    )
                    cache_path.unlink(missing_ok=True)
            except Exception as e:
                cache_load_time = time.time() - cache_load_start
                self.logger.warning(
                    f"❌ キャッシュ読み込みエラー: {cache_path}, time={cache_load_time:.3f}s, error={e}"
                )
                # エラーのあるキャッシュファイルを削除
                cache_path.unlink(missing_ok=True)
        else:
            self.logger.debug(f"📭 キャッシュファイル未存在: {cache_path}")

        # 動画から読み込み
        video_load_start = time.time()
        self.logger.info(f"🎬 動画からフレーム読み込み開始: frame={frame_number}")

        # パスの正規化と存在確認
        if not os.path.isabs(video_path):
            # プロジェクトルートを基準にした絶対パスに変換
            project_root = Path(__file__).parent.parent.parent.parent.parent
            video_path = str(project_root / video_path)

        # 元のパスが存在しない場合、代替パスを試す
        if not Path(video_path).exists():
            self.logger.debug(f"🔍 代替パス検索中: {video_path}")
            project_root = Path(__file__).parent.parent.parent.parent.parent
            alternative_paths = [
                project_root / "web_interface" / "uploads" / f"{video_id}.mp4",
                project_root / "uploads" / f"{video_id}.mp4",
                project_root / f"{video_id}.mp4",
            ]

            found_path = None
            for alt_path in alternative_paths:
                if alt_path.exists():
                    found_path = str(alt_path)
                    break

            if found_path:
                video_path = found_path
                self.logger.info(f"✅ 代替パスを使用: {video_id} -> {video_path}")
            else:
                self.logger.error(
                    f"❌ 動画ファイルが見つかりません: {video_path} (video_id: {video_id})"
                )
                return None

        # VideoCapture を取得（キャッシュ経由で高速化）
        opencv_start = time.time()
        cap = self._get_video_capture(video_path)
        opencv_init_time = time.time() - opencv_start

        if cap is None:
            self.logger.error(
                f"❌ VideoCapture取得失敗: {video_path}, init_time={opencv_init_time:.3f}s"
            )
            return None

        self.logger.debug(f"📹 VideoCapture取得完了: init_time={opencv_init_time:.3f}s")

        try:
            # フレーム位置設定
            seek_start = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            seek_time = time.time() - seek_start
            self.logger.debug(
                f"⏩ フレーム位置設定: frame={frame_number}, seek_time={seek_time:.3f}s"
            )

            # 複数回読み込みを試行（デコードエラー対策）
            max_retries = 3
            for retry in range(max_retries):
                read_start = time.time()
                ret, frame = cap.read()
                read_time = time.time() - read_start

                self.logger.debug(
                    f"🎞️ フレーム読み込み試行 {retry + 1}/{max_retries}: ret={ret}, read_time={read_time:.3f}s"
                )

                if ret and frame is not None and frame.size > 0:
                    # フレームが正常に読み込めた場合
                    self.logger.info(
                        f"✅ フレーム読み込み成功: frame={frame_number}, shape={frame.shape}, read_time={read_time:.3f}s"
                    )

                    # キャッシュに保存
                    cache_save_start = time.time()
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    success = cv2.imwrite(str(cache_path), frame)
                    cache_save_time = time.time() - cache_save_start

                    if success:
                        self.logger.debug(
                            f"💾 キャッシュ保存成功: {cache_path}, save_time={cache_save_time:.3f}s"
                        )
                        return frame
                    else:
                        self.logger.warning(
                            f"⚠️ キャッシュ保存失敗: {cache_path}, save_time={cache_save_time:.3f}s"
                        )
                        return frame  # 保存は失敗したがフレームは有効
                elif retry < max_retries - 1:
                    # リトライする場合は少し位置をずらす
                    self.logger.warning(
                        f"🔄 フレーム読み込みリトライ {retry + 1}/{max_retries}: frame={frame_number}, read_time={read_time:.3f}s"
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1 + retry))
                    continue
                else:
                    # 最終試行でも失敗
                    total_video_time = time.time() - video_load_start
                    self.logger.error(
                        f"❌ フレーム読み込み失敗（全試行終了）: {video_path} frame={frame_number}, total_time={total_video_time:.3f}s"
                    )

                    # ダミーフレームを返す（学習を継続するため）
                    dummy_frame = self._create_dummy_frame()
                    if dummy_frame is not None:
                        self.logger.warning(f"🔧 ダミーフレームを使用: frame={frame_number}")
                        return dummy_frame

                    return None

        except Exception as e:
            total_video_time = time.time() - video_load_start
            self.logger.error(
                f"❌ フレーム読み込み中にエラー: {video_path} frame={frame_number}, total_time={total_video_time:.3f}s, error={e}"
            )
            # エラー時もダミーフレームで継続
            dummy_frame = self._create_dummy_frame()
            if dummy_frame is not None:
                return dummy_frame
            return None
        finally:
            # VideoCapture はキャッシュで管理されるため解放しない
            total_video_time = time.time() - video_load_start
            self.logger.debug(f"📹 動画読み込み完了: total_video_time={total_video_time:.3f}s")

    def _create_dummy_frame(self) -> np.ndarray | None:
        """
        エラー時のダミーフレームを作成

        Returns:
            ダミーフレーム（224x224のグレー画像）
        """
        try:
            # 224x224の灰色画像を作成
            dummy_frame = np.full((224, 224, 3), 128, dtype=np.uint8)  # 灰色
            return dummy_frame
        except Exception as e:
            self.logger.error(f"ダミーフレーム作成エラー: {e}")
            return None

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
