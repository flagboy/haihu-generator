"""
教師データ管理システム

教師データの保存、読み込み、バージョン管理を行う
"""

import hashlib
import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.config import ConfigManager
from ..utils.file_io import FileIOHelper
from ..utils.logger import LoggerMixin
from .annotation_data import AnnotationData, FrameAnnotation, VideoAnnotation


class DatasetManager(LoggerMixin):
    """教師データ管理クラス"""

    def __init__(self, config_manager: ConfigManager | None = None):
        """
        初期化

        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or ConfigManager()
        self.config = self.config_manager.get_config()

        # データベース設定
        self.db_path = Path(
            self.config.get("training", {}).get("database_path", "data/training/dataset.db")
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # データセット保存ディレクトリ
        self.dataset_root = Path(
            self.config.get("training", {}).get("dataset_root", "data/training")
        )
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ
        self.images_dir = self.dataset_root / "images"
        self.annotations_dir = self.dataset_root / "annotations"
        self.versions_dir = self.dataset_root / "versions"
        self.exports_dir = self.dataset_root / "exports"

        for dir_path in [
            self.images_dir,
            self.annotations_dir,
            self.versions_dir,
            self.exports_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # データベース初期化
        self._init_database()

        self.logger.info(f"DatasetManager初期化完了: {self.dataset_root}")

    def _init_database(self):
        """データベースの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 動画テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    duration REAL,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT
                )
            """)

            # フレームテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY,
                    video_id TEXT,
                    image_path TEXT NOT NULL,
                    timestamp REAL,
                    width INTEGER,
                    height INTEGER,
                    quality_score REAL,
                    is_valid BOOLEAN,
                    scene_type TEXT,
                    game_phase TEXT,
                    annotated_at TEXT,
                    annotator TEXT,
                    notes TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                )
            """)

            # 牌アノテーションテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tile_annotations (
                    id TEXT PRIMARY KEY,
                    frame_id TEXT,
                    tile_id TEXT NOT NULL,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    confidence REAL,
                    area_type TEXT,
                    is_face_up BOOLEAN,
                    is_occluded BOOLEAN,
                    occlusion_ratio REAL,
                    annotator TEXT,
                    notes TEXT,
                    FOREIGN KEY (frame_id) REFERENCES frames (id)
                )
            """)

            # データセットバージョンテーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT,
                    frame_count INTEGER,
                    tile_count INTEGER,
                    export_path TEXT,
                    checksum TEXT
                )
            """)

            # インデックス作成
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames (video_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tiles_frame_id ON tile_annotations (frame_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tiles_tile_id ON tile_annotations (tile_id)"
            )

            conn.commit()

    def save_annotation_data(self, annotation_data: AnnotationData) -> bool:
        """
        アノテーションデータをデータベースに保存

        Args:
            annotation_data: アノテーションデータ

        Returns:
            保存成功かどうか
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for _video_id, video_annotation in annotation_data.video_annotations.items():
                    # 動画情報を保存
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO videos
                        (id, name, path, duration, fps, width, height, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            video_annotation.video_id,
                            video_annotation.video_name,
                            video_annotation.video_path,
                            video_annotation.duration,
                            video_annotation.fps,
                            video_annotation.width,
                            video_annotation.height,
                            video_annotation.created_at.isoformat()
                            if video_annotation.created_at
                            else None,
                            video_annotation.updated_at.isoformat()
                            if video_annotation.updated_at
                            else None,
                            json.dumps(video_annotation.metadata)
                            if video_annotation.metadata
                            else None,
                        ),
                    )

                    # フレーム情報を保存
                    for frame in video_annotation.frames:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO frames
                            (id, video_id, image_path, timestamp, width, height, quality_score,
                             is_valid, scene_type, game_phase, annotated_at, annotator, notes)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                frame.frame_id,
                                video_annotation.video_id,
                                frame.image_path,
                                frame.timestamp,
                                frame.image_width,
                                frame.image_height,
                                frame.quality_score,
                                frame.is_valid,
                                frame.scene_type,
                                frame.game_phase,
                                frame.annotated_at.isoformat() if frame.annotated_at else None,
                                frame.annotator,
                                frame.notes,
                            ),
                        )

                        # 牌アノテーションを保存
                        for tile in frame.tiles:
                            tile_annotation_id = str(uuid.uuid4())
                            cursor.execute(
                                """
                                INSERT OR REPLACE INTO tile_annotations
                                (id, frame_id, tile_id, x1, y1, x2, y2, confidence, area_type,
                                 is_face_up, is_occluded, occlusion_ratio, annotator, notes)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    tile_annotation_id,
                                    frame.frame_id,
                                    tile.tile_id,
                                    tile.bbox.x1,
                                    tile.bbox.y1,
                                    tile.bbox.x2,
                                    tile.bbox.y2,
                                    tile.confidence,
                                    tile.area_type,
                                    tile.is_face_up,
                                    tile.is_occluded,
                                    tile.occlusion_ratio,
                                    tile.annotator,
                                    tile.notes,
                                ),
                            )

                conn.commit()
                self.logger.info("アノテーションデータをデータベースに保存しました")
                return True

        except Exception as e:
            self.logger.error(f"アノテーションデータの保存に失敗: {e}")
            return False

    def load_annotation_data(self, video_id: str | None = None) -> AnnotationData:
        """
        データベースからアノテーションデータを読み込み

        Args:
            video_id: 特定の動画IDを指定（Noneの場合は全て）

        Returns:
            アノテーションデータ
        """
        annotation_data = AnnotationData()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 動画情報を取得
                if video_id:
                    cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
                else:
                    cursor.execute("SELECT * FROM videos")

                videos = cursor.fetchall()

                for video_row in videos:
                    vid = video_row[0]

                    # VideoAnnotationを作成
                    video_annotation = VideoAnnotation(
                        video_id=video_row[0],
                        video_path=video_row[2],
                        video_name=video_row[1],
                        duration=video_row[3] or 0.0,
                        fps=video_row[4] or 30.0,
                        width=video_row[5] or 1920,
                        height=video_row[6] or 1080,
                        frames=[],
                        created_at=datetime.fromisoformat(video_row[7]) if video_row[7] else None,
                        updated_at=datetime.fromisoformat(video_row[8]) if video_row[8] else None,
                        metadata=json.loads(video_row[9]) if video_row[9] else {},
                    )

                    # フレーム情報を取得
                    cursor.execute("SELECT * FROM frames WHERE video_id = ?", (vid,))
                    frames = cursor.fetchall()

                    for frame_row in frames:
                        frame_id = frame_row[0]

                        # 牌アノテーションを取得
                        cursor.execute(
                            "SELECT * FROM tile_annotations WHERE frame_id = ?", (frame_id,)
                        )
                        tile_rows = cursor.fetchall()

                        tiles = []
                        for tile_row in tile_rows:
                            from .annotation_data import BoundingBox, TileAnnotation

                            bbox = BoundingBox(
                                x1=tile_row[3], y1=tile_row[4], x2=tile_row[5], y2=tile_row[6]
                            )

                            tile = TileAnnotation(
                                tile_id=tile_row[2],
                                bbox=bbox,
                                confidence=tile_row[7] or 1.0,
                                area_type=tile_row[8] or "unknown",
                                is_face_up=bool(tile_row[9]) if tile_row[9] is not None else True,
                                is_occluded=bool(tile_row[10])
                                if tile_row[10] is not None
                                else False,
                                occlusion_ratio=tile_row[11] or 0.0,
                                annotator=tile_row[12] or "unknown",
                                notes=tile_row[13] or "",
                            )
                            tiles.append(tile)

                        # FrameAnnotationを作成
                        frame_annotation = FrameAnnotation(
                            frame_id=frame_row[0],
                            image_path=frame_row[2],
                            image_width=frame_row[4] or 1920,
                            image_height=frame_row[5] or 1080,
                            timestamp=frame_row[3] or 0.0,
                            tiles=tiles,
                            quality_score=frame_row[6] or 1.0,
                            is_valid=bool(frame_row[7]) if frame_row[7] is not None else True,
                            scene_type=frame_row[8] or "game",
                            game_phase=frame_row[9] or "unknown",
                            annotated_at=datetime.fromisoformat(frame_row[10])
                            if frame_row[10]
                            else None,
                            annotator=frame_row[11] or "unknown",
                            notes=frame_row[12] or "",
                        )

                        video_annotation.frames.append(frame_annotation)

                    annotation_data.video_annotations[vid] = video_annotation

                self.logger.info(
                    f"アノテーションデータを読み込み: {len(annotation_data.video_annotations)}動画"
                )

        except Exception as e:
            self.logger.error(f"アノテーションデータの読み込みに失敗: {e}")

        return annotation_data

    def create_dataset_version(
        self, annotation_data: AnnotationData, version: str, description: str = ""
    ) -> str | None:
        """
        データセットのバージョンを作成

        Args:
            annotation_data: アノテーションデータ
            version: バージョン名
            description: 説明

        Returns:
            バージョンID
        """
        try:
            version_id = str(uuid.uuid4())
            version_dir = self.versions_dir / version_id
            version_dir.mkdir(exist_ok=True)

            # アノテーションデータをJSONで保存
            annotation_path = version_dir / "annotations.json"
            annotation_data.save_to_json(str(annotation_path))

            # 統計情報を保存
            stats = annotation_data.get_all_statistics()
            stats_path = version_dir / "statistics.json"
            # default=str to handle datetime objects
            stats_json = json.loads(json.dumps(stats, default=str))
            FileIOHelper.save_json(stats_json, stats_path, pretty=True)

            # チェックサムを計算
            checksum = self._calculate_checksum(annotation_path)

            # データベースに記録
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO dataset_versions
                    (id, version, description, created_at, frame_count, tile_count, export_path, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        version_id,
                        version,
                        description,
                        datetime.now().isoformat(),
                        stats.get("total_annotated_frames", 0),
                        stats.get("total_tiles", 0),
                        str(version_dir),
                        checksum,
                    ),
                )
                conn.commit()

            self.logger.info(f"データセットバージョン作成: {version} (ID: {version_id})")
            return version_id

        except Exception as e:
            self.logger.error(f"データセットバージョンの作成に失敗: {e}")
            return None

    def export_dataset(
        self, version_id: str, export_format: str = "yolo", output_dir: str | None = None
    ) -> bool:
        """
        データセットをエクスポート

        Args:
            version_id: バージョンID
            export_format: エクスポート形式 ("yolo", "coco", "pascal_voc")
            output_dir: 出力ディレクトリ

        Returns:
            エクスポート成功かどうか
        """
        try:
            # バージョン情報を取得
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM dataset_versions WHERE id = ?", (version_id,))
                version_row = cursor.fetchone()

                if not version_row:
                    self.logger.error(f"バージョンが見つかりません: {version_id}")
                    return False

            version_dir = Path(version_row[6])  # export_path
            annotation_path = version_dir / "annotations.json"

            # アノテーションデータを読み込み
            annotation_data = AnnotationData()
            annotation_data.load_from_json(str(annotation_path))

            # 出力ディレクトリを設定
            if output_dir is None:
                output_dir = self.exports_dir / f"{version_row[1]}_{export_format}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # 形式に応じてエクスポート
            if export_format == "yolo":
                return self._export_yolo_format(annotation_data, output_dir)
            elif export_format == "coco":
                return self._export_coco_format(annotation_data, output_dir)
            elif export_format == "pascal_voc":
                return self._export_pascal_voc_format(annotation_data, output_dir)
            else:
                self.logger.error(f"サポートされていない形式: {export_format}")
                return False

        except Exception as e:
            self.logger.error(f"データセットエクスポートに失敗: {e}")
            return False

    def _export_yolo_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """YOLO形式でエクスポート"""
        try:
            # クラスマッピングを作成
            all_tile_types = set()
            for video_annotation in annotation_data.video_annotations.values():
                for frame in video_annotation.frames:
                    for tile in frame.tiles:
                        all_tile_types.add(tile.tile_id)

            class_mapping = {tile_type: i for i, tile_type in enumerate(sorted(all_tile_types))}

            # YOLO形式でエクスポート
            return annotation_data.export_yolo_format(str(output_dir), class_mapping)

        except Exception as e:
            self.logger.error(f"YOLO形式エクスポートに失敗: {e}")
            return False

    def _export_coco_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """COCO形式でエクスポート（基本実装）"""
        # TODO: COCO形式の実装
        self.logger.warning("COCO形式は未実装です")
        return False

    def _export_pascal_voc_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """Pascal VOC形式でエクスポート（基本実装）"""
        # TODO: Pascal VOC形式の実装
        self.logger.warning("Pascal VOC形式は未実装です")
        return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """ファイルのチェックサムを計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_dataset_statistics(self) -> dict[str, Any]:
        """データセット全体の統計情報を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 基本統計
                cursor.execute("SELECT COUNT(*) FROM videos")
                video_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM frames")
                frame_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM tile_annotations")
                tile_count = cursor.fetchone()[0]

                # 牌種類別統計
                cursor.execute("SELECT tile_id, COUNT(*) FROM tile_annotations GROUP BY tile_id")
                tile_distribution = dict(cursor.fetchall())

                # エリア別統計
                cursor.execute(
                    "SELECT area_type, COUNT(*) FROM tile_annotations GROUP BY area_type"
                )
                area_distribution = dict(cursor.fetchall())

                # バージョン情報
                cursor.execute("SELECT COUNT(*) FROM dataset_versions")
                version_count = cursor.fetchone()[0]

                return {
                    "video_count": video_count,
                    "frame_count": frame_count,
                    "tile_count": tile_count,
                    "tile_distribution": tile_distribution,
                    "area_distribution": area_distribution,
                    "version_count": version_count,
                    "dataset_root": str(self.dataset_root),
                    "database_path": str(self.db_path),
                }

        except Exception as e:
            self.logger.error(f"統計情報の取得に失敗: {e}")
            return {}

    def list_versions(self) -> list[dict[str, Any]]:
        """データセットバージョン一覧を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, version, description, created_at, frame_count, tile_count
                    FROM dataset_versions ORDER BY created_at DESC
                """)

                versions = []
                for row in cursor.fetchall():
                    versions.append(
                        {
                            "id": row[0],
                            "version": row[1],
                            "description": row[2],
                            "created_at": row[3],
                            "frame_count": row[4],
                            "tile_count": row[5],
                        }
                    )

                return versions

        except Exception as e:
            self.logger.error(f"バージョン一覧の取得に失敗: {e}")
            return []

    def cleanup_old_versions(self, keep_count: int = 5) -> bool:
        """古いバージョンをクリーンアップ"""
        try:
            versions = self.list_versions()
            if len(versions) <= keep_count:
                return True

            # 古いバージョンを削除
            to_delete = versions[keep_count:]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for version in to_delete:
                    # ディレクトリを削除
                    version_dir = self.versions_dir / version["id"]
                    if version_dir.exists():
                        shutil.rmtree(version_dir)

                    # データベースから削除
                    cursor.execute("DELETE FROM dataset_versions WHERE id = ?", (version["id"],))

                conn.commit()

            self.logger.info(f"古いバージョンを削除: {len(to_delete)}個")
            return True

        except Exception as e:
            self.logger.error(f"バージョンクリーンアップに失敗: {e}")
            return False
