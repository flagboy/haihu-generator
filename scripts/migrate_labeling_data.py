#!/usr/bin/env python3
"""
手牌ラベリングデータ移行スクリプト
旧システムから新システムへのデータ移行
"""

import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.labeling.core.labeling_session import LabelingSession
from src.utils.logger import setup_logger


class LabelingDataMigrator:
    """ラベリングデータ移行クラス"""

    def __init__(self, old_data_dir: str, new_data_dir: str):
        """
        初期化

        Args:
            old_data_dir: 旧データディレクトリ
            new_data_dir: 新データディレクトリ
        """
        self.old_data_dir = Path(old_data_dir)
        self.new_data_dir = Path(new_data_dir)

        # ログ設定
        setup_logger("data_migration")

        # 統計情報
        self.stats = {
            "sessions_migrated": 0,
            "frames_migrated": 0,
            "annotations_migrated": 0,
            "hand_areas_migrated": 0,
            "errors": [],
        }

    def migrate_all(self):
        """全データを移行"""
        logger.info("データ移行を開始します")
        logger.info(f"旧データ: {self.old_data_dir}")
        logger.info(f"新データ: {self.new_data_dir}")

        # ディレクトリ作成
        self.new_data_dir.mkdir(parents=True, exist_ok=True)

        # 各種データの移行
        self._migrate_hand_areas()
        self._migrate_frame_data()
        self._migrate_annotations()
        self._migrate_sessions()

        # 統計情報を表示
        self._print_statistics()

    def _migrate_hand_areas(self):
        """手牌領域設定を移行"""
        logger.info("手牌領域設定の移行を開始")

        # hand_labeling_systemの設定ファイルを検索
        labeling_configs = list(
            self.old_data_dir.glob("hand_labeling_system/**/hand_regions*.json")
        )

        # hand_training_systemの設定ファイルを検索
        training_configs = list(self.old_data_dir.glob("hand_training_system/**/config*.json"))

        all_configs = labeling_configs + training_configs

        for config_file in all_configs:
            try:
                with open(config_file, encoding="utf-8") as f:
                    config_data = json.load(f)

                # 手牌領域データを抽出
                hand_regions = None

                if "hand_regions" in config_data:
                    hand_regions = config_data["hand_regions"]
                elif "regions" in config_data:
                    hand_regions = config_data["regions"]

                if hand_regions:
                    # プレイヤー番号から方向への変換
                    converted_regions = self._convert_player_to_direction(hand_regions)

                    # 新しい形式で保存
                    output_file = (
                        self.new_data_dir / "hand_areas" / f"{config_file.stem}_migrated.json"
                    )
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "regions": converted_regions,
                                "migrated_from": str(config_file),
                                "migrated_at": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                    self.stats["hand_areas_migrated"] += 1
                    logger.debug(f"手牌領域設定を移行: {config_file.name}")

            except Exception as e:
                logger.error(f"手牌領域設定の移行エラー: {config_file} - {e}")
                self.stats["errors"].append(f"Hand area: {config_file.name} - {str(e)}")

    def _convert_player_to_direction(self, regions: dict) -> dict:
        """プレイヤー番号を方向に変換"""
        # マッピング
        player_to_direction = {
            "player1": "bottom",
            "player2": "right",
            "player3": "top",
            "player4": "left",
        }

        converted = {}
        for key, value in regions.items():
            # すでに方向形式の場合はそのまま
            if key in ["bottom", "right", "top", "left"]:
                converted[key] = value
            # プレイヤー番号形式の場合は変換
            elif key in player_to_direction:
                converted[player_to_direction[key]] = value
            else:
                # その他の形式は保持
                converted[key] = value

        return converted

    def _migrate_frame_data(self):
        """フレームデータを移行"""
        logger.info("フレームデータの移行を開始")

        # 旧システムのフレームディレクトリ
        old_frame_dirs = [
            self.old_data_dir / "hand_labeling_system" / "data" / "frames",
            self.old_data_dir / "hand_training_system" / "data" / "frames",
            self.old_data_dir / "frames",
            self.old_data_dir / "training" / "frames",
        ]

        # 新しいフレームディレクトリ
        new_frame_dir = self.new_data_dir / "frames"
        new_frame_dir.mkdir(parents=True, exist_ok=True)

        for old_dir in old_frame_dirs:
            if not old_dir.exists():
                continue

            # フレーム画像をコピー
            frame_files = list(old_dir.glob("**/*.jpg")) + list(old_dir.glob("**/*.png"))

            for frame_file in frame_files:
                try:
                    # 相対パスを保持してコピー
                    relative_path = frame_file.relative_to(old_dir)
                    new_path = new_frame_dir / relative_path
                    new_path.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(frame_file, new_path)
                    self.stats["frames_migrated"] += 1

                except Exception as e:
                    logger.error(f"フレーム移行エラー: {frame_file} - {e}")
                    self.stats["errors"].append(f"Frame: {frame_file.name} - {str(e)}")

        logger.info(f"{self.stats['frames_migrated']}個のフレームを移行しました")

    def _migrate_annotations(self):
        """アノテーションデータを移行"""
        logger.info("アノテーションデータの移行を開始")

        # 旧システムのアノテーションファイル
        annotation_patterns = ["**/*annotations*.json", "**/*labels*.json", "**/dataset.db"]

        for pattern in annotation_patterns:
            annotation_files = list(self.old_data_dir.glob(pattern))

            for ann_file in annotation_files:
                if ann_file.suffix == ".db":
                    self._migrate_sqlite_annotations(ann_file)
                else:
                    self._migrate_json_annotations(ann_file)

    def _migrate_json_annotations(self, json_file: Path):
        """JSONアノテーションを移行"""
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # アノテーション形式を判定して変換
            if isinstance(data, list):
                # リスト形式の場合
                for item in data:
                    self._process_annotation_item(item)
            elif isinstance(data, dict):
                # 辞書形式の場合
                if "annotations" in data:
                    for ann in data["annotations"]:
                        self._process_annotation_item(ann)
                else:
                    self._process_annotation_item(data)

            logger.debug(f"JSONアノテーションを移行: {json_file.name}")

        except Exception as e:
            logger.error(f"JSONアノテーション移行エラー: {json_file} - {e}")
            self.stats["errors"].append(f"Annotation: {json_file.name} - {str(e)}")

    def _migrate_sqlite_annotations(self, db_file: Path):
        """SQLiteデータベースからアノテーションを移行"""
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # テーブル構造を確認
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # アノテーションテーブルの場合
                if "annotation" in table_name.lower() or "label" in table_name.lower():
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()

                    # カラム名を取得
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]

                    # 各行を処理
                    for row in rows:
                        ann_data = dict(zip(columns, row, strict=False))
                        self._process_annotation_item(ann_data)

            conn.close()
            logger.debug(f"SQLiteアノテーションを移行: {db_file.name}")

        except Exception as e:
            logger.error(f"SQLiteアノテーション移行エラー: {db_file} - {e}")
            self.stats["errors"].append(f"SQLite: {db_file.name} - {str(e)}")

    def _process_annotation_item(self, ann_data: dict[str, Any]):
        """個別のアノテーションアイテムを処理"""
        try:
            # 新しいセッションを作成（または既存のものを使用）
            session_id = (
                ann_data.get("session_id") or ann_data.get("video_id") or "migrated_session"
            )
            session = LabelingSession(
                session_id=session_id, data_dir=str(self.new_data_dir / "sessions")
            )

            # フレーム番号を取得
            frame_number = ann_data.get("frame_number") or ann_data.get("frame_id") or 0

            # プレイヤーを特定
            player = ann_data.get("player") or self._determine_player(ann_data)

            # 牌データを抽出
            tiles = self._extract_tiles(ann_data)

            if tiles:
                session.add_annotation(frame_number, player, tiles)
                self.stats["annotations_migrated"] += 1

        except Exception as e:
            logger.error(f"アノテーション処理エラー: {e}")

    def _determine_player(self, ann_data: dict) -> str:
        """アノテーションデータからプレイヤーを特定"""
        # 位置情報から推定
        if "position" in ann_data:
            pos = ann_data["position"]
            if isinstance(pos, str):
                return self._convert_player_to_direction({"player": pos}).get("player", "bottom")

        # 座標から推定
        if "y" in ann_data:
            y = ann_data["y"]
            if y < 0.3:
                return "top"
            elif y > 0.7:
                return "bottom"

        return "bottom"  # デフォルト

    def _extract_tiles(self, ann_data: dict) -> list[dict]:
        """アノテーションデータから牌情報を抽出"""
        tiles = []

        # tiles フィールドがある場合
        if "tiles" in ann_data:
            return ann_data["tiles"]

        # label フィールドがある場合
        if "label" in ann_data:
            tiles.append(
                {
                    "index": ann_data.get("index", 0),
                    "label": ann_data["label"],
                    "x": ann_data.get("x", 0),
                    "y": ann_data.get("y", 0),
                    "w": ann_data.get("w", 50),
                    "h": ann_data.get("h", 70),
                    "confidence": ann_data.get("confidence", 1.0),
                }
            )

        return tiles

    def _migrate_sessions(self):
        """セッションデータを移行"""
        logger.info("セッションデータの移行を開始")

        # 既存のセッションファイルを検索
        session_files = list(self.old_data_dir.glob("**/session*.json"))

        for session_file in session_files:
            try:
                with open(session_file, encoding="utf-8") as f:
                    session_data = json.load(f)

                # 新しいセッションとして保存
                new_session = LabelingSession(data_dir=str(self.new_data_dir / "sessions"))

                # 動画情報を設定
                if "video_info" in session_data:
                    new_session.set_video_info(session_data["video_info"])

                # 手牌領域を設定
                if "hand_regions" in session_data:
                    regions = self._convert_player_to_direction(session_data["hand_regions"])
                    new_session.set_hand_regions(regions)

                self.stats["sessions_migrated"] += 1
                logger.debug(f"セッションを移行: {session_file.name}")

            except Exception as e:
                logger.error(f"セッション移行エラー: {session_file} - {e}")
                self.stats["errors"].append(f"Session: {session_file.name} - {str(e)}")

    def _print_statistics(self):
        """統計情報を表示"""
        logger.info("=" * 50)
        logger.info("データ移行統計:")
        logger.info(f"  セッション: {self.stats['sessions_migrated']}個")
        logger.info(f"  フレーム: {self.stats['frames_migrated']}個")
        logger.info(f"  アノテーション: {self.stats['annotations_migrated']}個")
        logger.info(f"  手牌領域設定: {self.stats['hand_areas_migrated']}個")
        logger.info(f"  エラー: {len(self.stats['errors'])}個")

        if self.stats["errors"]:
            logger.warning("エラー詳細:")
            for error in self.stats["errors"][:10]:  # 最初の10個のみ表示
                logger.warning(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                logger.warning(f"  ... 他 {len(self.stats['errors']) - 10} 個のエラー")

        logger.info("=" * 50)


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="手牌ラベリングデータを移行")
    parser.add_argument(
        "--old-dir",
        type=str,
        default=".",
        help="旧データディレクトリ（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument(
        "--new-dir",
        type=str,
        default="data/training",
        help="新データディレクトリ（デフォルト: data/training）",
    )
    parser.add_argument("--dry-run", action="store_true", help="実際の移行を行わずに確認のみ")

    args = parser.parse_args()

    if args.dry_run:
        logger.info("ドライランモード: 実際の移行は行いません")
        # TODO: ドライラン実装
    else:
        migrator = LabelingDataMigrator(args.old_dir, args.new_dir)
        migrator.migrate_all()


if __name__ == "__main__":
    main()
