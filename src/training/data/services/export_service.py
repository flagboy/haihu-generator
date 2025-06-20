"""
エクスポートサービス

データセットのエクスポート機能を提供
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any

from ....utils.file_io import FileIOHelper
from ....utils.logger import LoggerMixin
from ...annotation_data import AnnotationData


class ExportService(LoggerMixin):
    """エクスポートサービスクラス"""

    def __init__(self, export_root: Path | str = "data/training/exports"):
        """
        初期化

        Args:
            export_root: エクスポートルートディレクトリ
        """
        self.export_root = Path(export_root)
        self.export_root.mkdir(parents=True, exist_ok=True)

    def export_dataset(
        self,
        annotation_data: AnnotationData,
        export_format: str,
        output_dir: Path | str | None = None,
        version_name: str | None = None,
    ) -> Path | None:
        """
        データセットをエクスポート

        Args:
            annotation_data: アノテーションデータ
            export_format: エクスポート形式 ("yolo", "coco", "pascal_voc")
            output_dir: 出力ディレクトリ
            version_name: バージョン名

        Returns:
            エクスポートディレクトリパスまたはNone
        """
        try:
            # 出力ディレクトリの設定
            if output_dir is None:
                version_suffix = f"_{version_name}" if version_name else ""
                output_dir = self.export_root / f"export_{export_format}{version_suffix}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # 形式に応じてエクスポート
            success = False
            if export_format == "yolo":
                success = self._export_yolo_format(annotation_data, output_dir)
            elif export_format == "coco":
                success = self._export_coco_format(annotation_data, output_dir)
            elif export_format == "pascal_voc":
                success = self._export_pascal_voc_format(annotation_data, output_dir)
            else:
                self.logger.error(f"サポートされていない形式: {export_format}")
                return None

            if success:
                self.logger.info(f"データセットエクスポート完了: {output_dir}")
                return output_dir
            else:
                return None

        except Exception as e:
            self.logger.error(f"データセットエクスポートに失敗: {e}")
            return None

    def _export_yolo_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """
        YOLO形式でエクスポート

        Args:
            annotation_data: アノテーションデータ
            output_dir: 出力ディレクトリ

        Returns:
            成功したかどうか
        """
        try:
            # クラスマッピングを作成
            all_tile_types = set()
            for video_annotation in annotation_data.video_annotations.values():
                for frame in video_annotation.frames:
                    for tile in frame.tiles:
                        all_tile_types.add(tile.tile_id)

            class_mapping = {tile_type: i for i, tile_type in enumerate(sorted(all_tile_types))}

            # classes.txtを保存
            classes_file = output_dir / "classes.txt"
            with open(classes_file, "w") as f:
                for tile_type in sorted(all_tile_types):
                    f.write(f"{tile_type}\n")

            # YOLO形式でエクスポート
            success = annotation_data.export_yolo_format(str(output_dir), class_mapping)

            # データセット情報を保存
            dataset_info = {
                "format": "yolo",
                "classes": class_mapping,
                "num_classes": len(class_mapping),
                "statistics": annotation_data.get_all_statistics(),
            }
            info_file = output_dir / "dataset_info.json"
            FileIOHelper.save_json(dataset_info, info_file, pretty=True)

            return success

        except Exception as e:
            self.logger.error(f"YOLO形式エクスポートに失敗: {e}")
            return False

    def _export_coco_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """
        COCO形式でエクスポート

        Args:
            annotation_data: アノテーションデータ
            output_dir: 出力ディレクトリ

        Returns:
            成功したかどうか
        """
        try:
            # COCO形式のデータ構造を作成
            coco_data = {
                "info": {
                    "description": "Mahjong Tile Detection Dataset",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "HaihuGenerator",
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [],
            }

            # カテゴリを作成
            all_tile_types = set()
            for video_annotation in annotation_data.video_annotations.values():
                for frame in video_annotation.frames:
                    for tile in frame.tiles:
                        all_tile_types.add(tile.tile_id)

            for i, tile_type in enumerate(sorted(all_tile_types)):
                coco_data["categories"].append(
                    {
                        "id": i + 1,  # COCOは1-indexed
                        "name": tile_type,
                        "supercategory": "tile",
                    }
                )

            # 画像とアノテーションを追加
            image_id = 1
            annotation_id = 1

            images_dir = output_dir / "images"
            images_dir.mkdir(exist_ok=True)

            for video_annotation in annotation_data.video_annotations.values():
                for frame in video_annotation.frames:
                    # 画像情報を追加
                    image_name = Path(frame.image_path).name
                    coco_data["images"].append(
                        {
                            "id": image_id,
                            "file_name": image_name,
                            "width": frame.image_width,
                            "height": frame.image_height,
                        }
                    )

                    # 画像をコピー
                    src_path = Path(frame.image_path)
                    if src_path.exists():
                        dst_path = images_dir / image_name
                        shutil.copy2(src_path, dst_path)

                    # アノテーションを追加
                    for tile in frame.tiles:
                        category_id = sorted(all_tile_types).index(tile.tile_id) + 1
                        bbox_width = tile.bbox.x2 - tile.bbox.x1
                        bbox_height = tile.bbox.y2 - tile.bbox.y1

                        coco_data["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [tile.bbox.x1, tile.bbox.y1, bbox_width, bbox_height],
                                "area": bbox_width * bbox_height,
                                "segmentation": [],
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1

                    image_id += 1

            # annotations.jsonを保存
            annotations_file = output_dir / "annotations.json"
            FileIOHelper.save_json(coco_data, annotations_file, pretty=True)

            self.logger.info(f"COCO形式エクスポート完了: {len(coco_data['images'])}画像")
            return True

        except Exception as e:
            self.logger.error(f"COCO形式エクスポートに失敗: {e}")
            return False

    def _export_pascal_voc_format(self, annotation_data: AnnotationData, output_dir: Path) -> bool:
        """
        Pascal VOC形式でエクスポート

        Args:
            annotation_data: アノテーションデータ
            output_dir: 出力ディレクトリ

        Returns:
            成功したかどうか
        """
        try:
            # Pascal VOC形式のディレクトリ構造を作成
            annotations_dir = output_dir / "Annotations"
            images_dir = output_dir / "JPEGImages"
            imagesets_dir = output_dir / "ImageSets" / "Main"

            for dir_path in [annotations_dir, images_dir, imagesets_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # ファイルリスト
            all_files = []

            for video_annotation in annotation_data.video_annotations.values():
                for frame in video_annotation.frames:
                    # 画像名（拡張子なし）
                    image_name = Path(frame.image_path).stem
                    all_files.append(image_name)

                    # 画像をコピー
                    src_path = Path(frame.image_path)
                    if src_path.exists():
                        dst_path = images_dir / f"{image_name}.jpg"
                        shutil.copy2(src_path, dst_path)

                    # XMLアノテーションを作成
                    xml_content = self._create_pascal_voc_xml(frame, image_name)
                    xml_path = annotations_dir / f"{image_name}.xml"
                    with open(xml_path, "w") as f:
                        f.write(xml_content)

            # trainval.txtを作成
            trainval_file = imagesets_dir / "trainval.txt"
            with open(trainval_file, "w") as f:
                for file_name in all_files:
                    f.write(f"{file_name}\n")

            self.logger.info(f"Pascal VOC形式エクスポート完了: {len(all_files)}画像")
            return True

        except Exception as e:
            self.logger.error(f"Pascal VOC形式エクスポートに失敗: {e}")
            return False

    def _create_pascal_voc_xml(self, frame: Any, image_name: str) -> str:
        """
        Pascal VOC形式のXMLを作成

        Args:
            frame: フレームアノテーション
            image_name: 画像名

        Returns:
            XML文字列
        """
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<annotation>",
            f"  <filename>{image_name}.jpg</filename>",
            "  <source>",
            "    <database>HaihuGenerator</database>",
            "  </source>",
            "  <size>",
            f"    <width>{frame.image_width}</width>",
            f"    <height>{frame.image_height}</height>",
            "    <depth>3</depth>",
            "  </size>",
            "  <segmented>0</segmented>",
        ]

        for tile in frame.tiles:
            xml_parts.extend(
                [
                    "  <object>",
                    f"    <name>{tile.tile_id}</name>",
                    "    <pose>Unspecified</pose>",
                    "    <truncated>0</truncated>",
                    f"    <difficult>{1 if tile.is_occluded else 0}</difficult>",
                    "    <bndbox>",
                    f"      <xmin>{tile.bbox.x1}</xmin>",
                    f"      <ymin>{tile.bbox.y1}</ymin>",
                    f"      <xmax>{tile.bbox.x2}</xmax>",
                    f"      <ymax>{tile.bbox.y2}</ymax>",
                    "    </bndbox>",
                    "  </object>",
                ]
            )

        xml_parts.append("</annotation>")
        return "\n".join(xml_parts)

    def calculate_checksum(self, directory: Path) -> str:
        """
        ディレクトリのチェックサムを計算

        Args:
            directory: ディレクトリパス

        Returns:
            チェックサム
        """
        hash_md5 = hashlib.md5()

        # ディレクトリ内のファイルをソートして処理
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                # ファイルパスとコンテンツをハッシュに含める
                relative_path = file_path.relative_to(directory)
                hash_md5.update(str(relative_path).encode())

                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

        return hash_md5.hexdigest()
