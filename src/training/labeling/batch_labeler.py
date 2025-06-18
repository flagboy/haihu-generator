"""
バッチラベリング機能

複数フレームの一括ラベリングを効率的に行うための機能を提供
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BatchLabeler:
    """複数フレームの一括ラベリング機能"""

    def __init__(self, interface=None):
        """
        Args:
            interface: ラベリングインターフェース（Webまたはスタンドアロン）
        """
        self.interface = interface
        self.templates = {}
        self.optical_flow_cache = {}

    def create_template_from_frame(
        self, frame_id: str, annotations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        現在のフレームからテンプレートを作成

        Args:
            frame_id: フレームID
            annotations: フレームのアノテーションリスト

        Returns:
            作成されたテンプレート
        """
        template = {
            "id": f"template_{datetime.now().timestamp()}",
            "source_frame_id": frame_id,
            "created_at": datetime.now().isoformat(),
            "annotations": [],
        }

        # フレームサイズを取得（正規化のため）
        frame_shape = self.interface.get_frame_shape(frame_id) if self.interface else (1920, 1080)

        height, width = frame_shape[:2]

        for ann in annotations:
            # 相対位置を計算（0-1の範囲に正規化）
            bbox = ann["bbox"]
            rel_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]

            template["annotations"].append(
                {
                    "relative_bbox": rel_bbox,
                    "absolute_bbox": bbox,
                    "tile_type": ann.get("tile_type", "unknown"),
                    "tile_id": ann.get("tile_id", "unknown"),
                    "player_id": ann.get("player_id", 0),
                    "area_type": ann.get("area_type", "hand"),
                }
            )

        # テンプレートを保存
        self.templates[template["id"]] = template

        return template

    def apply_template_to_frames(
        self,
        template_id: str,
        frame_ids: list[str],
        auto_adjust: bool = True,
        confidence_threshold: float = 0.8,
    ) -> list[dict[str, Any]]:
        """
        テンプレートを複数フレームに適用

        Args:
            template_id: テンプレートID
            frame_ids: 適用先のフレームIDリスト
            auto_adjust: 自動位置調整を行うか
            confidence_threshold: 適用の信頼度閾値

        Returns:
            適用結果のリスト
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")

        template = self.templates[template_id]
        results = []

        for frame_id in frame_ids:
            if auto_adjust and self.interface:
                # フレーム間の位置ずれを自動補正
                adjusted_template = self._adjust_template_for_frame(
                    template, frame_id, confidence_threshold
                )
            else:
                adjusted_template = template

            # アノテーションの適用
            annotations = self._apply_template(adjusted_template, frame_id)

            results.append(
                {
                    "frame_id": frame_id,
                    "annotations": annotations,
                    "success": len(annotations) > 0,
                    "applied_count": len(annotations),
                    "confidence": self._calculate_confidence(annotations),
                }
            )

        return results

    def smart_propagation(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        annotations: list[dict[str, Any]],
        confidence_threshold: float = 0.8,
        max_movement: float = 50.0,
    ) -> dict[int, list[dict]]:
        """
        スマートな前方伝播

        Args:
            start_frame_idx: 開始フレームインデックス
            end_frame_idx: 終了フレームインデックス
            annotations: 開始フレームのアノテーション
            confidence_threshold: 信頼度閾値
            max_movement: 最大移動量（ピクセル）

        Returns:
            フレームインデックスとアノテーションのマッピング
        """
        propagated_annotations = {start_frame_idx: annotations}
        current_annotations = annotations.copy()

        for frame_idx in range(start_frame_idx + 1, end_frame_idx + 1):
            if not self.interface:
                break

            # 前フレームとの差分を計算
            frame_diff = self._calculate_frame_difference(frame_idx - 1, frame_idx)

            if frame_diff < 0.1:  # ほぼ同じフレーム
                # アノテーションをそのままコピー
                propagated_annotations[frame_idx] = [ann.copy() for ann in current_annotations]
            else:
                # オプティカルフローで位置を追跡
                try:
                    adjusted_annotations = self._track_tiles_optical_flow(
                        current_annotations, frame_idx - 1, frame_idx, max_movement
                    )

                    # 信頼度の高いものだけを保持
                    filtered = [
                        ann
                        for ann in adjusted_annotations
                        if ann.get("confidence", 0) > confidence_threshold
                    ]

                    if filtered:
                        propagated_annotations[frame_idx] = filtered
                        current_annotations = filtered
                    else:
                        # 追跡失敗
                        break

                except Exception as e:
                    print(f"オプティカルフロー追跡エラー: {e}")
                    break

        return propagated_annotations

    def batch_apply_by_pattern(
        self, pattern: str, annotations: list[dict[str, Any]], frame_range: tuple[int, int]
    ) -> dict[int, list[dict]]:
        """
        パターンに基づいてバッチ適用

        Args:
            pattern: 適用パターン（'every_n', 'similar', 'static'）
            annotations: 適用するアノテーション
            frame_range: フレーム範囲

        Returns:
            適用結果
        """
        start_idx, end_idx = frame_range
        results = {}

        if pattern == "every_n":
            # N フレームごとに適用
            n = 5  # 5フレームごと
            for idx in range(start_idx, end_idx + 1, n):
                results[idx] = [ann.copy() for ann in annotations]

        elif pattern == "similar":
            # 類似フレームに適用
            if self.interface:
                base_frame = self.interface.get_frame(start_idx)

                for idx in range(start_idx, end_idx + 1):
                    current_frame = self.interface.get_frame(idx)
                    similarity = self._calculate_similarity(base_frame, current_frame)

                    if similarity > 0.9:  # 90%以上類似
                        results[idx] = [ann.copy() for ann in annotations]

        elif pattern == "static":
            # 静止シーンに適用
            for idx in range(start_idx, end_idx + 1):
                if idx > start_idx:
                    diff = self._calculate_frame_difference(idx - 1, idx)
                    if diff < 0.05:  # ほぼ静止
                        results[idx] = [ann.copy() for ann in annotations]

        return results

    def _adjust_template_for_frame(
        self, template: dict[str, Any], frame_id: str, confidence_threshold: float
    ) -> dict[str, Any]:
        """テンプレートをフレームに合わせて調整"""
        if not self.interface:
            return template

        # フレームの特徴点を検出して位置合わせ
        source_frame_id = template["source_frame_id"]

        try:
            # 簡易的な位置調整（実際にはより高度な手法を使用）
            offset = self._estimate_global_offset(source_frame_id, frame_id)

            adjusted_template = template.copy()
            adjusted_template["annotations"] = []

            for ann in template["annotations"]:
                adjusted_ann = ann.copy()
                # オフセットを適用
                adjusted_ann["absolute_bbox"] = [
                    ann["absolute_bbox"][0] + offset[0],
                    ann["absolute_bbox"][1] + offset[1],
                    ann["absolute_bbox"][2] + offset[0],
                    ann["absolute_bbox"][3] + offset[1],
                ]
                adjusted_template["annotations"].append(adjusted_ann)

            return adjusted_template

        except Exception:
            # 調整に失敗した場合は元のテンプレートを返す
            return template

    def _apply_template(self, template: dict[str, Any], frame_id: str) -> list[dict[str, Any]]:
        """テンプレートを適用してアノテーションを生成"""
        annotations = []

        # フレームサイズを取得
        frame_shape = self.interface.get_frame_shape(frame_id) if self.interface else (1920, 1080)

        height, width = frame_shape[:2]

        for template_ann in template["annotations"]:
            # 絶対座標を優先、なければ相対座標から計算
            if "absolute_bbox" in template_ann:
                bbox = template_ann["absolute_bbox"]
            else:
                rel_bbox = template_ann["relative_bbox"]
                bbox = [
                    rel_bbox[0] * width,
                    rel_bbox[1] * height,
                    rel_bbox[2] * width,
                    rel_bbox[3] * height,
                ]

            # 画像範囲内に収まるようクリップ
            bbox = [
                max(0, min(bbox[0], width)),
                max(0, min(bbox[1], height)),
                max(0, min(bbox[2], width)),
                max(0, min(bbox[3], height)),
            ]

            annotation = {
                "frame_id": frame_id,
                "bbox": bbox,
                "tile_type": template_ann.get("tile_type", "unknown"),
                "tile_id": template_ann.get("tile_id", "unknown"),
                "player_id": template_ann.get("player_id", 0),
                "area_type": template_ann.get("area_type", "hand"),
                "from_template": True,
                "template_id": template.get("id", "unknown"),
            }

            annotations.append(annotation)

        return annotations

    def _track_tiles_optical_flow(
        self, annotations: list[dict], prev_frame_idx: int, curr_frame_idx: int, max_movement: float
    ) -> list[dict]:
        """オプティカルフローを使用した牌の追跡"""
        if not self.interface:
            return annotations

        # フレームを取得
        prev_frame = self.interface.get_frame(prev_frame_idx)
        curr_frame = self.interface.get_frame(curr_frame_idx)

        # グレースケールに変換
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # オプティカルフローのキャッシュを確認
        cache_key = f"{prev_frame_idx}_{curr_frame_idx}"
        if cache_key in self.optical_flow_cache:
            flow = self.optical_flow_cache[cache_key]
        else:
            # オプティカルフロー計算
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            self.optical_flow_cache[cache_key] = flow

        adjusted_annotations = []

        for ann in annotations:
            # バウンディングボックスの中心点を計算
            bbox = ann["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # 中心点でのフローを取得
            if 0 <= int(cy) < flow.shape[0] and 0 <= int(cx) < flow.shape[1]:
                dx, dy = flow[int(cy), int(cx)]

                # 移動量が閾値以内の場合のみ適用
                movement = np.sqrt(dx**2 + dy**2)
                if movement <= max_movement:
                    # 新しい位置を計算
                    new_bbox = [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

                    # 信頼度を計算
                    confidence = self._calculate_tracking_confidence(flow, bbox, movement)

                    adjusted_ann = ann.copy()
                    adjusted_ann["bbox"] = new_bbox
                    adjusted_ann["confidence"] = confidence
                    adjusted_ann["movement"] = float(movement)

                    adjusted_annotations.append(adjusted_ann)

        return adjusted_annotations

    def _calculate_frame_difference(self, frame_idx1: int, frame_idx2: int) -> float:
        """フレーム間の差分を計算"""
        if not self.interface:
            return 0.0

        try:
            frame1 = self.interface.get_frame(frame_idx1)
            frame2 = self.interface.get_frame(frame_idx2)

            # 差分を計算
            diff = cv2.absdiff(frame1, frame2)
            return np.mean(diff) / 255.0

        except Exception:
            return 1.0  # エラーの場合は最大差分

    def _calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """フレーム間の類似度を計算"""
        try:
            # ヒストグラム比較による類似度
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()

            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()

            # 相関係数を計算
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            return float(similarity)

        except Exception:
            return 0.0

    def _estimate_global_offset(
        self, source_frame_id: str, target_frame_id: str
    ) -> tuple[float, float]:
        """フレーム間のグローバルオフセットを推定"""
        # 簡易実装（実際にはより高度な手法を使用）
        return (0.0, 0.0)

    def _calculate_tracking_confidence(
        self, flow: np.ndarray, bbox: list[float], movement: float
    ) -> float:
        """トラッキングの信頼度を計算"""
        # 移動量が小さいほど信頼度が高い
        movement_confidence = max(0, 1 - movement / 50.0)

        # フロー領域の一貫性を確認
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi_flow = flow[y1:y2, x1:x2]

        if roi_flow.size > 0:
            # フローの分散が小さいほど信頼度が高い
            flow_variance = np.var(roi_flow)
            consistency_confidence = max(0, 1 - flow_variance / 100.0)
        else:
            consistency_confidence = 0.5

        # 総合的な信頼度
        confidence = (movement_confidence + consistency_confidence) / 2

        return float(confidence)

    def _calculate_confidence(self, annotations: list[dict[str, Any]]) -> float:
        """アノテーションセットの信頼度を計算"""
        if not annotations:
            return 0.0

        confidences = [ann.get("confidence", 1.0) for ann in annotations]
        return float(np.mean(confidences))

    def save_template(self, template: dict[str, Any], filepath: str):
        """テンプレートをファイルに保存"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

    def load_template(self, filepath: str) -> dict[str, Any]:
        """テンプレートをファイルから読み込み"""
        with open(filepath, encoding="utf-8") as f:
            template = json.load(f)

        self.templates[template["id"]] = template
        return template

    def clear_cache(self):
        """キャッシュをクリア"""
        self.optical_flow_cache.clear()

    def get_statistics(self) -> dict[str, Any]:
        """統計情報を取得"""
        return {
            "template_count": len(self.templates),
            "cache_size": len(self.optical_flow_cache),
            "cache_memory_mb": sum(flow.nbytes for flow in self.optical_flow_cache.values())
            / (1024 * 1024),
        }
