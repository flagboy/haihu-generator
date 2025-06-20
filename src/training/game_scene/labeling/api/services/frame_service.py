"""
フレーム処理サービス

フレームの取得、処理などのビジネスロジックを提供
"""

import base64
from typing import Any

import cv2
import numpy as np

from ......utils.logger import LoggerMixin
from ...scene_labeling_session import SceneLabelingSession
from ..middleware.error_handler import NotFoundError, ValidationError


class FrameService(LoggerMixin):
    """フレーム処理サービス"""

    def get_frame(self, session: SceneLabelingSession, frame_number: int) -> dict[str, Any]:
        """
        フレームを取得

        Args:
            session: セッション
            frame_number: フレーム番号

        Returns:
            フレーム情報

        Raises:
            ValidationError: 無効なフレーム番号
            NotFoundError: フレームが見つからない
        """
        # フレーム番号の検証
        if frame_number < 0 or frame_number >= session.total_frames:
            raise ValidationError(
                f"無効なフレーム番号: {frame_number} (範囲: 0-{session.total_frames - 1})"
            )

        # フレーム画像を取得
        frame_image = session.get_frame(frame_number)
        if frame_image is None:
            raise NotFoundError("フレーム", f"番号 {frame_number}")

        # フレーム情報を取得
        frame_info = session.get_frame_info(frame_number)

        # 画像をBase64エンコード
        image_data = self._encode_image(frame_image)

        return {
            "frame_number": frame_number,
            "timestamp": frame_number / session.fps if hasattr(session, "fps") else 0.0,
            "image": image_data,
            "label": frame_info.get("label"),
            "confidence": frame_info.get("confidence"),
            "is_labeled": frame_info.get("label") is not None,
            "metadata": frame_info.get("metadata", {}),
        }

    def get_next_unlabeled_frame(
        self, session: SceneLabelingSession, current_frame: int | None = None
    ) -> dict[str, Any] | None:
        """
        次の未ラベルフレームを取得

        Args:
            session: セッション
            current_frame: 現在のフレーム番号

        Returns:
            フレーム情報またはNone
        """
        start_frame = current_frame + 1 if current_frame is not None else 0

        for frame_number in range(start_frame, session.total_frames):
            frame_info = session.get_frame_info(frame_number)
            if not frame_info.get("label"):
                return self.get_frame(session, frame_number)

        # 最初から探す
        if start_frame > 0:
            for frame_number in range(0, start_frame):
                frame_info = session.get_frame_info(frame_number)
                if not frame_info.get("label"):
                    return self.get_frame(session, frame_number)

        return None

    def get_uncertainty_frame(
        self, session: SceneLabelingSession, threshold: float = 0.5
    ) -> dict[str, Any] | None:
        """
        不確実性の高いフレームを取得

        Args:
            session: セッション
            threshold: 不確実性の閾値

        Returns:
            フレーム情報またはNone
        """
        # 自動ラベリングされたフレームから不確実性の高いものを探す
        uncertain_frames = []

        for frame_number in range(session.total_frames):
            frame_info = session.get_frame_info(frame_number)
            confidence = frame_info.get("confidence")

            if confidence is not None and confidence < threshold:
                uncertain_frames.append((frame_number, confidence))

        if not uncertain_frames:
            return None

        # 最も不確実性の高いフレームを選択
        uncertain_frames.sort(key=lambda x: x[1])
        frame_number = uncertain_frames[0][0]

        return self.get_frame(session, frame_number)

    def get_frame_segments(self, session: SceneLabelingSession) -> list[dict[str, Any]]:
        """
        フレームセグメントを取得

        Args:
            session: セッション

        Returns:
            セグメント情報のリスト
        """
        segments = []
        current_segment = None

        for frame_number in range(session.total_frames):
            frame_info = session.get_frame_info(frame_number)
            label = frame_info.get("label")

            if label:
                if current_segment is None or current_segment["label"] != label:
                    # 新しいセグメント開始
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = {
                        "label": label,
                        "start_frame": frame_number,
                        "end_frame": frame_number,
                        "frame_count": 1,
                    }
                else:
                    # 既存セグメントを拡張
                    current_segment["end_frame"] = frame_number
                    current_segment["frame_count"] += 1
            else:
                # ラベルなしフレームでセグメント終了
                if current_segment:
                    segments.append(current_segment)
                    current_segment = None

        # 最後のセグメントを追加
        if current_segment:
            segments.append(current_segment)

        return segments

    def _encode_image(self, image: np.ndarray) -> str:
        """
        画像をBase64エンコード

        Args:
            image: 画像データ

        Returns:
            Base64エンコードされた文字列
        """
        try:
            # JPEGエンコード
            _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Base64エンコード
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"
        except Exception as e:
            self.logger.error(f"画像エンコードエラー: {e}")
            raise ValidationError(f"画像のエンコードに失敗しました: {str(e)}") from e
