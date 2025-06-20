"""
ラベリングサービス

ラベル付けのビジネスロジックを提供
"""

from typing import Any

from ......utils.logger import LoggerMixin
from ...scene_labeling_session import SceneLabelingSession
from ..middleware.error_handler import ValidationError
from ..schemas.request_schemas import BatchLabelRequest


class LabelingService(LoggerMixin):
    """ラベリングサービス"""

    def label_frame(
        self,
        session: SceneLabelingSession,
        frame_number: int,
        label: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        フレームにラベルを付ける

        Args:
            session: セッション
            frame_number: フレーム番号
            label: ラベル
            confidence: 信頼度
            metadata: メタデータ

        Returns:
            ラベル付け結果

        Raises:
            ValidationError: 無効な入力
        """
        # フレーム番号の検証
        if frame_number < 0 or frame_number >= session.total_frames:
            raise ValidationError(
                f"無効なフレーム番号: {frame_number} (範囲: 0-{session.total_frames - 1})"
            )

        # ラベルの検証
        if not label:
            raise ValidationError("ラベルは必須です")

        # 信頼度の検証
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValidationError("信頼度は0.0から1.0の範囲である必要があります")

        # ラベル付け実行
        try:
            success = session.label_frame(
                frame_number, label, confidence=confidence, metadata=metadata
            )

            if success:
                self.logger.info(
                    f"フレーム {frame_number} にラベル '{label}' を付けました "
                    f"(信頼度: {confidence})"
                )
                return {
                    "success": True,
                    "frame_number": frame_number,
                    "label": label,
                    "confidence": confidence,
                }
            else:
                raise ValidationError("ラベル付けに失敗しました")

        except Exception as e:
            self.logger.error(f"ラベル付けエラー: {e}")
            raise ValidationError(f"ラベル付けに失敗しました: {str(e)}") from e

    def batch_label_frames(
        self, session: SceneLabelingSession, batch_request: BatchLabelRequest
    ) -> dict[str, Any]:
        """
        複数フレームに一括でラベルを付ける

        Args:
            session: セッション
            batch_request: バッチラベル付けリクエスト

        Returns:
            バッチ処理結果
        """
        results = []
        success_count = 0
        error_count = 0

        for label_request in batch_request.labels:
            try:
                result = self.label_frame(
                    session,
                    label_request.frame_number,
                    label_request.label,
                    label_request.confidence,
                    label_request.metadata,
                )
                results.append(result)
                success_count += 1
            except ValidationError as e:
                results.append(
                    {
                        "success": False,
                        "frame_number": label_request.frame_number,
                        "error": str(e),
                    }
                )
                error_count += 1
            except Exception as e:
                self.logger.error(f"バッチラベル付けエラー: {e}")
                results.append(
                    {
                        "success": False,
                        "frame_number": label_request.frame_number,
                        "error": "内部エラー",
                    }
                )
                error_count += 1

        return {
            "success": error_count == 0,
            "results": results,
            "summary": {
                "total": len(batch_request.labels),
                "success": success_count,
                "error": error_count,
            },
        }

    def get_label_statistics(self, session: SceneLabelingSession) -> dict[str, Any]:
        """
        ラベル統計を取得

        Args:
            session: セッション

        Returns:
            統計情報
        """
        label_counts = {}
        labeled_frames = 0
        confidence_sum = 0.0
        confidence_count = 0

        for frame_number in range(session.total_frames):
            frame_info = session.get_frame_info(frame_number)
            label = frame_info.get("label")

            if label:
                labeled_frames += 1
                label_counts[label] = label_counts.get(label, 0) + 1

                confidence = frame_info.get("confidence")
                if confidence is not None:
                    confidence_sum += confidence
                    confidence_count += 1

        return {
            "total_frames": session.total_frames,
            "labeled_frames": labeled_frames,
            "unlabeled_frames": session.total_frames - labeled_frames,
            "progress": labeled_frames / session.total_frames if session.total_frames > 0 else 0,
            "label_distribution": label_counts,
            "average_confidence": confidence_sum / confidence_count
            if confidence_count > 0
            else None,
        }
