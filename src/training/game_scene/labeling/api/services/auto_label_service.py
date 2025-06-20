"""
自動ラベリングサービス

AIモデルを使用した自動ラベリングのビジネスロジックを提供
"""

from typing import Any

from ......utils.logger import LoggerMixin
from ....core.game_scene_classifier import GameSceneClassifier
from ...scene_labeling_session import SceneLabelingSession
from ..middleware.error_handler import InternalError, ValidationError


class AutoLabelService(LoggerMixin):
    """自動ラベリングサービス"""

    def __init__(self, classifier: GameSceneClassifier | None = None):
        """
        初期化

        Args:
            classifier: 分類器インスタンス
        """
        self._classifier = classifier

    def auto_label_frames(
        self,
        session: SceneLabelingSession,
        confidence_threshold: float = 0.8,
        max_frames: int | None = None,
        skip_labeled: bool = True,
    ) -> dict[str, Any]:
        """
        フレームを自動的にラベル付け

        Args:
            session: セッション
            confidence_threshold: 信頼度の閾値
            max_frames: 最大処理フレーム数
            skip_labeled: ラベル付き済みフレームをスキップ

        Returns:
            自動ラベリング結果
        """
        # 検証
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValidationError("信頼度閾値は0.0から1.0の範囲である必要があります")

        # 分類器の確認
        if not self._classifier:
            raise InternalError("分類器が初期化されていません")

        # 処理対象フレームを収集
        target_frames = []
        for frame_number in range(session.total_frames):
            if max_frames and len(target_frames) >= max_frames:
                break

            frame_info = session.get_frame_info(frame_number)
            if skip_labeled and frame_info.get("label"):
                continue

            target_frames.append(frame_number)

        if not target_frames:
            return {
                "success": True,
                "message": "処理対象のフレームがありません",
                "summary": {"processed": 0, "labeled": 0, "skipped": 0},
            }

        # 自動ラベリング実行
        results = []
        labeled_count = 0
        skipped_count = 0
        error_count = 0

        self.logger.info(f"{len(target_frames)}フレームの自動ラベリングを開始")

        for i, frame_number in enumerate(target_frames):
            try:
                # フレーム画像を取得
                frame_image = session.get_frame(frame_number)
                if frame_image is None:
                    error_count += 1
                    continue

                # 分類実行
                prediction = self._classifier.predict(frame_image)
                label = prediction["label"]
                confidence = prediction["confidence"]

                # 信頼度チェック
                if confidence < confidence_threshold:
                    skipped_count += 1
                    results.append(
                        {
                            "frame_number": frame_number,
                            "label": label,
                            "confidence": confidence,
                            "labeled": False,
                            "reason": "信頼度が閾値未満",
                        }
                    )
                    continue

                # ラベル付け
                success = session.label_frame(
                    frame_number, label, confidence=confidence, metadata={"auto_labeled": True}
                )

                if success:
                    labeled_count += 1
                    results.append(
                        {
                            "frame_number": frame_number,
                            "label": label,
                            "confidence": confidence,
                            "labeled": True,
                        }
                    )
                else:
                    error_count += 1
                    results.append(
                        {
                            "frame_number": frame_number,
                            "error": "ラベル付けに失敗",
                            "labeled": False,
                        }
                    )

                # 進捗ログ
                if (i + 1) % 10 == 0:
                    self.logger.info(f"進捗: {i + 1}/{len(target_frames)}フレーム処理済み")

            except Exception as e:
                self.logger.error(f"フレーム {frame_number} の処理エラー: {e}")
                error_count += 1
                results.append({"frame_number": frame_number, "error": str(e), "labeled": False})

        # 結果サマリー
        summary = {
            "processed": len(target_frames),
            "labeled": labeled_count,
            "skipped": skipped_count,
            "error": error_count,
            "success_rate": labeled_count / len(target_frames) if target_frames else 0,
        }

        self.logger.info(
            f"自動ラベリング完了: {labeled_count}/{len(target_frames)}フレームにラベル付け"
        )

        return {
            "success": True,
            "summary": summary,
            "results": results,
        }

    def predict_frame(self, session: SceneLabelingSession, frame_number: int) -> dict[str, Any]:
        """
        単一フレームの予測

        Args:
            session: セッション
            frame_number: フレーム番号

        Returns:
            予測結果
        """
        # フレーム番号の検証
        if frame_number < 0 or frame_number >= session.total_frames:
            raise ValidationError(
                f"無効なフレーム番号: {frame_number} (範囲: 0-{session.total_frames - 1})"
            )

        # 分類器の確認
        if not self._classifier:
            raise InternalError("分類器が初期化されていません")

        # フレーム画像を取得
        frame_image = session.get_frame(frame_number)
        if frame_image is None:
            raise ValidationError(f"フレーム {frame_number} の取得に失敗しました")

        # 予測実行
        try:
            prediction = self._classifier.predict(frame_image)
            return {
                "frame_number": frame_number,
                "prediction": prediction,
            }
        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            raise InternalError(f"予測に失敗しました: {str(e)}") from e
