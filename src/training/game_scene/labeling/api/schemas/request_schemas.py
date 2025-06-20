"""
リクエストスキーマ定義

APIリクエストのバリデーションとシリアライゼーション
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CreateSessionRequest:
    """セッション作成リクエスト"""

    video_path: str
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """バリデーション"""
        errors = []
        if not self.video_path:
            errors.append("video_pathは必須です")
        return errors


@dataclass
class LabelRequest:
    """ラベル付けリクエスト"""

    frame_number: int
    label: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """バリデーション"""
        errors = []
        if self.frame_number < 0:
            errors.append("frame_numberは0以上である必要があります")
        if not self.label:
            errors.append("labelは必須です")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            errors.append("confidenceは0.0から1.0の範囲である必要があります")
        return errors


@dataclass
class BatchLabelRequest:
    """バッチラベル付けリクエスト"""

    labels: list[LabelRequest]
    overwrite: bool = False

    def validate(self) -> list[str]:
        """バリデーション"""
        errors = []
        if not self.labels:
            errors.append("labelsは必須です")
        for i, label in enumerate(self.labels):
            label_errors = label.validate()
            for error in label_errors:
                errors.append(f"labels[{i}]: {error}")
        return errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchLabelRequest":
        """辞書から生成"""
        labels = [
            LabelRequest(
                frame_number=label_data["frame_number"],
                label=label_data["label"],
                confidence=label_data.get("confidence"),
                metadata=label_data.get("metadata", {}),
            )
            for label_data in data.get("labels", [])
        ]
        return cls(labels=labels, overwrite=data.get("overwrite", False))


@dataclass
class AutoLabelRequest:
    """自動ラベリングリクエスト"""

    confidence_threshold: float = 0.8
    max_frames: int | None = None
    skip_labeled: bool = True

    def validate(self) -> list[str]:
        """バリデーション"""
        errors = []
        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append("confidence_thresholdは0.0から1.0の範囲である必要があります")
        if self.max_frames is not None and self.max_frames < 1:
            errors.append("max_framesは1以上である必要があります")
        return errors
