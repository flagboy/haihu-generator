"""
統一されたエラーハンドリングのためのカスタム例外クラス

このモジュールは、プロジェクト全体で使用される
統一されたエラーハンドリングのための例外クラスを提供します。
"""

from typing import Any


class HaihuGeneratorError(Exception):
    """haihu-generatorの基底例外クラス"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Args:
            message: エラーメッセージ
            error_code: エラーコード（例: "E001", "VIDEO_001"）
            details: 追加の詳細情報
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """エラー情報を辞書形式で返す"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


# 動画処理関連のエラー
class VideoProcessingError(HaihuGeneratorError):
    """動画処理中のエラー"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="VIDEO_001", details=details)


class VideoNotFoundError(VideoProcessingError):
    """動画ファイルが見つからない"""

    def __init__(self, video_path: str):
        super().__init__(
            f"動画ファイルが見つかりません: {video_path}",
            details={"video_path": video_path},
        )
        self.error_code = "VIDEO_002"


class InvalidVideoFormatError(VideoProcessingError):
    """無効な動画フォーマット"""

    def __init__(self, video_path: str, format: str | None = None):
        super().__init__(
            f"無効な動画フォーマット: {video_path}",
            details={"video_path": video_path, "format": format},
        )
        self.error_code = "VIDEO_003"


# AI/ML関連のエラー
class ModelError(HaihuGeneratorError):
    """モデル関連のエラー"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="MODEL_001", details=details)


class ModelNotFoundError(ModelError):
    """モデルファイルが見つからない"""

    def __init__(self, model_path: str):
        super().__init__(
            f"モデルファイルが見つかりません: {model_path}",
            details={"model_path": model_path},
        )
        self.error_code = "MODEL_002"


class ModelLoadError(ModelError):
    """モデルの読み込みエラー"""

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"モデルの読み込みに失敗しました: {model_path}",
            details={"model_path": model_path, "reason": reason},
        )
        self.error_code = "MODEL_003"


class InferenceError(ModelError):
    """推論エラー"""

    def __init__(self, message: str, model_name: str | None = None):
        super().__init__(
            f"推論中にエラーが発生しました: {message}",
            details={"model_name": model_name},
        )
        self.error_code = "MODEL_004"


# データセット関連のエラー
class DatasetError(HaihuGeneratorError):
    """データセット関連のエラー"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="DATA_001", details=details)


class DatasetNotFoundError(DatasetError):
    """データセットが見つからない"""

    def __init__(self, dataset_name: str):
        super().__init__(
            f"データセットが見つかりません: {dataset_name}",
            details={"dataset_name": dataset_name},
        )
        self.error_code = "DATA_002"


class InvalidDataFormatError(DatasetError):
    """無効なデータフォーマット"""

    def __init__(self, message: str, expected_format: str | None = None):
        super().__init__(
            message,
            details={"expected_format": expected_format},
        )
        self.error_code = "DATA_003"


# 検証関連のエラー
class ValidationError(HaihuGeneratorError):
    """検証エラー"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="VAL_001", details=details)


class QualityValidationError(ValidationError):
    """品質検証エラー"""

    def __init__(self, message: str, quality_score: float | None = None):
        super().__init__(
            message,
            details={"quality_score": quality_score},
        )
        self.error_code = "VAL_002"


# 設定関連のエラー
class ConfigurationError(HaihuGeneratorError):
    """設定エラー"""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(
            message,
            error_code="CONFIG_001",
            details={"config_key": config_key},
        )


# 学習関連のエラー
class TrainingError(HaihuGeneratorError):
    """学習エラー"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, error_code="TRAIN_001", details=details)


class InsufficientDataError(TrainingError):
    """データ不足エラー"""

    def __init__(self, required: int, actual: int):
        super().__init__(
            f"学習に必要なデータが不足しています。必要: {required}, 実際: {actual}",
            details={"required": required, "actual": actual},
        )
        self.error_code = "TRAIN_002"


# パイプライン関連のエラー
class PipelineError(HaihuGeneratorError):
    """パイプライン実行エラー"""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(
            message,
            error_code="PIPE_001",
            details={"stage": stage},
        )


# Web API関連のエラー
class APIError(HaihuGeneratorError):
    """API関連のエラー"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code="API_001", details=details)
        self.status_code = status_code


class BadRequestError(APIError):
    """不正なリクエスト"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=400, details=details)
        self.error_code = "API_400"


class NotFoundError(APIError):
    """リソースが見つからない"""

    def __init__(self, resource: str):
        super().__init__(
            f"リソースが見つかりません: {resource}",
            status_code=404,
            details={"resource": resource},
        )
        self.error_code = "API_404"


class UnauthorizedError(APIError):
    """認証エラー"""

    def __init__(self, message: str = "認証が必要です"):
        super().__init__(message, status_code=401)
        self.error_code = "API_401"


class ForbiddenError(APIError):
    """権限エラー"""

    def __init__(self, message: str = "アクセスが拒否されました"):
        super().__init__(message, status_code=403)
        self.error_code = "API_403"


# リトライ可能なエラー
class RetryableError(HaihuGeneratorError):
    """リトライ可能なエラーを示す基底クラス"""

    def __init__(
        self,
        message: str,
        max_retries: int = 3,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code="RETRY_001", details=details)
        self.max_retries = max_retries
        self.retry_after = retry_after  # 秒単位
