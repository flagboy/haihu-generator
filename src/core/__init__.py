"""
コアモジュール

例外クラスとエラーハンドリングシステムを提供
"""

from .error_handler import (
    ErrorHandler,
    create_context,
    handle_error,
    with_error_handling,
)
from .exceptions import (
    BaseHaihuError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseQueryError,
    DataValidationError,
    FileFormatError,
    FileOperationError,
    FileReadError,
    FileWriteError,
    FormatValidationError,
    FrameExtractionError,
    GameLogicError,
    InvalidActionError,
    InvalidConfigError,
    InvalidFrameError,
    InvalidGameStateError,
    MissingConfigError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ValidationError,
    VideoCodecError,
    VideoOpenError,
    VideoProcessingError,
)

__all__ = [
    # 例外クラス
    "BaseHaihuError",
    "FileOperationError",
    "FileReadError",
    "FileWriteError",
    "FileFormatError",
    "VideoProcessingError",
    "VideoOpenError",
    "VideoCodecError",
    "FrameExtractionError",
    "InvalidFrameError",
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "GameLogicError",
    "InvalidGameStateError",
    "InvalidActionError",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "ValidationError",
    "DataValidationError",
    "FormatValidationError",
    # エラーハンドラー
    "ErrorHandler",
    "with_error_handling",
    "handle_error",
    "create_context",
]
