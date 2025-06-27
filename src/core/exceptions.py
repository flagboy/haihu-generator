"""
カスタム例外クラス定義

システム全体で使用される統一された例外クラスを定義
"""

from typing import Any


class BaseHaihuError(Exception):
    """
    Haihu Generatorの基本例外クラス

    すべてのカスタム例外はこのクラスを継承する
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """
        例外の初期化

        Args:
            message: エラーメッセージ
            details: エラーの詳細情報
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """文字列表現"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ファイル操作関連の例外
class FileOperationError(BaseHaihuError):
    """ファイル操作エラーの基本クラス"""

    pass


class FileReadError(FileOperationError):
    """ファイル読み込みエラー"""

    pass


class FileWriteError(FileOperationError):
    """ファイル書き込みエラー"""

    pass


class FileFormatError(FileOperationError):
    """ファイルフォーマットエラー"""

    pass


# 動画処理関連の例外
class VideoProcessingError(BaseHaihuError):
    """動画処理エラーの基本クラス"""

    pass


class VideoOpenError(VideoProcessingError):
    """動画ファイルを開けないエラー"""

    pass


class VideoCodecError(VideoProcessingError):
    """動画コーデックエラー"""

    pass


class FrameExtractionError(VideoProcessingError):
    """フレーム抽出エラー"""

    pass


class InvalidFrameError(VideoProcessingError):
    """無効なフレームエラー"""

    pass


# 設定関連の例外
class ConfigurationError(BaseHaihuError):
    """設定エラーの基本クラス"""

    pass


class InvalidConfigError(ConfigurationError):
    """無効な設定エラー"""

    pass


class MissingConfigError(ConfigurationError):
    """設定が見つからないエラー"""

    pass


# AI/モデル関連の例外
class ModelError(BaseHaihuError):
    """モデルエラーの基本クラス"""

    pass


class ModelLoadError(ModelError):
    """モデル読み込みエラー"""

    pass


class ModelInferenceError(ModelError):
    """モデル推論エラー"""

    pass


# ゲームロジック関連の例外
class GameLogicError(BaseHaihuError):
    """ゲームロジックエラーの基本クラス"""

    pass


class InvalidGameStateError(GameLogicError):
    """無効なゲーム状態エラー"""

    pass


class InvalidActionError(GameLogicError):
    """無効なアクションエラー"""

    pass


# データベース関連の例外
class DatabaseError(BaseHaihuError):
    """データベースエラーの基本クラス"""

    pass


class DatabaseConnectionError(DatabaseError):
    """データベース接続エラー"""

    pass


class DatabaseQueryError(DatabaseError):
    """データベースクエリエラー"""

    pass


# バリデーション関連の例外
class ValidationError(BaseHaihuError):
    """バリデーションエラーの基本クラス"""

    pass


class DataValidationError(ValidationError):
    """データバリデーションエラー"""

    pass


class FormatValidationError(ValidationError):
    """フォーマットバリデーションエラー"""

    pass
