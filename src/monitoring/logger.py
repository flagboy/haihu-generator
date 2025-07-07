"""
構造化ログシステム

JSON形式での構造化ログ出力と、メタデータの自動付与を提供
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger


class StructuredLogger:
    """構造化ログクラス"""

    def __init__(
        self,
        name: str,
        log_dir: Path | None = None,
        level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """
        Args:
            name: ロガー名
            log_dir: ログディレクトリ
            level: ログレベル
            enable_console: コンソール出力を有効化
            enable_file: ファイル出力を有効化
            max_file_size: ログファイルの最大サイズ
            backup_count: バックアップファイル数
        """
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # structlogの設定
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_metadata,  # type: ignore[list-item]
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # 標準ロガーの設定
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        self.logger.handlers = []

        # コンソールハンドラー
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self._get_console_formatter())
            self.logger.addHandler(console_handler)

        # ファイルハンドラー
        if enable_file:
            from logging.handlers import RotatingFileHandler

            log_file = self.log_dir / f"{name}.json"
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(self._get_json_formatter())
            self.logger.addHandler(file_handler)

        # structlogラッパー
        self.structured_logger = structlog.get_logger(name)

    def _get_console_formatter(self) -> logging.Formatter:
        """コンソール用フォーマッタを取得"""
        return logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    def _get_json_formatter(self) -> jsonlogger.JsonFormatter:
        """JSON用フォーマッタを取得"""
        return jsonlogger.JsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"levelname": "level", "asctime": "timestamp"},
        )

    def _add_metadata(self, logger: Any, method_name: str, event_dict: dict) -> dict:
        """メタデータを追加"""
        # リクエストIDの追加（もし存在すれば）
        if hasattr(logger, "request_id"):
            event_dict["request_id"] = logger.request_id

        # セッションIDの追加（もし存在すれば）
        if hasattr(logger, "session_id"):
            event_dict["session_id"] = logger.session_id

        # ホスト名の追加
        import socket

        event_dict["hostname"] = socket.gethostname()

        # プロセスIDの追加
        import os

        event_dict["pid"] = os.getpid()

        return event_dict

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """コンテキスト情報をバインド"""
        bound_logger = self.structured_logger.bind(**kwargs)
        new_logger = StructuredLogger.__new__(StructuredLogger)
        new_logger.__dict__.update(self.__dict__)
        new_logger.structured_logger = bound_logger
        return new_logger

    def debug(self, msg: str, **kwargs: Any) -> None:
        """デバッグログ"""
        self.structured_logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """情報ログ"""
        self.structured_logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """警告ログ"""
        self.structured_logger.warning(msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """エラーログ"""
        if exc_info:
            kwargs["exc_info"] = sys.exc_info()
        self.structured_logger.error(msg, **kwargs)

    def critical(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """重大エラーログ"""
        if exc_info:
            kwargs["exc_info"] = sys.exc_info()
        self.structured_logger.critical(msg, **kwargs)

    def log_performance(
        self, operation: str, duration: float, success: bool = True, **kwargs: Any
    ) -> None:
        """パフォーマンスログ"""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration * 1000,
            success=success,
            performance=True,
            **kwargs,
        )

    def log_error_with_context(self, error: Exception, operation: str, **context: Any) -> None:
        """コンテキスト付きエラーログ"""
        self.error(
            f"Error in {operation}: {str(error)}",
            exc_info=True,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback=traceback.format_exc(),
            **context,
        )


# グローバルロガーインスタンス
_loggers: dict[str, StructuredLogger] = {}


def get_structured_logger(name: str, **kwargs: Any) -> StructuredLogger:
    """構造化ログインスタンスを取得"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, **kwargs)
    return _loggers[name]


# デフォルトロガー
default_logger = get_structured_logger("haihu_generator")
