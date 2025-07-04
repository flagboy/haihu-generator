"""
ログ設定モジュール

モニタリングシステムと統合されたログ設定を提供
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

# モニタリングシステムとの統合は遅延インポート
from .config import ConfigManager


def setup_logger(config_manager: ConfigManager | None = None) -> None:
    """
    ログ設定を初期化

    Args:
        config_manager: 設定管理インスタンス
    """
    if config_manager is None:
        config_manager = ConfigManager()

    # 既存のハンドラーを削除
    logger.remove()

    # ログ設定を取得
    log_config = config_manager.get_logging_config()
    level = log_config.get("level", "INFO")
    format_str = log_config.get(
        "format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    file_path = log_config.get("file_path", "logs/mahjong_system.log")
    rotation = log_config.get("rotation", "1 day")
    retention = log_config.get("retention", "30 days")

    # コンソール出力設定
    logger.add(
        sys.stdout, level=level, format=format_str, colorize=True, backtrace=True, diagnose=True
    )

    # ファイル出力設定
    log_file_path = Path(file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        file_path,
        level=level,
        format=format_str,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info("ログシステムが初期化されました")


def get_logger(name: str):
    """
    名前付きロガーを取得

    Args:
        name: ロガー名

    Returns:
        ロガーインスタンス
    """
    return logger.bind(name=name)


def setup_logging(config_manager: ConfigManager | None = None) -> None:
    """
    ログ設定を初期化（setup_loggerのエイリアス）

    Args:
        config_manager: 設定管理インスタンス
    """
    setup_logger(config_manager)


class LoggerMixin:
    """ログ機能を提供するミックスインクラス"""

    @property
    def logger(self):
        """クラス名をベースにしたロガーを取得"""
        return get_logger(self.__class__.__name__)


class MonitoredLogger:
    """モニタリング機能を統合したロガー"""

    def __init__(self, name: str):
        """
        Args:
            name: ロガー名
        """
        self.name = name
        self.loguru_logger = get_logger(name)
        # 構造化ロガーは遅延初期化
        self._structured_logger = None

    def _get_structured_logger(self):
        """構造化ロガーの遅延初期化"""
        if self._structured_logger is None:
            from ..monitoring import get_structured_logger

            self._structured_logger = get_structured_logger(self.name)
        return self._structured_logger

    @property
    def structured_logger(self):
        """構造化ロガープロパティ"""
        return self._get_structured_logger()

    def debug(self, message: str, **kwargs: Any) -> None:
        """デバッグログ"""
        self.loguru_logger.debug(message, **kwargs)
        self.structured_logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """情報ログ"""
        self.loguru_logger.info(message, **kwargs)
        self.structured_logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """警告ログ"""
        self.loguru_logger.warning(message, **kwargs)
        self.structured_logger.warning(message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """エラーログ"""
        if exc_info:
            self.loguru_logger.exception(message, **kwargs)
            self.structured_logger.error(message, exc_info=True, **kwargs)

            # エラートラッカーに記録（循環参照を避けるため条件付き）
            if not kwargs.get("_from_error_tracker", False):
                import sys

                exc_type, exc_value, _ = sys.exc_info()
                if exc_value:
                    import contextlib

                    with contextlib.suppress(Exception):
                        _get_error_tracker().track_error(
                            exc_value, operation=kwargs.get("operation", "unknown"), context=kwargs
                        )
        else:
            self.loguru_logger.error(message, **kwargs)
            self.structured_logger.error(message, **kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """重大エラーログ"""
        if exc_info:
            self.loguru_logger.exception(message, **kwargs)
            self.structured_logger.critical(message, exc_info=True, **kwargs)
        else:
            self.loguru_logger.critical(message, **kwargs)
            self.structured_logger.critical(message, **kwargs)

    def log_performance(self, operation: str, duration: float, **kwargs: Any) -> None:
        """パフォーマンスログ"""
        message = f"Performance: {operation} took {duration:.3f}s"
        self.info(message, operation=operation, duration=duration, **kwargs)
        self.structured_logger.log_performance(operation, duration, **kwargs)

        # グローバルメトリクスに記録
        from ..monitoring import get_global_metrics

        get_global_metrics().record(f"{operation}_duration_seconds", duration)

    def bind(self, **kwargs: Any) -> "MonitoredLogger":
        """コンテキスト情報をバインド"""
        new_logger = MonitoredLogger.__new__(MonitoredLogger)
        new_logger.name = self.name
        new_logger.loguru_logger = self.loguru_logger.bind(**kwargs)
        new_logger.structured_logger = self.structured_logger.bind(**kwargs)
        return new_logger


# グローバル関数の更新
def get_monitored_logger(name: str) -> MonitoredLogger:
    """モニタリング機能付きロガーを取得"""
    return MonitoredLogger(name)


# 遅延インポートでエラートラッカーを取得
def _get_error_tracker():
    """エラートラッカーを遅延インポート"""
    from ..monitoring import get_error_tracker

    return get_error_tracker()


class MonitoredLoggerMixin:
    """モニタリング機能付きログミックスイン"""

    @property
    def logger(self) -> MonitoredLogger:
        """モニタリング機能付きロガーを取得"""
        if not hasattr(self, "_monitored_logger"):
            self._monitored_logger = get_monitored_logger(self.__class__.__name__)
        return self._monitored_logger
