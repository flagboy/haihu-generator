"""
ログ設定モジュール
"""

import sys
from pathlib import Path

from loguru import logger

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
