"""
統一されたエラーハンドリングシステム

例外の捕捉、ログ記録、ユーザーへの通知を統一的に処理
"""

import logging
import traceback
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from .exceptions import BaseHaihuError

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ErrorHandler:
    """統一されたエラーハンドラー"""

    def __init__(self, log_errors: bool = True, raise_errors: bool = True):
        """
        エラーハンドラーの初期化

        Args:
            log_errors: エラーをログに記録するか
            raise_errors: エラーを再発生させるか
        """
        self.log_errors = log_errors
        self.raise_errors = raise_errors

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        user_message: str | None = None,
    ) -> None:
        """
        エラーを処理

        Args:
            error: 発生した例外
            context: エラーコンテキスト情報
            user_message: ユーザー向けメッセージ
        """
        context = context or {}

        # エラー情報を構築
        error_info = {"type": type(error).__name__, "message": str(error), "context": context}

        # BaseHaihuErrorの場合は詳細情報を追加
        if isinstance(error, BaseHaihuError):
            error_info["details"] = error.details

        # トレースバック情報を追加
        error_info["traceback"] = traceback.format_exc()

        # ログに記録
        if self.log_errors:
            self._log_error(error, error_info, user_message)

        # エラーを再発生
        if self.raise_errors:
            raise

    def _log_error(
        self, error: Exception, error_info: dict[str, Any], user_message: str | None
    ) -> None:
        """エラーをログに記録"""
        if isinstance(error, BaseHaihuError):
            # カスタム例外は詳細にログ
            logger.error(
                f"{error_info['type']}: {error_info['message']}",
                extra={
                    "error_details": error.details,
                    "context": error_info["context"],
                    "user_message": user_message,
                },
            )
        else:
            # 標準例外
            logger.error(
                f"{error_info['type']}: {error_info['message']}",
                extra={
                    "context": error_info["context"],
                    "user_message": user_message,
                    "traceback": error_info["traceback"],
                },
            )


def with_error_handling(
    handler: ErrorHandler | None = None,
    context_func: Callable[..., dict[str, Any]] | None = None,
    user_message: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    エラーハンドリングデコレーター

    Args:
        handler: 使用するエラーハンドラー（Noneの場合はデフォルト）
        context_func: コンテキスト情報を生成する関数
        user_message: ユーザー向けメッセージ

    Returns:
        デコレーター関数
    """
    if handler is None:
        handler = ErrorHandler()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # コンテキスト情報を生成
                context = {}
                if context_func:
                    context = context_func(*args, **kwargs)

                # 関数情報を追加
                context.update({"function": func.__name__, "module": func.__module__})

                # エラーを処理
                handler.handle_error(e, context, user_message)
                raise  # デコレーターは常に例外を再発生させる

        return wrapper

    return decorator


def create_context(**kwargs) -> dict[str, Any]:
    """
    エラーコンテキストを作成するヘルパー関数

    Args:
        **kwargs: コンテキストに含める情報

    Returns:
        コンテキスト辞書
    """
    context = {}

    for key, value in kwargs.items():
        # パスオブジェクトは文字列に変換
        if isinstance(value, Path):
            context[key] = str(value)
        # 大きなオブジェクトは要約
        elif hasattr(value, "__len__") and len(str(value)) > 1000:
            context[key] = f"<{type(value).__name__} with {len(value)} items>"
        else:
            context[key] = value

    return context


# グローバルエラーハンドラー
_global_handler = ErrorHandler()


def handle_error(
    error: Exception, context: dict[str, Any] | None = None, user_message: str | None = None
) -> None:
    """
    グローバルエラーハンドラーを使用してエラーを処理

    Args:
        error: 発生した例外
        context: エラーコンテキスト情報
        user_message: ユーザー向けメッセージ
    """
    _global_handler.handle_error(error, context, user_message)
