"""
エラーハンドリングを支援するデコレータとユーティリティ

このモジュールは、統一されたエラーハンドリングを実現するための
デコレータとユーティリティ関数を提供します。
"""

import functools
import logging
import traceback
from collections.abc import Callable
from typing import Any

from .exceptions import HaihuGeneratorError, RetryableError


def handle_errors(
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
    default_return: Any = None,
    log_level: str = "error",
    raise_on: tuple[type[Exception], ...] | None = None,
    retry_on: tuple[type[Exception], ...] | None = None,
    max_retries: int = 3,
    custom_handler: Callable[[Exception], Any] | None = None,
) -> Callable:
    """
    エラーハンドリングを統一するデコレータ

    Args:
        exceptions: キャッチする例外のタプル
        default_return: エラー時のデフォルト戻り値
        log_level: ログレベル（"error", "warning", "info", "debug"）
        raise_on: これらの例外は再発生させる
        retry_on: これらの例外はリトライする
        max_retries: 最大リトライ回数
        custom_handler: カスタムエラーハンドラー

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # ロガーを取得
            if args and hasattr(args[0], "logger"):
                logger = args[0].logger
            else:
                logger = logging.getLogger(func.__module__)

            retries = 0
            last_error = None

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e

                    # 再発生させるべき例外かチェック
                    if raise_on and isinstance(e, raise_on):
                        raise

                    # カスタムハンドラーがある場合は実行
                    if custom_handler:
                        result = custom_handler(e)
                        if result is not None:
                            return result

                    # リトライ可能な例外かチェック
                    if retry_on and isinstance(e, retry_on) and retries < max_retries:
                        retries += 1
                        logger.warning(
                            f"{func.__name__}でリトライ可能なエラーが発生しました "
                            f"(試行 {retries}/{max_retries}): {str(e)}"
                        )
                        continue

                    # エラーログを記録
                    error_info = _format_error_info(e, func)
                    getattr(logger, log_level)(error_info)

                    # HaihuGeneratorErrorの場合は詳細情報も記録
                    if isinstance(e, HaihuGeneratorError):
                        logger.debug(f"エラー詳細: {e.to_dict()}")

                    # デフォルト値を返すか例外を再発生
                    if default_return is not None:
                        return default_return
                    else:
                        raise

            # リトライ回数を超えた場合
            if last_error:
                logger.error(f"{func.__name__}で最大リトライ回数を超えました: {str(last_error)}")
                if default_return is not None:
                    return default_return
                else:
                    raise last_error

        return wrapper

    return decorator


def _format_error_info(exception: Exception, func: Callable) -> str:
    """エラー情報をフォーマット"""
    error_type = type(exception).__name__
    error_msg = str(exception)
    func_name = func.__name__
    module_name = func.__module__

    # スタックトレースの最後の部分を取得
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    relevant_tb = "".join(tb_lines[-3:])  # 最後の3行

    return (
        f"{module_name}.{func_name}でエラーが発生しました\n"
        f"エラータイプ: {error_type}\n"
        f"エラーメッセージ: {error_msg}\n"
        f"スタックトレース:\n{relevant_tb}"
    )


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    logger: logging.Logger | None = None,
    **kwargs,
) -> Any:
    """
    関数を安全に実行する

    Args:
        func: 実行する関数
        *args: 関数の位置引数
        default: エラー時のデフォルト値
        exceptions: キャッチする例外
        logger: ログ出力用のロガー
        **kwargs: 関数のキーワード引数

    Returns:
        関数の戻り値またはデフォルト値
    """
    try:
        return func(*args, **kwargs)
    except exceptions as e:
        if logger:
            logger.error(f"{func.__name__}の実行中にエラー: {str(e)}")
        return default


def convert_exception(
    from_exceptions: tuple[type[Exception], ...],
    to_exception: type[Exception],
    message_format: str | None = None,
) -> Callable:
    """
    特定の例外を別の例外に変換するデコレータ

    Args:
        from_exceptions: 変換元の例外
        to_exception: 変換先の例外
        message_format: メッセージフォーマット（{original}で元のメッセージを参照）

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except from_exceptions as e:
                message = message_format.format(original=str(e)) if message_format else str(e)

                # 詳細情報を保持して新しい例外を作成
                if issubclass(to_exception, HaihuGeneratorError):
                    raise to_exception(
                        message,
                        details={
                            "original_error": type(e).__name__,
                            "original_message": str(e),
                        },
                    ) from e
                else:
                    raise to_exception(message) from e

        return wrapper

    return decorator


def log_exceptions(
    logger_name: str | None = None,
    log_level: str = "error",
    include_traceback: bool = True,
) -> Callable:
    """
    例外をログに記録するデコレータ（例外は再発生）

    Args:
        logger_name: ロガー名（Noneの場合は関数のモジュール名を使用）
        log_level: ログレベル
        include_traceback: トレースバックを含めるか

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(logger_name or func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = f"{func.__name__}でエラーが発生: {str(e)}"

                if include_traceback:
                    message += f"\n{traceback.format_exc()}"

                getattr(logger, log_level)(message)
                raise

        return wrapper

    return decorator


def retry_on_error(
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: logging.Logger | None = None,
) -> Callable:
    """
    エラー時にリトライするデコレータ

    Args:
        exceptions: リトライする例外
        max_retries: 最大リトライ回数
        delay: 初回リトライまでの遅延（秒）
        backoff: リトライごとの遅延倍率
        logger: ログ出力用のロガー

    Returns:
        デコレートされた関数
    """
    import time

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__}でエラーが発生しました。"
                            f"リトライします (試行 {attempt + 1}/{max_retries}): {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__}で最大リトライ回数に達しました: {str(e)}")

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def validate_not_none(*param_names: str) -> Callable:
    """
    指定されたパラメータがNoneでないことを検証するデコレータ

    Args:
        *param_names: 検証するパラメータ名

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 関数のシグネチャを取得
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 指定されたパラメータをチェック
            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is None:
                        raise ValueError(
                            f"{func.__name__}のパラメータ '{param_name}' はNoneにできません"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ErrorContext:
    """
    エラーコンテキストを管理するコンテキストマネージャー

    使用例:
        with ErrorContext("データ処理中", logger=logger):
            process_data()
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger | None = None,
        raise_on_error: bool = True,
        default_return: Any = None,
    ):
        """
        Args:
            operation: 実行中の操作の説明
            logger: ログ出力用のロガー
            raise_on_error: エラー時に例外を再発生させるか
            default_return: エラー時のデフォルト戻り値
        """
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error
        self.default_return = default_return
        self.error_occurred = False
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.error = exc_val

            # エラー情報をログに記録
            self.logger.error(
                f"{self.operation}中にエラーが発生しました: {exc_val}",
                exc_info=True,
            )

            # HaihuGeneratorErrorの場合は詳細情報も記録
            if isinstance(exc_val, HaihuGeneratorError):
                self.logger.debug(f"エラー詳細: {exc_val.to_dict()}")

            # raise_on_errorがFalseの場合は例外を抑制
            return not self.raise_on_error

        return True
