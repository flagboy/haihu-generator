"""
モニタリングデコレーター

関数やメソッドの実行を自動的に監視するためのデコレーター
"""

import functools
import time
from collections.abc import Callable
from typing import Any

from .error_tracker import error_tracker
from .logger import get_structured_logger
from .metrics import performance_tracker


def monitor_performance(
    operation_name: str | None = None, track_items: bool = False, log_errors: bool = True
) -> Callable:
    """
    パフォーマンスモニタリングデコレーター

    Args:
        operation_name: オペレーション名（Noneの場合は関数名を使用）
        track_items: アイテム数を追跡するか
        log_errors: エラーをログに記録するか

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        logger = get_structured_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            success = True
            items_processed = 0
            error = None

            try:
                result = func(*args, **kwargs)

                # アイテム数の抽出
                if track_items:
                    if isinstance(result, list | tuple):
                        items_processed = len(result)
                    elif isinstance(result, dict) and "items" in result:
                        items_processed = len(result["items"])
                    elif hasattr(result, "__len__"):
                        items_processed = len(result)

                return result

            except Exception as e:
                success = False
                error = e

                if log_errors:
                    # エラーを追跡
                    error_tracker.track_error(
                        e,
                        operation=name,
                        context={
                            "function": func.__name__,
                            "module": func.__module__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                    )

                raise

            finally:
                duration = time.time() - start_time

                # パフォーマンスを記録
                performance_tracker.track_operation(
                    operation=name,
                    duration=duration,
                    success=success,
                    items_processed=items_processed if track_items else None,
                )

                # ログ出力
                if success:
                    logger.info(
                        f"Operation completed: {name}",
                        operation=name,
                        duration=duration,
                        success=success,
                        items_processed=items_processed if track_items else None,
                    )
                else:
                    logger.error(
                        f"Operation failed: {name}",
                        operation=name,
                        duration=duration,
                        success=success,
                        error=str(error) if error else None,
                    )

        return wrapper

    return decorator


def monitor_batch_processing(func: Callable) -> Callable:
    """
    バッチ処理モニタリングデコレーター

    バッチサイズ、処理時間、成功率を自動的に追跡

    Args:
        func: バッチ処理関数（batch_size引数を持つことが期待される）

    Returns:
        デコレートされた関数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        batch_size = kwargs.get("batch_size", 0)

        # argsからbatch_sizeを探す
        if not batch_size and args:
            # 最初の引数がselfの場合をスキップ
            arg_offset = 1 if hasattr(args[0], "__class__") else 0
            if len(args) > arg_offset:
                first_arg = args[arg_offset]
                if isinstance(first_arg, int):
                    batch_size = first_arg
                elif isinstance(first_arg, list | tuple):
                    batch_size = len(first_arg)

        start_time = time.time()
        success_count = 0
        error_count = 0

        try:
            result = func(*args, **kwargs)

            # 結果から成功/エラー数を抽出
            if isinstance(result, dict):
                success_count = result.get("success_count", batch_size)
                error_count = result.get("error_count", 0)
            elif isinstance(result, list | tuple):
                success_count = len(result)
            else:
                success_count = batch_size

            return result

        except Exception:
            error_count = batch_size
            raise

        finally:
            processing_time = time.time() - start_time

            # バッチ処理メトリクスを記録
            performance_tracker.track_batch_processing(
                batch_size=batch_size,
                processing_time=processing_time,
                success_count=success_count,
                error_count=error_count,
            )

    return wrapper


def monitor_memory_usage(func: Callable) -> Callable:
    """
    メモリ使用量モニタリングデコレーター

    関数実行前後のメモリ使用量を記録

    Args:
        func: 監視する関数

    Returns:
        デコレートされた関数
    """
    import psutil

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        process = psutil.Process()

        # 実行前のメモリ使用量
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 実行後のメモリ使用量
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = memory_after - memory_before

            # メトリクスを記録
            performance_tracker.metrics.gauge(f"{func.__name__}_memory_mb", memory_after)
            performance_tracker.metrics.record(f"{func.__name__}_memory_diff_mb", memory_diff)

            # 大きなメモリ増加があった場合は警告
            if memory_diff > 100:  # 100MB以上
                logger = get_structured_logger(func.__module__)
                logger.warning(
                    f"Large memory increase in {func.__name__}",
                    function=func.__name__,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_diff_mb=memory_diff,
                )

    return wrapper


def monitor_critical_section(section_name: str, timeout_seconds: float = 30.0) -> Callable:
    """
    クリティカルセクションモニタリングデコレーター

    重要な処理区間の実行時間を監視し、タイムアウト時に警告

    Args:
        section_name: セクション名
        timeout_seconds: タイムアウト時間（秒）

    Returns:
        デコレートされた関数
    """

    def decorator(func: Callable) -> Callable:
        logger = get_structured_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time

                # メトリクスを記録
                performance_tracker.metrics.record(
                    f"critical_section_{section_name}_duration", duration
                )

                # タイムアウトチェック
                if duration > timeout_seconds:
                    logger.warning(
                        f"Critical section timeout: {section_name}",
                        section=section_name,
                        duration=duration,
                        timeout=timeout_seconds,
                        function=func.__name__,
                    )

        return wrapper

    return decorator
