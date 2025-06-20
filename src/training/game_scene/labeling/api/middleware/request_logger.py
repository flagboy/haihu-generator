"""
リクエストログミドルウェア

APIリクエストとレスポンスのログを記録
"""

import time
from collections.abc import Callable
from functools import wraps

from flask import g, request

from ......utils.logger import get_logger

logger = get_logger(__name__)


def request_logger(f: Callable) -> Callable:
    """
    リクエストログデコレータ

    リクエストとレスポンスの情報をログに記録
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # リクエスト開始時刻を記録
        g.start_time = time.time()

        # リクエスト情報をログ
        logger.info(
            f"リクエスト開始: {request.method} {request.path} "
            f"- IP: {request.remote_addr} "
            f"- User-Agent: {request.headers.get('User-Agent', 'Unknown')}"
        )

        if request.method in ["POST", "PUT", "PATCH"] and request.is_json:
            # リクエストボディをログ（センシティブな情報は除外）
            body = request.get_json()
            if body:
                # パスワードなどのセンシティブな情報を除外
                safe_body = {k: v for k, v in body.items() if k not in ["password", "token"]}
                logger.debug(f"リクエストボディ: {safe_body}")

        try:
            # 実際の処理を実行
            response = f(*args, **kwargs)

            # レスポンス時間を計算
            elapsed_time = time.time() - g.start_time

            # レスポンス情報をログ
            status_code = response[1] if isinstance(response, tuple) else 200
            logger.info(
                f"リクエスト完了: {request.method} {request.path} "
                f"- ステータス: {status_code} "
                f"- 処理時間: {elapsed_time:.3f}秒"
            )

            return response

        except Exception as e:
            # エラー時のログ
            elapsed_time = time.time() - g.start_time
            logger.error(
                f"リクエストエラー: {request.method} {request.path} "
                f"- エラー: {str(e)} "
                f"- 処理時間: {elapsed_time:.3f}秒",
                exc_info=True,
            )
            raise

    return decorated_function


def log_slow_requests(threshold: float = 1.0) -> Callable:
    """
    遅いリクエストを警告ログに記録するデコレータ

    Args:
        threshold: 警告を出す閾値（秒）
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed_time = time.time() - start_time

            if elapsed_time > threshold:
                logger.warning(
                    f"遅いリクエスト検出: {request.method} {request.path} "
                    f"- 処理時間: {elapsed_time:.3f}秒 (閾値: {threshold}秒)"
                )

            return result

        return decorated_function

    return decorator
