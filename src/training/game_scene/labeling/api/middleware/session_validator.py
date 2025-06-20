"""
セッション検証ミドルウェア

セッションの存在確認と検証を行う
"""

from collections.abc import Callable
from functools import wraps

from flask import g

from ......utils.logger import get_logger
from .error_handler import NotFoundError

logger = get_logger(__name__)


def validate_session(session_manager_func: Callable) -> Callable:
    """
    セッション検証デコレータ

    Args:
        session_manager_func: セッションマネージャーを取得する関数
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(session_id: str, *args, **kwargs):
            # セッションマネージャーを取得
            session_manager = session_manager_func()

            # セッションの存在確認
            if not session_manager.session_exists(session_id):
                logger.warning(f"存在しないセッションへのアクセス: {session_id}")
                raise NotFoundError("セッション", session_id)

            # セッションを取得してgオブジェクトに保存
            session = session_manager.get_session(session_id)
            if not session:
                logger.error(f"セッションの取得に失敗: {session_id}")
                raise NotFoundError("セッション", session_id)

            g.current_session = session
            g.session_id = session_id

            # セッションの状態を確認
            if hasattr(session, "is_closed") and session.is_closed():
                logger.warning(f"終了済みセッションへのアクセス: {session_id}")
                raise NotFoundError("アクティブなセッション", session_id)

            return f(session_id, *args, **kwargs)

        return decorated_function

    return decorator


def require_active_session(f: Callable) -> Callable:
    """
    アクティブなセッションを要求するデコレータ

    validate_sessionと組み合わせて使用
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, "current_session"):
            logger.error("セッション検証が行われていません")
            raise RuntimeError("セッション検証が必要です")

        session = g.current_session
        if hasattr(session, "is_active") and not session.is_active():
            logger.warning(f"非アクティブなセッションへのアクセス: {g.session_id}")
            raise NotFoundError("アクティブなセッション", g.session_id)

        return f(*args, **kwargs)

    return decorated_function
