"""
エラーハンドリングミドルウェア

APIエラーの統一的な処理を提供
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from ......utils.logger import get_logger

logger = get_logger(__name__)


class APIError(Exception):
    """API用カスタム例外"""

    def __init__(self, message: str, status_code: int = 400, payload: dict[str, Any] | None = None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        """エラーレスポンスを辞書形式に変換"""
        rv = {"error": {"message": self.message, "code": self.status_code}}
        if self.payload:
            rv["error"]["details"] = self.payload
        return rv


class ValidationError(APIError):
    """バリデーションエラー"""

    def __init__(self, message: str, errors: dict[str, Any] | None = None):
        super().__init__(message, status_code=422, payload=errors)


class NotFoundError(APIError):
    """リソース未検出エラー"""

    def __init__(self, resource: str, identifier: str | None = None):
        message = f"{resource}が見つかりません"
        if identifier:
            message += f": {identifier}"
        super().__init__(message, status_code=404)


class ConflictError(APIError):
    """競合エラー"""

    def __init__(self, message: str):
        super().__init__(message, status_code=409)


class InternalError(APIError):
    """内部エラー"""

    def __init__(self, message: str = "内部エラーが発生しました"):
        super().__init__(message, status_code=500)


def error_handler(f: Callable) -> Callable:
    """
    エラーハンドリングデコレータ

    関数内で発生した例外を捕捉し、適切なレスポンスを返す
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except APIError as e:
            logger.warning(f"APIエラー: {e.message}", exc_info=True)
            return jsonify(e.to_dict()), e.status_code
        except Exception as e:
            logger.error(f"予期しないエラー: {str(e)}", exc_info=True)
            error = InternalError()
            return jsonify(error.to_dict()), error.status_code

    return decorated_function


def handle_api_error(error: APIError):
    """APIエラーハンドラー"""
    logger.warning(f"APIエラー: {error.message}")
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def handle_http_error(error: HTTPException):
    """HTTPエラーハンドラー"""
    logger.warning(f"HTTPエラー: {error.code} - {error.description}")
    response = jsonify(
        {"error": {"message": error.description or "エラーが発生しました", "code": error.code}}
    )
    response.status_code = error.code
    return response


def handle_generic_error(error: Exception):
    """一般的なエラーハンドラー"""
    logger.error(f"予期しないエラー: {str(error)}", exc_info=True)
    response = jsonify({"error": {"message": "内部エラーが発生しました", "code": 500}})
    response.status_code = 500
    return response


def register_error_handlers(app: Flask):
    """
    Flaskアプリケーションにエラーハンドラーを登録

    Args:
        app: Flaskアプリケーション
    """
    app.register_error_handler(APIError, handle_api_error)
    app.register_error_handler(HTTPException, handle_http_error)
    app.register_error_handler(Exception, handle_generic_error)

    # 特定のHTTPステータスコード用のハンドラー
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": {"message": "リソースが見つかりません", "code": 404}}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return (
            jsonify({"error": {"message": "許可されていないHTTPメソッドです", "code": 405}}),
            405,
        )

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": {"message": "内部エラーが発生しました", "code": 500}}), 500
