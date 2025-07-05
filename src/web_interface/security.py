"""
Webインターフェースのセキュリティ機能
"""

import hashlib
import hmac
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import magic
from flask import abort, current_app, request, session
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# 許可するファイル拡張子
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

# 許可するMIMEタイプ
ALLOWED_VIDEO_MIMETYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
}
ALLOWED_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
}

# ファイルサイズ制限（バイト）
MAX_VIDEO_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB


class SecurityValidator:
    """セキュリティ検証クラス"""

    def __init__(self):
        """初期化"""
        self.magic = magic.Magic(mime=True)

    def validate_file_upload(
        self, file: FileStorage, file_type: str = "video"
    ) -> tuple[bool, str | None]:
        """
        ファイルアップロードの検証

        Args:
            file: アップロードされたファイル
            file_type: ファイルタイプ（"video" または "image"）

        Returns:
            (検証成功, エラーメッセージ)
        """
        if not file or file.filename == "":
            return False, "ファイルが選択されていません"

        # ファイル名の検証
        filename = secure_filename(file.filename) if file.filename else ""
        if not filename:
            return False, "無効なファイル名です"

        # 拡張子の検証
        file_ext = Path(filename).suffix.lower()
        if file_type == "video":
            if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
                return (
                    False,
                    f"許可されていない動画形式です。対応形式: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
                )
        elif file_type == "image":
            if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
                return (
                    False,
                    f"許可されていない画像形式です。対応形式: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
                )
        else:
            return False, "不明なファイルタイプです"

        # ファイルサイズの検証
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # ファイルポインタを戻す

        if file_type == "video" and file_size > MAX_VIDEO_SIZE:
            return (
                False,
                f"動画ファイルサイズが大きすぎます（最大: {MAX_VIDEO_SIZE // (1024**3)}GB）",
            )
        elif file_type == "image" and file_size > MAX_IMAGE_SIZE:
            return (
                False,
                f"画像ファイルサイズが大きすぎます（最大: {MAX_IMAGE_SIZE // (1024**2)}MB）",
            )

        # MIMEタイプの検証（マジックナンバーベース）
        try:
            # ファイルの最初の部分を読み取ってMIMEタイプを判定
            file_header = file.read(8192)
            file.seek(0)  # ファイルポインタを戻す

            detected_mime = self.magic.from_buffer(file_header)

            if file_type == "video" and detected_mime not in ALLOWED_VIDEO_MIMETYPES:
                return False, f"ファイル内容が動画形式ではありません（検出: {detected_mime}）"
            elif file_type == "image" and detected_mime not in ALLOWED_IMAGE_MIMETYPES:
                return False, f"ファイル内容が画像形式ではありません（検出: {detected_mime}）"
        except Exception as e:
            return False, f"ファイル形式の検証中にエラーが発生しました: {str(e)}"

        return True, None

    def sanitize_path(self, path: str, base_dir: Path) -> Path | None:
        """
        パスのサニタイズ（ディレクトリトラバーサル対策）

        Args:
            path: 検証するパス
            base_dir: 基準ディレクトリ

        Returns:
            安全なパス or None（危険な場合）
        """
        try:
            # パスを正規化
            target_path = (base_dir / path).resolve()
            base_path = base_dir.resolve()

            # base_dir配下にあることを確認
            if not str(target_path).startswith(str(base_path)):
                return None

            return target_path
        except Exception:
            return None


def validate_json_input(required_fields: list[str] | None = None) -> Callable:
    """
    JSON入力の検証デコレーター

    Args:
        required_fields: 必須フィールドのリスト
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            if not request.is_json:
                abort(400, "Content-Type must be application/json")

            data = request.get_json()
            if not data:
                abort(400, "JSONデータが空です")

            # 必須フィールドのチェック
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    abort(400, f"必須フィールドが不足しています: {', '.join(missing_fields)}")

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_auth(f: Callable) -> Callable:
    """
    認証要求デコレーター（将来の実装用）
    """

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # TODO: 認証システムの実装
        # 現在は仮実装として常に通す
        return f(*args, **kwargs)

    return decorated_function


def rate_limit(max_requests: int = 60, window: int = 60) -> Callable:
    """
    レート制限デコレーター（簡易版）

    Args:
        max_requests: ウィンドウ内の最大リクエスト数
        window: ウィンドウサイズ（秒）
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            # TODO: Redis等を使った本格的なレート制限の実装
            # 現在は仮実装
            return f(*args, **kwargs)

        return decorated_function

    return decorator


def generate_csrf_token() -> str:
    """
    CSRFトークンの生成
    """
    if "csrf_token" not in session:
        secret_key = current_app.secret_key
        if isinstance(secret_key, str):
            secret_key = secret_key.encode()
        session["csrf_token"] = hmac.new(secret_key, os.urandom(64), hashlib.sha256).hexdigest()
    return session["csrf_token"]


def validate_csrf_token(token: str) -> bool:
    """
    CSRFトークンの検証

    Args:
        token: 検証するトークン

    Returns:
        検証成功かどうか
    """
    return hmac.compare_digest(session.get("csrf_token", ""), token)


def escape_html(text: str) -> str:
    """
    HTMLエスケープ

    Args:
        text: エスケープするテキスト

    Returns:
        エスケープ済みテキスト
    """
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        "<": "&lt;",
        ">": "&gt;",
    }
    return "".join(html_escape_table.get(c, c) for c in text)


def add_security_headers(response):
    """
    セキュリティヘッダーの追加

    Args:
        response: Flaskレスポンスオブジェクト

    Returns:
        ヘッダー追加済みレスポンス
    """
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: blob:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self' ws: wss:;"
    )

    # その他のセキュリティヘッダー
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # HSTS（HTTPS環境でのみ）
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response
