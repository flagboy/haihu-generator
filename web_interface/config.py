"""
Webインターフェースの設定ファイル
"""

import os
from pathlib import Path

# 基本設定
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


# Flaskアプリケーション設定
class Config:
    """基本設定クラス"""

    # セキュリティ設定
    SECRET_KEY = os.environ.get("SECRET_KEY", "mahjong-tile-detection-system-2024")
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # ファイルアップロード設定
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB

    # 許可するファイル拡張子
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

    # CORS設定
    CORS_ALLOWED_ORIGINS = "*"  # 開発環境用

    # WebSocket設定
    SOCKETIO_ASYNC_MODE = "threading"

    # データベース設定
    DATABASE_PATH = BASE_DIR.parent / "data" / "training" / "dataset.db"

    # ログ設定
    LOG_LEVEL = "INFO"
    LOG_FILE = BASE_DIR / "logs" / "web_interface.log"


class DevelopmentConfig(Config):
    """開発環境設定"""

    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """本番環境設定"""

    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

    # 本番環境では必ず環境変数から取得
    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY環境変数が設定されていません")

    # CORS設定（本番環境では特定のオリジンのみ許可）
    CORS_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")


class TestingConfig(Config):
    """テスト環境設定"""

    TESTING = True
    WTF_CSRF_ENABLED = False


# 環境に応じた設定を選択
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """現在の環境に応じた設定を取得"""
    env = os.environ.get("FLASK_ENV", "development")
    return config.get(env, DevelopmentConfig)
