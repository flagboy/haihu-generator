"""
麻雀牌検出システム - Webインターフェース設定
"""

import os
from pathlib import Path

class Config:
    """基本設定クラス"""
    
    # Flask設定
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'mahjong-tile-detection-system-2024'
    
    # アップロード設定
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
    
    # データベース設定
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///mahjong_tiles.db'
    
    # WebSocket設定
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # ログ設定
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'web_interface.log')
    
    # セッション設定
    SESSION_TIMEOUT = 3600  # 1時間
    
    # パフォーマンス設定
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1年
    
    @staticmethod
    def init_app(app):
        """アプリケーション初期化"""
        # ログディレクトリを作成
        log_dir = Path(Config.LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
        
        # アップロードディレクトリを作成
        upload_dir = Path(Config.UPLOAD_FOLDER)
        upload_dir.mkdir(exist_ok=True)


class DevelopmentConfig(Config):
    """開発環境設定"""
    DEBUG = True
    TESTING = False
    
    # 開発用設定
    TEMPLATES_AUTO_RELOAD = True
    EXPLAIN_TEMPLATE_LOADING = False


class ProductionConfig(Config):
    """本番環境設定"""
    DEBUG = False
    TESTING = False
    
    # セキュリティ設定
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # パフォーマンス設定
    SEND_FILE_MAX_AGE_DEFAULT = 31536000


class TestingConfig(Config):
    """テスト環境設定"""
    DEBUG = True
    TESTING = True
    
    # テスト用設定
    WTF_CSRF_ENABLED = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'test_uploads')


# 設定マッピング
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """現在の設定を取得"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])