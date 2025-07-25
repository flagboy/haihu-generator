#!/usr/bin/env python3
"""
麻雀牌検出システム - Webインターフェース起動スクリプト
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import app, socketio  # noqa: E402


def main():
    """メイン関数"""
    # 環境変数の設定
    os.environ.setdefault("FLASK_ENV", "development")

    # 開発サーバーの設定
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_ENV") == "development"

    print("麻雀牌検出システム Webインターフェースを起動します...")
    print(f"URL: http://{host}:{port}")
    print(f"デバッグモード: {debug}")

    # SocketIOサーバーを起動
    # 開発環境でのみ allow_unsafe_werkzeug=True を使用
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        log_output=debug,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
