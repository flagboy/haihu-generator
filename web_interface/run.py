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

from app import app, socketio

def main():
    """メイン関数"""
    # 環境変数の設定
    os.environ.setdefault('FLASK_ENV', 'development')
    
    # 開発サーバーの設定
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"麻雀牌検出システム Webインターフェースを起動します...")
    print(f"URL: http://{host}:{port}")
    print(f"デバッグモード: {debug}")
    
    # SocketIOサーバーを起動
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        log_output=debug
    )

if __name__ == '__main__':
    main()