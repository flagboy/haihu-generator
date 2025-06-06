#!/usr/bin/env python
"""
対局画面学習データ作成システムの起動スクリプト
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    """メイン処理"""
    print("=" * 60)
    print("対局画面学習データ作成システムを起動します")
    print("=" * 60)

    # プロジェクトルートに移動
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Webインターフェースのディレクトリ
    web_dir = project_root / "web_interface"

    # 環境変数を設定
    env = os.environ.copy()
    env["PORT"] = "5001"

    print("\n📌 起動情報:")
    print("  - ポート: 5001")
    print("  - URL: http://localhost:5001")
    print("  - 対局画面ラベリング: http://localhost:5001/scene_labeling")

    print("\n🚀 サーバーを起動中...")

    try:
        # uvでサーバーを起動
        cmd = ["uv", "run", "python", "run.py"]
        process = subprocess.Popen(
            cmd,
            cwd=web_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # 起動メッセージを表示
        print("\n📝 サーバーログ:")
        print("-" * 60)

        # 出力を監視
        for line in iter(process.stdout.readline, ""):
            if line:
                print(line.rstrip())

                # サーバーが起動したら使用方法を表示
                if "Running on" in line or "Serving Flask app" in line:
                    time.sleep(1)
                    print("\n" + "=" * 60)
                    print("✅ サーバーが起動しました！")
                    print("=" * 60)
                    print("\n📚 使用方法:")
                    print("1. ブラウザで以下のURLにアクセス:")
                    print("   http://localhost:5001/scene_labeling")
                    print("\n2. 動画をアップロード:")
                    print("   - メインページ → データ管理 → 動画アップロード")
                    print("\n3. 対局画面のラベリング:")
                    print("   - 動画を選択")
                    print("   - 自動検出結果を確認")
                    print("   - 必要に応じて手動修正")
                    print("\n4. キーボードショートカット:")
                    print("   - G: 対局画面に設定")
                    print("   - N: 非対局画面に設定")
                    print("   - Space: 再生/一時停止")
                    print("   - ←/→: フレーム移動")
                    print("\n5. 終了するには Ctrl+C を押してください")
                    print("=" * 60 + "\n")

        # プロセスが終了するまで待機
        process.wait()

    except KeyboardInterrupt:
        print("\n\n🛑 サーバーを停止しています...")
        if "process" in locals():
            process.terminate()
            process.wait()
        print("✅ サーバーを停止しました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
