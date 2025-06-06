#!/usr/bin/env python
"""
動画をデータベースに登録するスクリプト
"""

import sys
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from src.training.dataset_manager import DatasetManager


def main():
    """メイン処理"""
    # DatasetManagerを初期化
    dataset_manager = DatasetManager()

    # 動画ファイルのパス
    video_path = "web_interface/uploads/output.mp4"

    # 動画情報をデータベースに追加
    try:
        # データベースに直接挿入
        import sqlite3

        conn = sqlite3.connect(dataset_manager.db_path)
        cursor = conn.cursor()

        # 動画を追加
        video_id = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        cursor.execute(
            "INSERT INTO videos (id, name, path, created_at) VALUES (?, ?, ?, ?)",
            (video_id, "output.mp4", video_path, datetime.now().isoformat()),
        )

        # video_idは上で定義済み
        conn.commit()
        conn.close()

        print("✅ 動画を登録しました:")
        print(f"   - ID: {video_id}")
        print("   - 名前: output.mp4")
        print(f"   - パス: {video_path}")
        print("\n📝 次のステップ:")
        print("1. ブラウザで http://localhost:5001/scene_labeling にアクセス")
        print("2. 「output.mp4」を選択")
        print("3. 「新規セッション」をクリックしてラベリングを開始")

    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
