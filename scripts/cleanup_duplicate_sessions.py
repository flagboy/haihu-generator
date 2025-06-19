"""
重複セッションのクリーンアップスクリプト
"""

import sqlite3
from pathlib import Path


def cleanup_duplicate_sessions(db_path: str):
    """同じビデオIDで複数のセッションがある場合、ラベル数が最も多いものだけを残す"""

    if not Path(db_path).exists():
        print(f"データベースが見つかりません: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # ビデオIDごとにグループ化して、複数セッションがあるものを検出
        cursor.execute("""
            SELECT video_id, COUNT(*) as session_count
            FROM labeling_sessions
            GROUP BY video_id
            HAVING COUNT(*) > 1
        """)

        duplicate_videos = cursor.fetchall()

        if not duplicate_videos:
            print("重複セッションはありません")
            return

        print(f"{len(duplicate_videos)}個のビデオで重複セッションが見つかりました")

        total_deleted = 0

        for video_id, session_count in duplicate_videos:
            print(f"\nビデオID: {video_id} ({session_count}個のセッション)")

            # このビデオIDの全セッションを取得（ラベル数の降順）
            cursor.execute(
                """
                SELECT session_id, labeled_frames, created_at
                FROM labeling_sessions
                WHERE video_id = ?
                ORDER BY labeled_frames DESC, created_at DESC
            """,
                (video_id,),
            )

            sessions = cursor.fetchall()

            # 最初のセッション（最もラベル数が多い）を保持
            keep_session = sessions[0]
            print(f"  保持: {keep_session[0]} (ラベル数: {keep_session[1]})")

            # 残りのセッションを削除
            for session in sessions[1:]:
                session_id, labeled_frames, created_at = session
                print(f"  削除: {session_id} (ラベル数: {labeled_frames})")

                # セッション情報を削除
                cursor.execute(
                    """
                    DELETE FROM labeling_sessions
                    WHERE session_id = ?
                """,
                    (session_id,),
                )

                total_deleted += 1

        conn.commit()
        print(f"\n合計 {total_deleted} 個のセッションを削除しました")

        # 削除後の状態を確認
        cursor.execute("""
            SELECT video_id, COUNT(*) as session_count, SUM(labeled_frames) as total_labels
            FROM labeling_sessions
            GROUP BY video_id
            ORDER BY video_id
        """)

        print("\n削除後のセッション状態:")
        for video_id, session_count, total_labels in cursor.fetchall():
            print(f"  {video_id}: {session_count}個のセッション, {total_labels}個のラベル")

    except sqlite3.Error as e:
        print(f"データベースエラー: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    # web_interface内のデータベース
    cleanup_duplicate_sessions("web_interface/data/training/game_scene_labels.db")

    # プロジェクトルートのデータベース（もし存在すれば）
    if Path("data/training/game_scene_labels.db").exists():
        cleanup_duplicate_sessions("data/training/game_scene_labels.db")
