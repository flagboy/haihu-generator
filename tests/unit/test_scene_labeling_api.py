"""
対局画面ラベリングAPIのテスト
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.training.game_scene.labeling.api.scene_routes import (
    _sessions,
    scene_labeling_bp,
)


class TestSceneLabelingAPI(unittest.TestCase):
    """対局画面ラベリングAPIのテスト"""

    def setUp(self):
        """テストのセットアップ"""
        # セッションをクリア
        _sessions.clear()

        # テスト用の動画ファイルを作成
        self.test_video_dir = tempfile.mkdtemp()
        self.test_video_path = Path(self.test_video_dir) / "test_video.mp4"

        # ダミーの動画ファイルを作成
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, 30.0, (1920, 1080))

        # 100フレームのダミー動画
        for i in range(100):
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(
                frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
            )
            out.write(frame)

        out.release()

        # Flaskアプリのテストクライアントを作成
        from flask import Flask

        self.app = Flask(__name__)
        self.app.register_blueprint(scene_labeling_bp)
        self.client = self.app.test_client()

    def tearDown(self):
        """テストのクリーンアップ"""
        _sessions.clear()

        # テスト用ファイルを削除
        import shutil

        if Path(self.test_video_dir).exists():
            shutil.rmtree(self.test_video_dir)

    def test_create_session_new_video(self):
        """新しいビデオのセッション作成テスト"""
        # セッション作成リクエスト
        response = self.client.post(
            "/api/scene_labeling/sessions",
            data=json.dumps({"video_path": str(self.test_video_path)}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn("session_id", data)
        self.assertIn("video_info", data)
        self.assertEqual(data["video_info"]["total_frames"], 100)
        self.assertFalse(data["is_resumed"])

    @patch("src.training.game_scene.labeling.api.scene_routes.Path")
    @patch("sqlite3.connect")
    def test_create_session_existing_video(self, mock_connect, mock_path):
        """既存セッションがあるビデオのセッション作成テスト"""
        # Pathのモック設定
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # データベースのモック設定
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # 複数のfetchall呼び出しに対応
        mock_cursor.fetchall.side_effect = [
            # 最初の呼び出し: セッション情報
            [
                ("existing-session-id", 50),  # 50個のラベル
                ("old-session-id", 10),  # 10個のラベル（削除対象）
            ],
            # 2回目の呼び出し: ラベル情報（空）
            [],
        ]

        # セッション作成リクエスト
        response = self.client.post(
            "/api/scene_labeling/sessions",
            data=json.dumps({"video_path": str(self.test_video_path)}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        # 既存セッションIDが使用されることを確認
        self.assertEqual(data["session_id"], "existing-session-id")

        # ラベルの少ないセッションが削除されることを確認
        delete_calls = [
            call for call in mock_cursor.execute.call_args_list if "DELETE" in str(call)
        ]
        self.assertTrue(len(delete_calls) >= 1)

    def test_create_session_same_video_in_memory(self):
        """メモリ上に既存セッションがある場合のテスト"""
        # 最初のセッション作成
        response1 = self.client.post(
            "/api/scene_labeling/sessions",
            data=json.dumps({"video_path": str(self.test_video_path)}),
            content_type="application/json",
        )
        self.assertEqual(response1.status_code, 200)
        session1_id = json.loads(response1.data)["session_id"]

        # 同じビデオで2回目のセッション作成
        response2 = self.client.post(
            "/api/scene_labeling/sessions",
            data=json.dumps({"video_path": str(self.test_video_path)}),
            content_type="application/json",
        )
        self.assertEqual(response2.status_code, 200)
        session2_id = json.loads(response2.data)["session_id"]

        # 同じセッションIDが返されることを確認
        self.assertEqual(session1_id, session2_id)

        # メモリ上のセッション数が1つであることを確認
        self.assertEqual(len(_sessions), 1)

    def test_delete_session(self):
        """セッション削除のテスト"""
        # セッション作成
        response = self.client.post(
            "/api/scene_labeling/sessions",
            data=json.dumps({"video_path": str(self.test_video_path)}),
            content_type="application/json",
        )
        session_id = json.loads(response.data)["session_id"]

        # セッション削除
        response = self.client.delete(f"/api/scene_labeling/sessions/{session_id}")
        self.assertEqual(response.status_code, 200)

        # メモリから削除されていることを確認
        self.assertNotIn(session_id, _sessions)


if __name__ == "__main__":
    unittest.main()
