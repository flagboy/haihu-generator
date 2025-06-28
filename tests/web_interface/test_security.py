"""
Webインターフェースのセキュリティ機能テスト
"""

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from flask import Flask
from werkzeug.datastructures import FileStorage

from src.web_interface.security import (
    SecurityValidator,
    add_security_headers,
    escape_html,
    generate_csrf_token,
    rate_limit,
    validate_csrf_token,
    validate_json_input,
)


class TestSecurityValidator(TestCase):
    """SecurityValidatorのテスト"""

    def setUp(self):
        """テストのセットアップ"""
        self.validator = SecurityValidator()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テストのクリーンアップ"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_file_upload_valid_video(self):
        """有効な動画ファイルの検証"""
        # MP4ファイルのマジックナンバー
        mp4_header = b"\x00\x00\x00\x20ftypmp42"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(mp4_header)
            f.write(b"\x00" * 1000)  # ダミーデータ
            f.flush()

            # FileStorageオブジェクトを作成
            with open(f.name, "rb") as fp:
                file_storage = FileStorage(fp, filename="test.mp4", content_type="video/mp4")

                # magicモジュールをモック
                with patch.object(self.validator.magic, "from_buffer", return_value="video/mp4"):
                    is_valid, error = self.validator.validate_file_upload(file_storage, "video")

                self.assertTrue(is_valid)
                self.assertIsNone(error)

            os.unlink(f.name)

    def test_validate_file_upload_invalid_extension(self):
        """無効な拡張子のファイル"""
        file_storage = FileStorage(filename="test.exe", content_type="application/x-executable")

        is_valid, error = self.validator.validate_file_upload(file_storage, "video")

        self.assertFalse(is_valid)
        self.assertIn("許可されていない動画形式", error)

    def test_validate_file_upload_size_limit(self):
        """ファイルサイズ制限のテスト"""
        # 2GB + 1バイトのファイルを作成
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            # ファイルサイズを設定（実際には書き込まない）
            f.seek(2 * 1024 * 1024 * 1024 + 1)
            f.write(b"\x00")
            f.flush()

            with open(f.name, "rb") as fp:
                file_storage = FileStorage(fp, filename="large.mp4", content_type="video/mp4")

                is_valid, error = self.validator.validate_file_upload(file_storage, "video")

                self.assertFalse(is_valid)
                self.assertIn("動画ファイルサイズが大きすぎます", error)

            os.unlink(f.name)

    def test_validate_file_upload_mime_type_mismatch(self):
        """MIMEタイプの不一致"""
        # JPEGファイルのマジックナンバー
        jpeg_header = b"\xff\xd8\xff\xe0"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(jpeg_header)
            f.flush()

            with open(f.name, "rb") as fp:
                file_storage = FileStorage(fp, filename="fake.mp4", content_type="video/mp4")

                # magicモジュールをモック（JPEGと判定）
                with patch.object(self.validator.magic, "from_buffer", return_value="image/jpeg"):
                    is_valid, error = self.validator.validate_file_upload(file_storage, "video")

                self.assertFalse(is_valid)
                self.assertIn("ファイル内容が動画形式ではありません", error)

            os.unlink(f.name)

    def test_sanitize_path_valid(self):
        """有効なパスのサニタイズ"""
        base_dir = Path(self.temp_dir)
        valid_path = "uploads/test.mp4"

        result = self.validator.sanitize_path(valid_path, base_dir)

        self.assertIsNotNone(result)
        # 結果のパスが基準ディレクトリの下にあることを確認
        self.assertTrue(str(result).startswith(str(base_dir.resolve())))

    def test_sanitize_path_directory_traversal(self):
        """ディレクトリトラバーサル攻撃の防御"""
        base_dir = Path(self.temp_dir)
        malicious_paths = [
            "../../../etc/passwd",
            "uploads/../../../etc/passwd",
            "./../../sensitive_file",
        ]

        # Windowsスタイルのパスセパレータは、POSIXシステムでは有効なファイル名の一部として扱われる
        if os.name == "nt":
            malicious_paths.append("..\\..\\..\\windows\\system32\\config\\sam")

        for path in malicious_paths:
            result = self.validator.sanitize_path(path, base_dir)
            self.assertIsNone(result, f"パス '{path}' がブロックされませんでした")


class TestSecurityDecorators(TestCase):
    """セキュリティデコレーターのテスト"""

    def setUp(self):
        """Flaskアプリケーションのセットアップ"""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.secret_key = "test-secret-key"
        self.client = self.app.test_client()

    def test_validate_json_input(self):
        """JSON入力検証デコレーター"""

        @self.app.route("/test", methods=["POST"])
        @validate_json_input(required_fields=["name", "age"])
        def test_endpoint():
            return {"status": "ok"}

        # 有効なリクエスト
        response = self.client.post(
            "/test", json={"name": "test", "age": 25}, content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)

        # Content-Typeが不正
        response = self.client.post(
            "/test", data='{"name": "test", "age": 25}', content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)

        # 必須フィールドが不足
        response = self.client.post("/test", json={"name": "test"}, content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_rate_limit_decorator(self):
        """レート制限デコレーター（仮実装）"""

        @self.app.route("/test")
        @rate_limit(max_requests=5, window=60)
        def test_endpoint():
            return {"status": "ok"}

        # 現在は仮実装なので常に通る
        for _ in range(10):
            response = self.client.get("/test")
            self.assertEqual(response.status_code, 200)


class TestSecurityFunctions(TestCase):
    """セキュリティ関数のテスト"""

    def setUp(self):
        """Flaskアプリケーションのセットアップ"""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.secret_key = "test-secret-key"

    def test_csrf_token_generation(self):
        """CSRFトークン生成"""
        with self.app.test_request_context():
            token1 = generate_csrf_token()
            self.assertIsNotNone(token1)
            self.assertIsInstance(token1, str)
            self.assertTrue(len(token1) > 0)

            # 同じセッションでは同じトークン
            token2 = generate_csrf_token()
            self.assertEqual(token1, token2)

    def test_csrf_token_validation(self):
        """CSRFトークン検証"""
        with self.app.test_request_context():
            # トークン生成
            token = generate_csrf_token()

            # 正しいトークン
            self.assertTrue(validate_csrf_token(token))

            # 不正なトークン
            self.assertFalse(validate_csrf_token("invalid-token"))
            self.assertFalse(validate_csrf_token(""))

    def test_escape_html(self):
        """HTMLエスケープ"""
        test_cases = [
            (
                "<script>alert('XSS')</script>",
                "&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;",
            ),
            ('"><img src=x onerror=alert(1)>', "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"),
            ("&lt;already&gt;escaped&amp;", "&amp;lt;already&amp;gt;escaped&amp;amp;"),
            ("Normal text", "Normal text"),
            ("O'Reilly", "O&#x27;Reilly"),
        ]

        for input_text, expected in test_cases:
            result = escape_html(input_text)
            self.assertEqual(result, expected)

    def test_add_security_headers(self):
        """セキュリティヘッダーの追加"""
        with self.app.test_request_context():

            @self.app.route("/test")
            def test_endpoint():
                return "OK"

            with self.app.test_client() as client:
                response = client.get("/test")

                # セキュリティヘッダーを手動で追加
                response = add_security_headers(response)

                # ヘッダーの確認
                self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
                self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
                self.assertEqual(response.headers.get("X-XSS-Protection"), "1; mode=block")
                self.assertEqual(
                    response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin"
                )
                self.assertIn("Content-Security-Policy", response.headers)

    def test_add_security_headers_https(self):
        """HTTPS環境でのセキュリティヘッダー"""
        # HTTPSリクエストコンテキストでテスト
        with self.app.test_request_context(
            "https://example.com/test", environ_base={"wsgi.url_scheme": "https"}
        ):
            from flask import Response, request

            # HTTPSリクエストであることを確認
            self.assertTrue(request.is_secure)

            # レスポンスを作成してヘッダーを追加
            response = Response("OK")
            response = add_security_headers(response)

            # HSTSヘッダーの確認
            self.assertIn("Strict-Transport-Security", response.headers)
            self.assertIn("max-age=31536000", response.headers.get("Strict-Transport-Security"))
