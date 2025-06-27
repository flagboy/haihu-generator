"""
FileIOHelperのテスト
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.utils.file_io import FileIOHelper


class TestFileIOHelper:
    """FileIOHelperのテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        return {
            "name": "テストデータ",
            "value": 42,
            "nested": {"list": [1, 2, 3], "flag": True},
            "japanese": "麻雀牌譜",
        }

    def test_save_load_json(self, temp_dir, sample_data):
        """JSON保存・読み込みテスト"""
        # ファイルパス
        json_path = os.path.join(temp_dir, "test.json")

        # 保存
        FileIOHelper.save_json(sample_data, json_path)

        # ファイルが存在することを確認
        assert os.path.exists(json_path)

        # 読み込み
        loaded_data = FileIOHelper.load_json(json_path)

        # データが一致することを確認
        assert loaded_data == sample_data

    def test_save_json_pretty_print(self, temp_dir, sample_data):
        """JSON整形保存テスト"""
        # 整形あり
        pretty_path = os.path.join(temp_dir, "pretty.json")
        FileIOHelper.save_json(sample_data, pretty_path, pretty=True)

        # 整形なし
        compact_path = os.path.join(temp_dir, "compact.json")
        FileIOHelper.save_json(sample_data, compact_path, pretty=False)

        # ファイルサイズを比較（整形ありの方が大きいはず）
        pretty_size = os.path.getsize(pretty_path)
        compact_size = os.path.getsize(compact_path)
        assert pretty_size > compact_size

        # 内容は同じであることを確認
        assert FileIOHelper.load_json(pretty_path) == FileIOHelper.load_json(compact_path)

    def test_save_json_ensure_ascii(self, temp_dir):
        """JSON ASCII保存テスト"""
        data = {"japanese": "日本語テキスト"}

        # ensure_ascii=True
        ascii_path = os.path.join(temp_dir, "ascii.json")
        FileIOHelper.save_json(data, ascii_path, ensure_ascii=True)

        # ensure_ascii=False（デフォルト）
        unicode_path = os.path.join(temp_dir, "unicode.json")
        FileIOHelper.save_json(data, unicode_path, ensure_ascii=False)

        # ファイル内容を確認
        with open(ascii_path, encoding="utf-8") as f:
            ascii_content = f.read()
            assert "\\u" in ascii_content  # Unicodeエスケープされている

        with open(unicode_path, encoding="utf-8") as f:
            unicode_content = f.read()
            assert "日本語" in unicode_content  # 直接日本語が含まれる

    def test_save_load_yaml(self, temp_dir, sample_data):
        """YAML保存・読み込みテスト"""
        yaml_path = os.path.join(temp_dir, "test.yaml")

        # 保存
        FileIOHelper.save_yaml(sample_data, yaml_path)

        # 読み込み
        loaded_data = FileIOHelper.load_yaml(yaml_path)

        # データが一致することを確認
        assert loaded_data == sample_data

    def test_save_load_pickle(self, temp_dir, sample_data):
        """Pickle保存・読み込みテスト"""
        pickle_path = os.path.join(temp_dir, "test.pkl")

        # 保存
        FileIOHelper.save_pickle(sample_data, pickle_path)

        # 読み込み
        loaded_data = FileIOHelper.load_pickle(pickle_path)

        # データが一致することを確認
        assert loaded_data == sample_data

    def test_ensure_directory(self, temp_dir):
        """ディレクトリ作成テスト"""
        # 深いネストのディレクトリ
        deep_dir = os.path.join(temp_dir, "a", "b", "c", "d")

        # ディレクトリを作成
        created_path = FileIOHelper.ensure_directory(deep_dir)

        # ディレクトリが存在することを確認
        assert os.path.exists(deep_dir)
        assert os.path.isdir(deep_dir)
        assert created_path == Path(deep_dir)

    def test_safe_write(self, temp_dir):
        """安全な書き込みテスト"""
        file_path = os.path.join(temp_dir, "safe_write.txt")
        content = "安全に書き込まれたテキスト"

        # テキストファイルの書き込み
        FileIOHelper.safe_write(file_path, content)

        # ファイルが存在し、内容が正しいことを確認
        assert os.path.exists(file_path)
        with open(file_path, encoding="utf-8") as f:
            assert f.read() == content

        # バイナリファイルの書き込み
        binary_path = os.path.join(temp_dir, "safe_write.bin")
        binary_content = b"Binary content"
        FileIOHelper.safe_write(binary_path, binary_content, mode="wb", encoding=None)

        with open(binary_path, "rb") as f:
            assert f.read() == binary_content

    def test_safe_write_failure_cleanup(self, temp_dir, monkeypatch):
        """安全な書き込みの失敗時クリーンアップテスト"""
        from src.core.exceptions import FileWriteError

        file_path = os.path.join(temp_dir, "fail_write.txt")

        # 書き込み時にエラーを発生させる
        def mock_write(*args, **kwargs):
            # 一時ファイルを作成してからエラーを発生
            temp_path = Path(file_path).with_suffix(".txt.tmp")
            temp_path.touch()
            raise OSError("Mock write error")

        monkeypatch.setattr("builtins.open", mock_write)

        # エラーが発生することを確認
        with pytest.raises(FileWriteError):
            FileIOHelper.safe_write(file_path, "content")

        # 一時ファイルが削除されていることを確認
        temp_path = Path(file_path).with_suffix(".txt.tmp")
        assert not temp_path.exists()

    def test_auto_create_parent_directory(self, temp_dir):
        """親ディレクトリの自動作成テスト"""
        # 存在しない親ディレクトリを持つパス
        nested_path = os.path.join(temp_dir, "new", "dir", "file.json")
        data = {"test": "data"}

        # 保存（親ディレクトリが自動的に作成される）
        FileIOHelper.save_json(data, nested_path)

        # ファイルと親ディレクトリが存在することを確認
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))

    def test_load_nonexistent_file(self):
        """存在しないファイルの読み込みテスト"""
        from src.core.exceptions import FileReadError

        # JSON
        with pytest.raises(FileReadError):
            FileIOHelper.load_json("/nonexistent/file.json")

        # YAML
        with pytest.raises(FileReadError):
            FileIOHelper.load_yaml("/nonexistent/file.yaml")

        # Pickle
        with pytest.raises(FileReadError):
            FileIOHelper.load_pickle("/nonexistent/file.pkl")

    def test_pathlib_compatibility(self, temp_dir, sample_data):
        """PathlibのPathオブジェクトとの互換性テスト"""
        # Path オブジェクトを使用
        json_path = Path(temp_dir) / "pathlib_test.json"

        # 保存と読み込み
        FileIOHelper.save_json(sample_data, json_path)
        loaded_data = FileIOHelper.load_json(json_path)

        # データが一致することを確認
        assert loaded_data == sample_data
