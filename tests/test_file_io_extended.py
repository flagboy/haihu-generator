"""
FileIOHelperの拡張テスト - カバレッジ向上のため
"""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.core.exceptions import FileFormatError, FileReadError, FileWriteError
from src.utils.file_io import FileIOHelper


class TestFileIOHelperExtended:
    """FileIOHelperの拡張テストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_json_errors(self, temp_dir):
        """JSON保存時のエラーハンドリング"""
        # OSError (ディスク容量不足などをシミュレート)
        with patch("builtins.open", side_effect=OSError("Disk full")):
            with pytest.raises(FileWriteError) as exc_info:
                FileIOHelper.save_json({"test": "data"}, os.path.join(temp_dir, "test.json"))
            assert "JSONファイルの保存に失敗しました" in str(exc_info.value)
            assert exc_info.value.details["original_error"] == "Disk full"

        # TypeError (JSONシリアライズ不可能なオブジェクト)
        class NonSerializable:
            pass

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.save_json(
                {"obj": NonSerializable()}, os.path.join(temp_dir, "invalid.json")
            )
        assert "データをJSON形式に変換できません" in str(exc_info.value)

    def test_load_json_errors(self, temp_dir):
        """JSON読み込み時のエラーハンドリング"""
        # OSError (読み込み権限なしなど)
        json_path = os.path.join(temp_dir, "readable.json")
        FileIOHelper.save_json({"test": "data"}, json_path)

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(FileReadError) as exc_info:
                FileIOHelper.load_json(json_path)
            assert "JSONファイルの読み込みに失敗しました" in str(exc_info.value)

        # JSONDecodeError (不正なJSON)
        bad_json_path = os.path.join(temp_dir, "bad.json")
        with open(bad_json_path, "w") as f:
            f.write("{invalid json}")

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.load_json(bad_json_path)
        assert "無効なJSON形式です" in str(exc_info.value)
        assert "line" in exc_info.value.details

    def test_save_yaml_errors(self, temp_dir):
        """YAML保存時のエラーハンドリング"""
        # OSError
        with patch("builtins.open", side_effect=OSError("Disk error")):
            with pytest.raises(FileWriteError) as exc_info:
                FileIOHelper.save_yaml({"test": "data"}, os.path.join(temp_dir, "test.yaml"))
            assert "YAMLファイルの保存に失敗しました" in str(exc_info.value)

        # YAMLError (循環参照などをシミュレート)
        with patch("yaml.dump", side_effect=yaml.YAMLError("Circular reference")):
            with pytest.raises(FileFormatError) as exc_info:
                FileIOHelper.save_yaml({"test": "data"}, os.path.join(temp_dir, "circular.yaml"))
            assert "データをYAML形式に変換できません" in str(exc_info.value)

    def test_load_yaml_errors(self, temp_dir):
        """YAML読み込み時のエラーハンドリング"""
        # OSError
        yaml_path = os.path.join(temp_dir, "readable.yaml")
        FileIOHelper.save_yaml({"test": "data"}, yaml_path)

        with patch("builtins.open", side_effect=OSError("IO error")):
            with pytest.raises(FileReadError) as exc_info:
                FileIOHelper.load_yaml(yaml_path)
            assert "YAMLファイルの読み込みに失敗しました" in str(exc_info.value)

        # YAMLError (不正なYAML)
        bad_yaml_path = os.path.join(temp_dir, "bad.yaml")
        with open(bad_yaml_path, "w") as f:
            f.write("invalid:\n  - unmatched bracket: [")

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.load_yaml(bad_yaml_path)
        assert "無効なYAML形式です" in str(exc_info.value)

    def test_save_pickle_errors(self, temp_dir):
        """Pickle保存時のエラーハンドリング"""
        # OSError
        with patch("builtins.open", side_effect=OSError("Write error")):
            with pytest.raises(FileWriteError) as exc_info:
                FileIOHelper.save_pickle({"test": "data"}, os.path.join(temp_dir, "test.pkl"))
            assert "Pickleファイルの保存に失敗しました" in str(exc_info.value)

        # PicklingError (シリアライズ不可能なオブジェクト)
        import threading

        lock = threading.Lock()  # Lockオブジェクトはpickle化できない

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.save_pickle(lock, os.path.join(temp_dir, "unpicklable.pkl"))
        assert "データをPickle形式に変換できません" in str(exc_info.value)

    def test_load_pickle_errors(self, temp_dir):
        """Pickle読み込み時のエラーハンドリング"""
        # OSError
        pickle_path = os.path.join(temp_dir, "readable.pkl")
        FileIOHelper.save_pickle({"test": "data"}, pickle_path)

        with patch("builtins.open", side_effect=OSError("Read error")):
            with pytest.raises(FileReadError) as exc_info:
                FileIOHelper.load_pickle(pickle_path)
            assert "Pickleファイルの読み込みに失敗しました" in str(exc_info.value)

        # UnpicklingError (破損したpickleファイル)
        bad_pickle_path = os.path.join(temp_dir, "bad.pkl")
        with open(bad_pickle_path, "wb") as f:
            f.write(b"corrupted pickle data")

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.load_pickle(bad_pickle_path)
        assert "無効なPickle形式または破損したファイル" in str(exc_info.value)

        # EOFError (不完全なpickleファイル)
        incomplete_pickle = os.path.join(temp_dir, "incomplete.pkl")
        with open(incomplete_pickle, "wb") as f:
            f.write(pickle.dumps({"test": "data"})[:5])  # 途中で切れたpickle

        with pytest.raises(FileFormatError) as exc_info:
            FileIOHelper.load_pickle(incomplete_pickle)
        assert "無効なPickle形式または破損したファイル" in str(exc_info.value)

    def test_safe_write_error_cleanup(self, temp_dir):
        """safe_write のエラー時クリーンアップ処理"""
        file_path = os.path.join(temp_dir, "safe_test.txt")
        temp_path = Path(file_path).with_suffix(".txt.tmp")

        # OSError発生時に一時ファイルが削除されることを確認
        with patch("builtins.open", side_effect=OSError("Write failed")):
            # 一時ファイルを作成しておく
            temp_path.touch()

            with pytest.raises(FileWriteError) as exc_info:
                FileIOHelper.safe_write(file_path, "content")

            # エラーメッセージと詳細情報の確認
            assert "ファイルの安全な書き込みに失敗しました" in str(exc_info.value)
            assert exc_info.value.details["temp_path"] == str(temp_path)

        # 一時ファイルが削除されていることを確認
        assert not temp_path.exists()  # safe_writeは例外時に一時ファイルを削除する

    def test_safe_write_unexpected_error(self, temp_dir):
        """safe_write の予期しないエラー処理"""
        file_path = os.path.join(temp_dir, "unexpected.txt")

        # 予期しない例外
        with patch("builtins.open", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(FileWriteError) as exc_info:
                FileIOHelper.safe_write(file_path, "content")

            assert "予期しないエラーが発生しました" in str(exc_info.value)
            assert exc_info.value.details["error_type"] == "RuntimeError"

    def test_safe_write_temp_file_cleanup_failure(self, temp_dir):
        """一時ファイル削除失敗時の処理"""
        file_path = os.path.join(temp_dir, "cleanup_fail.txt")

        # 書き込みは失敗、一時ファイル削除も失敗するケース
        with (
            patch("builtins.open", side_effect=OSError("Write failed")),
            patch.object(Path, "unlink", side_effect=PermissionError("Cannot delete")),
            pytest.raises(FileWriteError),
        ):
            FileIOHelper.safe_write(file_path, "content")
            # 例外は抑制されるので、エラーにはならない

    def test_json_pretty_format(self, temp_dir):
        """JSON整形オプションの詳細テスト"""
        data = {"key": "value", "nested": {"item": 1}}

        # pretty=False, ensure_ascii=True
        compact_ascii_path = os.path.join(temp_dir, "compact_ascii.json")
        FileIOHelper.save_json(data, compact_ascii_path, pretty=False, ensure_ascii=True)

        with open(compact_ascii_path) as f:
            content = f.read()
            assert "\n" not in content  # 改行なし
            assert '","' in content or '":' in content  # コンパクトな区切り文字

    def test_yaml_flow_style(self, temp_dir):
        """YAMLフロースタイルオプションのテスト"""
        data = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}}

        # default_flow_style=True
        flow_path = os.path.join(temp_dir, "flow.yaml")
        FileIOHelper.save_yaml(data, flow_path, default_flow_style=True)

        with open(flow_path) as f:
            content = f.read()
            # フロースタイルでは波括弧や角括弧が使われる
            assert "{" in content or "[" in content

    def test_logger_debug_messages(self, temp_dir):
        """ログ出力のテスト"""
        with patch("src.utils.file_io.logger") as mock_logger:
            # save/load の各操作でdebugログが出力されることを確認
            json_path = os.path.join(temp_dir, "log_test.json")
            FileIOHelper.save_json({"test": "data"}, json_path)
            mock_logger.debug.assert_called_with(f"JSON saved to {json_path}")

            FileIOHelper.load_json(json_path)
            mock_logger.debug.assert_called_with(f"JSON loaded from {json_path}")

    def test_edge_cases(self, temp_dir):
        """エッジケースのテスト"""
        # 空のデータ
        empty_json = os.path.join(temp_dir, "empty.json")
        FileIOHelper.save_json({}, empty_json)
        assert FileIOHelper.load_json(empty_json) == {}

        # 大きなネストデータ
        deep_data = {"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}
        deep_json = os.path.join(temp_dir, "deep.json")
        FileIOHelper.save_json(deep_data, deep_json)
        loaded = FileIOHelper.load_json(deep_json)
        assert loaded["level1"]["level2"]["level3"]["level4"]["level5"] == "deep"
