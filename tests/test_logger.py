"""
LoggerおよびLoggerMixinのテスト
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.utils.config import ConfigManager
from src.utils.logger import LoggerMixin, get_logger, setup_logger, setup_logging


class TestLogger:
    """ロガー設定のテストクラス"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config_manager(self, temp_dir):
        """モックConfigManagerのフィクスチャ"""
        mock = Mock(spec=ConfigManager)
        mock.get_logging_config.return_value = {
            "level": "DEBUG",
            "format": "{time} | {level} | {message}",
            "file_path": os.path.join(temp_dir, "test.log"),
            "rotation": "100 MB",
            "retention": "7 days",
        }
        return mock

    @patch("src.utils.logger.logger")
    def test_setup_logger_default(self, mock_logger):
        """デフォルト設定でのロガー初期化テスト"""
        # ConfigManagerを渡さずに初期化
        setup_logger()

        # loggerの設定が呼ばれることを確認
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count == 2  # stdout と file
        mock_logger.info.assert_called_with("ログシステムが初期化されました")

    @patch("src.utils.logger.logger")
    def test_setup_logger_with_config(self, mock_logger, mock_config_manager):
        """カスタム設定でのロガー初期化テスト"""
        setup_logger(mock_config_manager)

        # 設定が反映されることを確認
        mock_config_manager.get_logging_config.assert_called_once()
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count == 2

        # stdout追加の引数を確認
        stdout_call = mock_logger.add.call_args_list[0]
        assert stdout_call[1]["level"] == "DEBUG"
        assert stdout_call[1]["format"] == "{time} | {level} | {message}"

        # ファイル追加の引数を確認
        file_call = mock_logger.add.call_args_list[1]
        assert "test.log" in file_call[0][0]
        assert file_call[1]["rotation"] == "100 MB"
        assert file_call[1]["retention"] == "7 days"

    @patch("src.utils.logger.logger")
    def test_setup_logger_creates_log_directory(self, mock_logger, temp_dir):
        """ログディレクトリ作成のテスト"""
        # 存在しないディレクトリを指定
        log_path = os.path.join(temp_dir, "logs", "subdir", "test.log")
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_logging_config.return_value = {
            "level": "INFO",
            "file_path": log_path,
        }

        setup_logger(mock_config)

        # ディレクトリが作成されることを確認
        assert Path(log_path).parent.exists()

    @patch("src.utils.logger.logger")
    def test_get_logger(self, mock_logger):
        """名前付きロガー取得のテスト"""
        name = "TestComponent"
        result = get_logger(name)

        # bind が正しく呼ばれることを確認
        mock_logger.bind.assert_called_once_with(name=name)
        assert result == mock_logger.bind.return_value

    @patch("src.utils.logger.setup_logger")
    def test_setup_logging_alias(self, mock_setup_logger):
        """setup_loggingエイリアスのテスト"""
        mock_config = Mock()
        setup_logging(mock_config)

        # setup_loggerが同じ引数で呼ばれることを確認
        mock_setup_logger.assert_called_once_with(mock_config)


class TestLoggerMixin:
    """LoggerMixinのテストクラス"""

    class SampleClass(LoggerMixin):
        """テスト用のサンプルクラス"""

        pass

    @patch("src.utils.logger.get_logger")
    def test_logger_property(self, mock_get_logger):
        """loggerプロパティのテスト"""
        instance = self.SampleClass()
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # loggerプロパティにアクセス
        result = instance.logger

        # クラス名でロガーが取得されることを確認
        mock_get_logger.assert_called_once_with("SampleClass")
        assert result == mock_logger

    @patch("src.utils.logger.get_logger")
    def test_logger_property_cached(self, mock_get_logger):
        """loggerプロパティのキャッシュテスト"""
        instance = self.SampleClass()

        # 複数回アクセス
        _ = instance.logger
        _ = instance.logger

        # get_loggerは毎回呼ばれる（プロパティなので）
        assert mock_get_logger.call_count == 2


class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_full_logging_setup(self, temp_dir):
        """実際のログ出力テスト"""
        # カスタム設定でロガーを初期化
        log_file = os.path.join(temp_dir, "integration_test.log")
        config = Mock(spec=ConfigManager)
        config.get_logging_config.return_value = {
            "level": "INFO",
            "file_path": log_file,
            "format": "{message}",  # シンプルなフォーマット
        }

        # パッチなしで実際のロガーを使用
        from loguru import logger as real_logger

        # 既存のハンドラーを削除
        real_logger.remove()

        # テスト用の設定
        real_logger.add(log_file, level="INFO", format="{message}", encoding="utf-8")

        # テスト用クラス
        class TestComponent(LoggerMixin):
            def do_something(self):
                self.logger.info("テストメッセージ")

        # ログ出力
        component = TestComponent()
        component.do_something()

        # ログファイルが作成され、メッセージが含まれることを確認
        assert Path(log_file).exists()
        with open(log_file, encoding="utf-8") as f:
            content = f.read()
            assert "テストメッセージ" in content
