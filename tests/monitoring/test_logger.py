"""
構造化ログシステムのテスト
"""

import tempfile
from pathlib import Path

import pytest

from src.monitoring.logger import StructuredLogger, get_structured_logger


class TestStructuredLogger:
    """StructuredLoggerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(name="test_logger", log_dir=Path(tmpdir), level="INFO")

            assert logger.name == "test_logger"
            assert logger.log_dir == Path(tmpdir)

    def test_log_levels(self):
        """ログレベルのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(name="test_logger", log_dir=Path(tmpdir), level="DEBUG")

            # 各レベルでログ出力（エラーは発生しないことを確認）
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

    def test_metadata_binding(self):
        """メタデータバインディングのテスト"""
        logger = get_structured_logger("test_logger")

        # コンテキスト情報をバインド
        bound_logger = logger.bind(request_id="req-123", user_id="user-456")

        # バインドされたロガーは新しいインスタンス
        assert bound_logger is not logger
        assert bound_logger.name == logger.name

    def test_performance_logging(self):
        """パフォーマンスログのテスト"""
        logger = get_structured_logger("test_logger")

        # パフォーマンスログ（エラーが発生しないことを確認）
        logger.log_performance(
            operation="test_operation", duration=1.234, success=True, items_processed=100
        )

    def test_error_with_context(self):
        """コンテキスト付きエラーログのテスト"""
        logger = get_structured_logger("test_logger")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            # エラーログ（エラーが発生しないことを確認）
            logger.log_error_with_context(
                e, operation="test_operation", input_data={"key": "value"}
            )

    def test_json_output(self):
        """JSON出力のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(
                name="test_logger",
                log_dir=Path(tmpdir),
                level="INFO",
                enable_console=False,
                enable_file=True,
            )

            # ログ出力
            logger.info("Test message", key="value", number=123)

            # ログディレクトリにファイルが作成されることを確認
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) > 0, f"No JSON log files found in {tmpdir}"

    def test_singleton_behavior(self):
        """シングルトン動作のテスト"""
        logger1 = get_structured_logger("test_singleton")
        logger2 = get_structured_logger("test_singleton")

        # 同じインスタンスが返される
        assert logger1 is logger2

    def test_log_with_exception(self):
        """例外情報付きログのテスト"""
        logger = get_structured_logger("test_logger")

        try:
            _ = 1 / 0
        except ZeroDivisionError:
            # 例外情報付きでログ（エラーが発生しないことを確認）
            logger.error("Division by zero", exc_info=True)


class TestLoggerIntegration:
    """ロガー統合のテスト"""

    def test_multiple_loggers(self):
        """複数ロガーの動作テスト"""
        logger1 = get_structured_logger("logger1")
        logger2 = get_structured_logger("logger2")

        # 異なるロガーは異なるインスタンス
        assert logger1 is not logger2
        assert logger1.name != logger2.name

        # それぞれ独立して動作
        logger1.info("Logger 1 message")
        logger2.info("Logger 2 message")

    def test_performance_tracking(self):
        """パフォーマンストラッキングのテスト"""
        import time

        logger = get_structured_logger("perf_test")

        # 処理時間を記録
        start_time = time.time()
        time.sleep(0.1)  # 100ms
        duration = time.time() - start_time

        logger.log_performance(operation="sleep_test", duration=duration, expected_duration=0.1)

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_log_levels_filtering(self, level):
        """ログレベルフィルタリングのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(name="filter_test", log_dir=Path(tmpdir), level=level)

            # 各レベルでログ出力を試行
            logger.debug("Debug")
            logger.info("Info")
            logger.warning("Warning")
            logger.error("Error")
            logger.critical("Critical")
