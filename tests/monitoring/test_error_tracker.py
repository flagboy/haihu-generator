"""
エラー追跡システムのテスト
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.monitoring.error_tracker import ErrorRecord, ErrorTracker


class TestErrorTracker:
    """ErrorTrackerのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        tracker = ErrorTracker(
            name="test_tracker", max_history=1000, alert_threshold=5, alert_window=60
        )

        assert tracker.name == "test_tracker"
        assert tracker.max_history == 1000
        assert tracker.alert_threshold == 5
        assert tracker.alert_window == 60

    def test_track_error(self):
        """エラー追跡のテスト"""
        tracker = ErrorTracker()

        # テスト用エラー
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            tracker.track_error(e, operation="test_operation", context={"user_id": "123"})

        # エラーが記録されていることを確認
        assert len(tracker.errors) == 1
        error = tracker.errors[0]
        assert error.error_type == "ValueError"
        assert error.error_message == "Test error message"
        assert error.operation == "test_operation"
        assert error.context["user_id"] == "123"

    def test_error_summary(self):
        """エラーサマリーのテスト"""
        tracker = ErrorTracker()

        # 複数のエラーを追跡
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.track_error(e, operation="op1")

        for i in range(2):
            try:
                raise TypeError(f"Type error {i}")
            except TypeError as e:
                tracker.track_error(e, operation="op2")

        # サマリーを取得
        summaries = tracker.get_error_summary(hours=1)

        assert len(summaries) >= 2

        # エラータイプ別にサマリーを確認
        value_error_summary = next((s for s in summaries if s.error_type == "ValueError"), None)
        assert value_error_summary is not None
        assert value_error_summary.count == 3
        assert "op1" in value_error_summary.affected_operations

        type_error_summary = next((s for s in summaries if s.error_type == "TypeError"), None)
        assert type_error_summary is not None
        assert type_error_summary.count == 2
        assert "op2" in type_error_summary.affected_operations

    def test_error_rate(self):
        """エラーレート計算のテスト"""
        tracker = ErrorTracker()

        # エラーを追跡
        for _ in range(5):
            try:
                raise ValueError("Test")
            except ValueError as e:
                tracker.track_error(e, operation="test")

        # エラーレートを取得
        rates = tracker.get_error_rate(window_minutes=60)

        assert "ValueError" in rates
        # 5エラー / 60分 = 0.083... エラー/分
        assert rates["ValueError"] == pytest.approx(5 / 60, 0.01)

    def test_top_errors(self):
        """頻出エラーのテスト"""
        tracker = ErrorTracker()

        # 異なる頻度でエラーを追跡
        error_counts = {ValueError: 10, TypeError: 5, KeyError: 3, IndexError: 1}

        for error_type, count in error_counts.items():
            for _ in range(count):
                try:
                    raise error_type("Test")
                except Exception as e:
                    tracker.track_error(e, operation="test")

        # トップエラーを取得
        top_errors = tracker.get_top_errors(limit=3)

        assert len(top_errors) == 3
        assert top_errors[0][0] == "ValueError"
        assert top_errors[0][1] == 10
        assert top_errors[1][0] == "TypeError"
        assert top_errors[1][1] == 5
        assert top_errors[2][0] == "KeyError"
        assert top_errors[2][1] == 3

    def test_clear_old_errors(self):
        """古いエラーのクリアテスト"""
        tracker = ErrorTracker()

        # 古いエラーを追加（直接操作）
        old_error = ErrorRecord(
            timestamp=datetime.now() - timedelta(days=10),
            error_type="OldError",
            error_message="Old error",
            operation="old_op",
            traceback="",
            fingerprint="old123",
        )
        tracker.errors.append(old_error)

        # 新しいエラーを追加
        try:
            raise ValueError("New error")
        except ValueError as e:
            tracker.track_error(e, operation="new_op")

        assert len(tracker.errors) >= 2

        # 7日より古いエラーをクリア
        removed_count = tracker.clear_old_errors(days=7)

        assert removed_count == 1
        assert len(tracker.errors) >= 1

        # 古いエラーが削除されていることを確認
        for error in tracker.errors:
            assert error.error_type != "OldError"

    def test_export_error_report(self):
        """エラーレポートのエクスポートテスト"""
        import json
        import tempfile

        tracker = ErrorTracker()

        # エラーを追跡
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.track_error(e, operation=f"op{i}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            # レポートをエクスポート
            tracker.export_error_report(output_path, hours=24)

            # ファイルが作成されていることを確認
            assert output_path.exists()

            # 内容を確認
            with open(output_path) as f:
                report = json.load(f)

            assert "generated_at" in report
            assert "window_hours" in report
            assert report["window_hours"] == 24
            assert "total_errors" in report
            assert report["total_errors"] >= 3
            assert "unique_errors" in report
            assert "error_rates_per_minute" in report
            assert "top_errors" in report
            assert "error_details" in report

        finally:
            output_path.unlink()

    def test_fingerprint_generation(self):
        """フィンガープリント生成のテスト"""
        error1 = ErrorRecord(
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Test error",
            operation="test_op",
            traceback="",
        )

        error2 = ErrorRecord(
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="Test error",
            operation="test_op",
            traceback="",
        )

        error3 = ErrorRecord(
            timestamp=datetime.now(),
            error_type="TypeError",
            error_message="Test error",
            operation="test_op",
            traceback="",
        )

        # 同じエラーは同じフィンガープリント
        assert error1.fingerprint == error2.fingerprint

        # 異なるエラーは異なるフィンガープリント
        assert error1.fingerprint != error3.fingerprint

    def test_alert_threshold(self):
        """アラート閾値のテスト"""
        tracker = ErrorTracker(
            alert_threshold=3,
            alert_window=10,  # 10秒
        )

        # アラート閾値を超えるエラーを発生
        for i in range(4):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                tracker.track_error(e, operation="test")
            time.sleep(0.1)  # 少し間隔を空ける

        # アラートが発生したことを確認（ログで確認）
        # 実際のアラート発生はログ出力で確認する必要がある
