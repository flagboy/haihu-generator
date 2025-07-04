"""
モニタリングテスト用の設定
"""

import os

import pytest

# テスト実行時はモニタリングの自動初期化を無効化
os.environ["DISABLE_MONITORING_AUTO_INIT"] = "1"


@pytest.fixture(autouse=True)
def disable_monitoring_logging(monkeypatch):
    """テスト時のモニタリングログを無効化"""
    # ロガーのファイル出力を無効化
    monkeypatch.setenv("MONITORING_LOG_TO_FILE", "false")
    monkeypatch.setenv("MONITORING_LOG_TO_CONSOLE", "false")
