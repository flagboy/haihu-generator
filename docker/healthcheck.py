#!/usr/bin/env python3
"""
Docker ヘルスチェックスクリプト
システムの健全性をチェック
"""

import sys
from pathlib import Path

# アプリケーションパスを追加
sys.path.insert(0, "/app")

try:
    from src.utils.config import ConfigManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def check_config():
    """設定ファイルの確認"""
    try:
        config_manager = ConfigManager("/app/config.yaml")
        config = config_manager.get_config()
        return config is not None
    except Exception as e:
        print(f"Config check failed: {e}")
        return False


def check_directories():
    """必要なディレクトリの確認"""
    required_dirs = [
        "/app/data/input",
        "/app/data/output",
        "/app/data/temp",
        "/app/logs",
        "/app/models",
    ]

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"Required directory missing: {dir_path}")
            return False

    return True


def check_python_modules():
    """必要なPythonモジュールの確認"""
    required_modules = ["numpy", "opencv-python", "pillow", "pyyaml", "pandas", "loguru"]

    for module in required_modules:
        try:
            __import__(module.replace("-", "_"))
        except ImportError:
            print(f"Required module missing: {module}")
            return False

    return True


def check_system_resources():
    """システムリソースの確認"""
    try:
        import psutil

        # メモリ使用量チェック
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            print(f"Memory usage too high: {memory.percent}%")
            return False

        # ディスク容量チェック
        disk = psutil.disk_usage("/app")
        disk_usage_percent = (disk.used / disk.total) * 100
        if disk_usage_percent > 95:
            print(f"Disk usage too high: {disk_usage_percent:.1f}%")
            return False

        return True

    except ImportError:
        # psutilが利用できない場合はスキップ
        return True
    except Exception as e:
        print(f"System resource check failed: {e}")
        return False


def main():
    """メインヘルスチェック関数"""
    checks = [
        ("Config", check_config),
        ("Directories", check_directories),
        ("Python Modules", check_python_modules),
        ("System Resources", check_system_resources),
    ]

    all_passed = True

    for check_name, check_func in checks:
        try:
            if check_func():
                print(f"✓ {check_name}: OK")
            else:
                print(f"✗ {check_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name}: ERROR - {e}")
            all_passed = False

    if all_passed:
        print("Health check: PASSED")
        sys.exit(0)
    else:
        print("Health check: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
