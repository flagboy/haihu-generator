"""
pytestの共通設定
"""

import os
import sys

# OpenCVのインポートエラーを回避するための設定
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

# テスト時のみ、OpenCVのインポートを無効化
if "pytest" in sys.modules:
    sys.modules["cv2"] = None
