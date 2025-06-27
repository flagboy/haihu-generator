"""
pytestの共通設定
"""

import os

# OpenCVのインポートエラーを回避するための設定
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
