"""
麻雀牌譜作成システム

動画から麻雀の牌譜を自動生成するシステムのメインパッケージ
"""

__version__ = "0.2.0"  # フェーズ2対応
__author__ = "Mahjong System Developer"

# 基本ユーティリティのみインポート（重い依存関係は遅延読み込み）
from .utils.config import ConfigManager
from .utils.logger import get_logger, setup_logger
from .utils.tile_definitions import TileDefinitions, TileType

# 重い依存関係は遅延読み込み用のプレースホルダー
TileClassifier = None
TileDetector = None
ModelManager = None
AIPipeline = None
VideoProcessor = None

def _lazy_import():
    """重い依存関係の遅延読み込み"""
    global TileClassifier, TileDetector, ModelManager, AIPipeline, VideoProcessor

    try:
        from .classification.tile_classifier import TileClassifier
        from .detection.tile_detector import TileDetector
        from .models.model_manager import ModelManager
        from .pipeline.ai_pipeline import AIPipeline
        from .video.video_processor import VideoProcessor
    except ImportError:
        # 依存関係が不足している場合はスキップ
        pass

__all__ = [
    # 基本モジュール
    "ConfigManager",
    "get_logger",
    "setup_logger",
    "TileDefinitions",
    "TileType",
    # AI/MLモジュール（遅延読み込み）
    "TileDetector",
    "TileClassifier",
    "AIPipeline",
    "ModelManager",
    "VideoProcessor",
]
