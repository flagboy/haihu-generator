"""
麻雀牌譜作成システム

動画から麻雀の牌譜を自動生成するシステムのメインパッケージ
"""

__version__ = "0.2.0"  # フェーズ2対応
__author__ = "Mahjong System Developer"

# 基本モジュールのインポート
from .utils.config import ConfigManager
from .utils.logger import get_logger, setup_logger
from .utils.tile_definitions import TileDefinitions, TileType
from .video.video_processor import VideoProcessor

# AI/MLモジュールのインポート（フェーズ2）
from .detection.tile_detector import TileDetector
from .classification.tile_classifier import TileClassifier
from .pipeline.ai_pipeline import AIPipeline
from .models.model_manager import ModelManager

__all__ = [
    # 基本モジュール
    "ConfigManager",
    "get_logger", 
    "setup_logger",
    "TileDefinitions",
    "TileType", 
    "VideoProcessor",
    # AI/MLモジュール
    "TileDetector",
    "TileClassifier",
    "AIPipeline",
    "ModelManager"
]