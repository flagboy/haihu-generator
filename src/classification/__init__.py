"""
麻雀牌分類モジュール
"""

# 重い依存関係の遅延読み込み
TileClassifier = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    global TileClassifier

    try:
        from .tile_classifier import TileClassifier
    except ImportError:
        # 依存関係が不足している場合はスキップ
        pass


__all__ = ["TileClassifier"]
