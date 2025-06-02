"""
麻雀牌分類モジュール
"""

# 重い依存関係の遅延読み込み
TileClassifier = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    import contextlib

    global TileClassifier

    with contextlib.suppress(ImportError):
        from .tile_classifier import TileClassifier


def __getattr__(name):
    if name == "TileClassifier":
        _lazy_import()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TileClassifier"]
