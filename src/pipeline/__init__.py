"""
データ処理パイプラインモジュール
"""

# 重い依存関係の遅延読み込み
AIPipeline = None
GamePipeline = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    global AIPipeline, GamePipeline

    try:
        from .ai_pipeline import AIPipeline
        from .game_pipeline import GamePipeline
    except ImportError:
        # 依存関係が不足している場合はスキップ
        pass


def __getattr__(name):
    if name in __all__:
        _lazy_import()
        return globals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AIPipeline", "GamePipeline"]
