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


__all__ = ["AIPipeline", "GamePipeline"]
