"""
システム統合モジュール
全コンポーネントの統合と協調動作を提供
"""

# 重い依存関係の遅延読み込み
SystemIntegrator = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    global SystemIntegrator

    try:
        from .system_integrator import SystemIntegrator
    except ImportError:
        # 依存関係が不足している場合はスキップ
        pass


__all__ = ["SystemIntegrator"]
