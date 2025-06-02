"""
システム統合モジュール
全コンポーネントの統合と協調動作を提供
"""

# 重い依存関係の遅延読み込み
SystemIntegrator = None


def _lazy_import():
    """重い依存関係の遅延読み込み"""
    import contextlib

    global SystemIntegrator

    with contextlib.suppress(ImportError):
        from .system_integrator import SystemIntegrator


def __getattr__(name):
    if name == "SystemIntegrator":
        _lazy_import()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SystemIntegrator"]
