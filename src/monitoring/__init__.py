"""
モニタリングモジュール

システムのパフォーマンス監視、ログ収集、メトリクス管理を提供
"""

# 遅延インポートを使用して循環参照を回避
__all__ = [
    "StructuredLogger",
    "get_structured_logger",
    "MetricsCollector",
    "PerformanceTracker",
    "SystemMonitor",
    "get_global_metrics",
    "get_performance_tracker",
    "get_error_tracker",
    "get_system_monitor",
    "global_metrics",
    "performance_tracker",
    "error_tracker",
    "system_monitor",
]


def __getattr__(name):
    """遅延インポート"""
    if name == "StructuredLogger":
        from .logger import StructuredLogger

        return StructuredLogger
    elif name == "get_structured_logger":
        from .logger import get_structured_logger

        return get_structured_logger
    elif name == "MetricsCollector":
        from .metrics import MetricsCollector

        return MetricsCollector
    elif name == "PerformanceTracker":
        from .metrics import PerformanceTracker

        return PerformanceTracker
    elif name == "get_global_metrics":
        from .metrics import get_global_metrics

        return get_global_metrics
    elif name == "get_performance_tracker":
        from .metrics import get_performance_tracker

        return get_performance_tracker
    elif name == "get_error_tracker":
        from .error_tracker import get_error_tracker

        return get_error_tracker
    elif name == "get_system_monitor":
        from .system_monitor import get_system_monitor

        return get_system_monitor
    elif name == "global_metrics":
        from .metrics import get_global_metrics

        return get_global_metrics()
    elif name == "performance_tracker":
        from .metrics import get_performance_tracker

        return get_performance_tracker()
    elif name == "error_tracker":
        from .error_tracker import get_error_tracker

        return get_error_tracker()
    elif name == "SystemMonitor":
        from .system_monitor import SystemMonitor

        return SystemMonitor
    elif name == "system_monitor":
        from .system_monitor import get_system_monitor

        return get_system_monitor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
