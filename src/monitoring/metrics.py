"""
メトリクス収集システム

パフォーマンスメトリクスの収集、集計、レポート機能を提供
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import psutil

from .logger import get_structured_logger


@dataclass
class Metric:
    """メトリクスデータ"""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """メトリクスサマリー"""

    count: int
    mean: float
    min: float
    max: float
    std: float
    percentiles: dict[int, float]

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricsSummary":
        """値のリストからサマリーを作成"""
        if not values:
            return cls(0, 0.0, 0.0, 0.0, 0.0, {})

        arr = np.array(values)
        return cls(
            count=len(values),
            mean=float(np.mean(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            std=float(np.std(arr)),
            percentiles={
                50: float(np.percentile(arr, 50)),
                75: float(np.percentile(arr, 75)),
                90: float(np.percentile(arr, 90)),
                95: float(np.percentile(arr, 95)),
                99: float(np.percentile(arr, 99)),
            },
        )


class MetricsCollector:
    """メトリクス収集クラス"""

    def __init__(
        self,
        name: str = "metrics",
        max_history: int = 10000,
        flush_interval: int = 60,
    ):
        """
        Args:
            name: コレクター名
            max_history: 保持する履歴の最大数
            flush_interval: フラッシュ間隔（秒）
        """
        self.name = name
        self.max_history = max_history
        self.flush_interval = flush_interval

        self.logger = get_structured_logger(f"{name}_metrics")
        self.metrics: dict[str, deque[Metric]] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = Lock()

        self._last_flush = time.time()

    def record(
        self, name: str, value: float, tags: dict[str, str] | None = None, **metadata: Any
    ) -> None:
        """メトリクスを記録"""
        metric = Metric(name=name, value=value, tags=tags or {}, metadata=metadata)

        with self.lock:
            self.metrics[name].append(metric)

        # 定期的にフラッシュ
        if time.time() - self._last_flush > self.flush_interval:
            self.flush()

    def increment(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        """カウンターをインクリメント"""
        with self.lock:
            # 最新の値を取得
            if name in self.metrics and self.metrics[name]:
                last_value = self.metrics[name][-1].value
            else:
                last_value = 0.0

            self.record(name, last_value + value, tags)

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """ゲージ値を設定"""
        self.record(name, value, tags)

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None):
        """タイマーコンテキストマネージャー"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record(f"{name}_duration_seconds", duration, tags)

    def get_summary(self, name: str, window_seconds: int | None = None) -> MetricsSummary | None:
        """メトリクスのサマリーを取得"""
        with self.lock:
            if name not in self.metrics:
                return None

            metrics = list(self.metrics[name])

            if window_seconds:
                cutoff_time = datetime.now().timestamp() - window_seconds
                metrics = [m for m in metrics if m.timestamp.timestamp() > cutoff_time]

            if not metrics:
                return None

            values = [m.value for m in metrics]
            return MetricsSummary.from_values(values)

    def get_all_summaries(self, window_seconds: int | None = None) -> dict[str, MetricsSummary]:
        """全メトリクスのサマリーを取得"""
        summaries = {}
        for name in list(self.metrics.keys()):
            summary = self.get_summary(name, window_seconds)
            if summary:
                summaries[name] = summary
        return summaries

    def flush(self) -> None:
        """メトリクスをログに出力"""
        summaries = self.get_all_summaries(window_seconds=self.flush_interval)

        if summaries:
            self.logger.info(
                "Metrics summary",
                metrics_count=len(summaries),
                summaries={
                    name: {
                        "count": s.count,
                        "mean": round(s.mean, 4),
                        "min": round(s.min, 4),
                        "max": round(s.max, 4),
                        "p50": round(s.percentiles.get(50, 0), 4),
                        "p95": round(s.percentiles.get(95, 0), 4),
                        "p99": round(s.percentiles.get(99, 0), 4),
                    }
                    for name, s in summaries.items()
                },
            )

        self._last_flush = time.time()

    def export_to_file(self, output_path: Path) -> None:
        """メトリクスをファイルにエクスポート"""
        import json

        with self.lock:
            data = {"collector": self.name, "timestamp": datetime.now().isoformat(), "metrics": {}}

            for name, metrics in self.metrics.items():
                data["metrics"][name] = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "tags": m.tags,
                        "metadata": m.metadata,
                    }
                    for m in metrics
                ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class PerformanceTracker:
    """パフォーマンストラッカー"""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        """
        Args:
            metrics_collector: メトリクスコレクター
        """
        self.metrics = metrics_collector or MetricsCollector("performance")
        self.logger = get_structured_logger("performance_tracker")

    @contextmanager
    def track(
        self,
        operation: str,
        tags: dict[str, str] | None = None,
        log_threshold: float = 1.0,  # 秒
    ):
        """操作のパフォーマンスを追跡"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
            success = True
        except Exception as e:
            success = False
            self.logger.error(
                f"Operation failed: {operation}", operation=operation, error=str(e), exc_info=True
            )
            raise
        finally:
            # 実行時間
            duration = time.time() - start_time

            # メモリ使用量
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory

            # メトリクスを記録
            self.metrics.record(f"operation_{operation}_duration", duration, tags)
            self.metrics.record(f"operation_{operation}_memory_delta_mb", memory_delta, tags)
            self.metrics.increment(
                f"operation_{operation}_{'success' if success else 'failure'}", tags=tags
            )

            # 閾値を超えた場合はログ出力
            if duration > log_threshold:
                self.logger.warning(
                    f"Slow operation: {operation}",
                    operation=operation,
                    duration_seconds=duration,
                    memory_delta_mb=memory_delta,
                    success=success,
                    tags=tags,
                )

    def record_batch_processing(
        self,
        batch_size: int,
        processing_time: float,
        success_count: int,
        failure_count: int,
        **metadata: Any,
    ) -> None:
        """バッチ処理のメトリクスを記録"""
        total_count = success_count + failure_count
        success_rate = success_count / total_count if total_count > 0 else 0.0

        self.metrics.record("batch_size", batch_size)
        self.metrics.record("batch_processing_time", processing_time)
        self.metrics.record("batch_success_rate", success_rate)
        self.metrics.record(
            "batch_throughput", total_count / processing_time if processing_time > 0 else 0
        )

        self.logger.info(
            "Batch processing completed",
            batch_size=batch_size,
            processing_time=processing_time,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_rate,
            **metadata,
        )

    def record_model_inference(
        self,
        model_name: str,
        inference_time: float,
        batch_size: int = 1,
        device: str = "cpu",
        **metadata: Any,
    ) -> None:
        """モデル推論のメトリクスを記録"""
        self.metrics.record(
            f"model_{model_name}_inference_time", inference_time, tags={"device": device}
        )

    def track_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        items_processed: int | None = None,
        **metadata: Any,
    ) -> None:
        """オペレーションのパフォーマンスを記録"""
        self.metrics.record(f"{operation}_duration_seconds", duration)
        self.metrics.increment(f"{operation}_{'success' if success else 'failure'}")

        if items_processed is not None:
            self.metrics.record(f"{operation}_items_processed", items_processed)
            if duration > 0:
                self.metrics.record(f"{operation}_throughput", items_processed / duration)

        self.logger.log_performance(
            operation=operation,
            duration=duration,
            success=success,
            items_processed=items_processed,
            **metadata,
        )

    def track_batch_processing(
        self,
        batch_size: int,
        processing_time: float,
        success_count: int,
        error_count: int,
        **metadata: Any,
    ) -> None:
        """バッチ処理のメトリクスを記録"""
        self.record_batch_processing(
            batch_size=batch_size,
            processing_time=processing_time,
            success_count=success_count,
            failure_count=error_count,
            **metadata,
        )

    @contextmanager
    def measure(self, operation: str, **tags):
        """コンテキストマネージャーで操作を計測"""
        with self.track(operation, tags=tags):
            yield


# グローバルインスタンス
global_metrics = MetricsCollector("global")
performance_tracker = PerformanceTracker(global_metrics)
