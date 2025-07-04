"""
エラー追跡システム

エラーの収集、分類、集計、アラート機能を提供
"""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from .metrics import MetricsCollector


@dataclass
class ErrorRecord:
    """エラーレコード"""

    timestamp: datetime
    error_type: str
    error_message: str
    operation: str
    traceback: str
    context: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self):
        """フィンガープリントを生成"""
        if not self.fingerprint:
            # エラーの一意性を判定するためのフィンガープリント
            content = f"{self.error_type}:{self.error_message}:{self.operation}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ErrorSummary:
    """エラーサマリー"""

    error_type: str
    count: int
    first_seen: datetime
    last_seen: datetime
    sample_message: str
    sample_traceback: str
    affected_operations: set[str]


class ErrorTracker:
    """エラー追跡クラス"""

    def __init__(
        self,
        name: str = "error_tracker",
        max_history: int = 10000,
        alert_threshold: int = 10,
        alert_window: int = 300,  # 5分
    ):
        """
        Args:
            name: トラッカー名
            max_history: 保持する履歴の最大数
            alert_threshold: アラート閾値（エラー数）
            alert_window: アラートウィンドウ（秒）
        """
        self.name = name
        self.max_history = max_history
        self.alert_threshold = alert_threshold
        self.alert_window = alert_window

        # ロガーの遅延初期化（循環参照を避けるため）
        self._logger: Any = None
        self.metrics = MetricsCollector(f"{name}_error_metrics")

        # エラー履歴
        self.errors: deque[ErrorRecord] = deque(maxlen=max_history)
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.error_rate: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))

        self.lock = Lock()
        self._last_alert_time: dict[str, float] = {}

    def _get_logger(self):
        """ロガーの遅延初期化"""
        if self._logger is None:
            from .logger import get_structured_logger

            self._logger = get_structured_logger(f"{self.name}_errors")
        return self._logger

    def track_error(
        self,
        error: Exception,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """エラーを追跡"""
        import traceback

        # 再帰呼び出しを防ぐためのフラグ
        if hasattr(self, "_tracking_error") and self._tracking_error:
            return

        self._tracking_error: bool = True

        try:
            error_record = ErrorRecord(
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                error_message=str(error),
                operation=operation,
                traceback=traceback.format_exc(),
                context=context or {},
            )

            with self.lock:
                # エラー履歴に追加
                self.errors.append(error_record)

                # カウントを更新
                self.error_counts[error_record.fingerprint] += 1

                # エラーレートを更新
                self.error_rate[error_record.error_type].append(time.time())

            # メトリクスを記録
            try:
                self.metrics.increment(f"error_{error_record.error_type}")
                self.metrics.increment(f"operation_{operation}_errors")
            except Exception:
                pass  # メトリクス記録エラーは無視

            # エラーをログ（ログエラーは無視）
            import contextlib

            with contextlib.suppress(Exception):
                self._get_logger().error(
                    f"Error tracked: {error_record.error_type} in {operation}",
                    error_type=error_record.error_type,
                    error_message=error_record.error_message,
                    operation=operation,
                    fingerprint=error_record.fingerprint,
                    context=context,
                    exc_info=True,
                )

            # アラートをチェック
            self._check_alerts(error_record)

        finally:
            self._tracking_error = False

    def _check_alerts(self, error_record: ErrorRecord) -> None:
        """アラートをチェック"""
        # エラーレートを計算
        current_time = time.time()
        recent_errors = [
            t
            for t in self.error_rate[error_record.error_type]
            if current_time - t < self.alert_window
        ]

        error_rate = len(recent_errors)

        # アラート閾値を超えているかチェック
        if error_rate >= self.alert_threshold:
            # 最後のアラートから十分時間が経過しているか
            last_alert = self._last_alert_time.get(error_record.error_type, 0)
            if current_time - last_alert > self.alert_window:
                self._send_alert(error_record, error_rate)
                self._last_alert_time[error_record.error_type] = current_time

    def _send_alert(self, error_record: ErrorRecord, error_rate: int) -> None:
        """アラートを送信"""
        self._get_logger().critical(
            f"High error rate alert: {error_record.error_type}",
            error_type=error_record.error_type,
            error_rate=error_rate,
            alert_window=self.alert_window,
            threshold=self.alert_threshold,
            operation=error_record.operation,
            message=error_record.error_message,
        )

        self.metrics.increment("error_alerts")

    def get_error_summary(
        self, hours: int = 24, error_type: str | None = None
    ) -> list[ErrorSummary]:
        """エラーサマリーを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            # 指定期間内のエラーを抽出
            recent_errors = [
                e
                for e in self.errors
                if e.timestamp > cutoff_time and (error_type is None or e.error_type == error_type)
            ]

        # エラーをグループ化
        error_groups: dict[str, list[ErrorRecord]] = defaultdict(list)
        for error in recent_errors:
            error_groups[error.fingerprint].append(error)

        # サマリーを作成
        summaries = []
        for _fingerprint, errors in error_groups.items():
            if errors:
                first_error = errors[0]
                summaries.append(
                    ErrorSummary(
                        error_type=first_error.error_type,
                        count=len(errors),
                        first_seen=min(e.timestamp for e in errors),
                        last_seen=max(e.timestamp for e in errors),
                        sample_message=first_error.error_message,
                        sample_traceback=first_error.traceback,
                        affected_operations={e.operation for e in errors},
                    )
                )

        # カウントでソート
        summaries.sort(key=lambda s: s.count, reverse=True)

        return summaries

    def get_error_rate(
        self, error_type: str | None = None, window_minutes: int = 60
    ) -> dict[str, float]:
        """エラーレートを取得"""
        current_time = time.time()
        window_seconds = window_minutes * 60

        rates = {}

        with self.lock:
            error_types = [error_type] if error_type else list(self.error_rate.keys())

            for err_type in error_types:
                recent_errors = [
                    t for t in self.error_rate[err_type] if current_time - t < window_seconds
                ]
                # エラー/分として計算
                rates[err_type] = len(recent_errors) / window_minutes if window_minutes > 0 else 0

        return rates

    def get_top_errors(self, limit: int = 10, hours: int = 24) -> list[tuple[str, int]]:
        """頻出エラーを取得"""
        summaries = self.get_error_summary(hours=hours)

        # エラータイプ別に集計
        error_counts: defaultdict[str, int] = defaultdict(int)
        for summary in summaries:
            error_counts[summary.error_type] += summary.count

        # 上位N件を返す
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def clear_old_errors(self, days: int = 7) -> int:
        """古いエラーをクリア"""
        cutoff_time = datetime.now() - timedelta(days=days)

        with self.lock:
            original_count = len(self.errors)

            # 新しいdequeを作成して古いエラーを除外
            new_errors = deque(
                (e for e in self.errors if e.timestamp > cutoff_time), maxlen=self.max_history
            )
            self.errors = new_errors

            removed_count = original_count - len(self.errors)

        if removed_count > 0:
            self._get_logger().info(
                f"Cleared {removed_count} old errors", removed_count=removed_count, days=days
            )

        return removed_count

    def export_error_report(self, output_path: Path, hours: int = 24) -> None:
        """エラーレポートをエクスポート"""
        import json

        summaries = self.get_error_summary(hours=hours)
        rates = self.get_error_rate(window_minutes=60)
        top_errors = self.get_top_errors(limit=20, hours=hours)

        report = {
            "generated_at": datetime.now().isoformat(),
            "window_hours": hours,
            "total_errors": sum(s.count for s in summaries),
            "unique_errors": len(summaries),
            "error_rates_per_minute": {k: round(v, 2) for k, v in rates.items()},
            "top_errors": [{"type": err_type, "count": count} for err_type, count in top_errors],
            "error_details": [
                {
                    "error_type": s.error_type,
                    "count": s.count,
                    "first_seen": s.first_seen.isoformat(),
                    "last_seen": s.last_seen.isoformat(),
                    "sample_message": s.sample_message,
                    "affected_operations": list(s.affected_operations),
                }
                for s in summaries[:50]  # 最大50件
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self._get_logger().info(
            "Error report exported",
            output_path=str(output_path),
            total_errors=report["total_errors"],
            unique_errors=report["unique_errors"],
        )


# グローバルインスタンス（遅延初期化）
_error_tracker = None


def get_error_tracker():
    """グローバルエラートラッカーを取得"""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


# 互換性のためのエイリアス
error_tracker = None
