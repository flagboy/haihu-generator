"""
システムモニタリング

システムリソース（CPU、メモリ、GPU）の監視とヘルスチェック
"""

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ..utils.device_utils import get_device_memory_info
from .metrics import MetricsCollector


@dataclass
class SystemStatus:
    """システムステータス"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_utilization: float | None = None
    process_count: int = 0
    thread_count: int = 0


class SystemMonitor:
    """システムモニタークラス"""

    def __init__(
        self,
        interval: int = 60,
        metrics_collector: MetricsCollector | None = None,
        alert_thresholds: dict[str, float] | None = None,
    ):
        """
        Args:
            interval: 監視間隔（秒）
            metrics_collector: メトリクスコレクター
            alert_thresholds: アラート閾値
        """
        self.interval = interval
        self.metrics = metrics_collector or MetricsCollector("system")
        # ロガーの遅延初期化
        self._logger = None

        # デフォルトのアラート閾値
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "disk_usage_percent": 90.0,
            "gpu_memory_percent": 95.0,
        }

        self._monitoring = False
        self._thread: threading.Thread | None = None
        self._last_status: SystemStatus | None = None

    def _get_logger(self):
        """ロガーの遅延初期化"""
        if self._logger is None:
            from .logger import get_structured_logger

            self._logger = get_structured_logger("system_monitor")
        return self._logger

    def start(self) -> None:
        """監視を開始"""
        if self._monitoring:
            self._get_logger().warning("System monitoring is already running")
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._get_logger().info("System monitoring started")

    def stop(self) -> None:
        """監視を停止"""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5)
        self._get_logger().info("System monitoring stopped")

    def get_current_status(self) -> SystemStatus:
        """現在のシステムステータスを取得"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # メモリ情報
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        memory_available_mb = memory.available / 1024 / 1024

        # ディスク使用率
        disk = psutil.disk_usage("/")
        disk_usage_percent = disk.percent

        # プロセス情報
        process = psutil.Process()
        process_count = len(psutil.pids())
        thread_count = process.num_threads()

        # GPU情報（利用可能な場合）
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization = None

        try:
            import torch

            if torch.cuda.is_available():
                device_info = get_device_memory_info(torch.device("cuda"))
                if device_info:
                    gpu_memory_used_mb = device_info["allocated"] * 1024  # GB to MB
                    gpu_memory_total_mb = device_info["reserved"] * 1024
                    gpu_utilization = (
                        device_info["allocated"] / device_info["reserved"] * 100
                        if device_info["reserved"] > 0
                        else 0
                    )
        except ImportError:
            pass

        return SystemStatus(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            process_count=process_count,
            thread_count=thread_count,
        )

    def _monitor_loop(self) -> None:
        """監視ループ"""
        while self._monitoring:
            try:
                status = self.get_current_status()
                self._last_status = status

                # メトリクスを記録
                self._record_metrics(status)

                # アラートをチェック
                self._check_alerts(status)

                # ステータスをログ
                self._log_status(status)

            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e), exc_info=True)

            time.sleep(self.interval)

    def _record_metrics(self, status: SystemStatus) -> None:
        """メトリクスを記録"""
        self.metrics.gauge("system_cpu_percent", status.cpu_percent)
        self.metrics.gauge("system_memory_percent", status.memory_percent)
        self.metrics.gauge("system_memory_used_mb", status.memory_used_mb)
        self.metrics.gauge("system_memory_available_mb", status.memory_available_mb)
        self.metrics.gauge("system_disk_usage_percent", status.disk_usage_percent)
        self.metrics.gauge("system_process_count", status.process_count)
        self.metrics.gauge("system_thread_count", status.thread_count)

        if status.gpu_memory_used_mb is not None:
            self.metrics.gauge("system_gpu_memory_used_mb", status.gpu_memory_used_mb)
            self.metrics.gauge("system_gpu_memory_total_mb", status.gpu_memory_total_mb)
            self.metrics.gauge("system_gpu_utilization", status.gpu_utilization)

    def _check_alerts(self, status: SystemStatus) -> None:
        """アラートをチェック"""
        alerts = []

        # CPU使用率
        if status.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "cpu_high",
                    "message": f"CPU usage is high: {status.cpu_percent:.1f}%",
                    "value": status.cpu_percent,
                    "threshold": self.alert_thresholds["cpu_percent"],
                }
            )

        # メモリ使用率
        if status.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "memory_high",
                    "message": f"Memory usage is high: {status.memory_percent:.1f}%",
                    "value": status.memory_percent,
                    "threshold": self.alert_thresholds["memory_percent"],
                }
            )

        # ディスク使用率
        if status.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(
                {
                    "type": "disk_high",
                    "message": f"Disk usage is high: {status.disk_usage_percent:.1f}%",
                    "value": status.disk_usage_percent,
                    "threshold": self.alert_thresholds["disk_usage_percent"],
                }
            )

        # GPU使用率
        if status.gpu_utilization is not None:
            gpu_memory_percent = (
                status.gpu_memory_used_mb / status.gpu_memory_total_mb * 100
                if status.gpu_memory_total_mb > 0
                else 0
            )
            if gpu_memory_percent > self.alert_thresholds["gpu_memory_percent"]:
                alerts.append(
                    {
                        "type": "gpu_memory_high",
                        "message": f"GPU memory usage is high: {gpu_memory_percent:.1f}%",
                        "value": gpu_memory_percent,
                        "threshold": self.alert_thresholds["gpu_memory_percent"],
                    }
                )

        # アラートをログ出力
        for alert in alerts:
            self._get_logger().warning(
                alert["message"],
                alert_type=alert["type"],
                value=alert["value"],
                threshold=alert["threshold"],
            )
            self.metrics.increment(f"system_alert_{alert['type']}")

    def _log_status(self, status: SystemStatus) -> None:
        """ステータスをログ出力"""
        log_data = {
            "cpu_percent": round(status.cpu_percent, 1),
            "memory_percent": round(status.memory_percent, 1),
            "memory_used_mb": round(status.memory_used_mb, 1),
            "memory_available_mb": round(status.memory_available_mb, 1),
            "disk_usage_percent": round(status.disk_usage_percent, 1),
            "process_count": status.process_count,
            "thread_count": status.thread_count,
        }

        if status.gpu_memory_used_mb is not None:
            log_data.update(
                {
                    "gpu_memory_used_mb": round(status.gpu_memory_used_mb, 1),
                    "gpu_memory_total_mb": round(status.gpu_memory_total_mb, 1),
                    "gpu_utilization": round(status.gpu_utilization, 1),
                }
            )

        self._get_logger().info("System status", **log_data)

    def get_health_check(self) -> dict[str, Any]:
        """ヘルスチェック結果を取得"""
        if not self._last_status:
            return {
                "status": "unknown",
                "message": "No monitoring data available",
                "timestamp": datetime.now().isoformat(),
            }

        status = self._last_status
        issues = []

        # CPU使用率チェック
        if status.cpu_percent > self.alert_thresholds["cpu_percent"]:
            issues.append(f"High CPU usage: {status.cpu_percent:.1f}%")

        # メモリ使用率チェック
        if status.memory_percent > self.alert_thresholds["memory_percent"]:
            issues.append(f"High memory usage: {status.memory_percent:.1f}%")

        # ディスク使用率チェック
        if status.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            issues.append(f"High disk usage: {status.disk_usage_percent:.1f}%")

        # GPU使用率チェック
        if status.gpu_utilization is not None:
            gpu_memory_percent = (
                status.gpu_memory_used_mb / status.gpu_memory_total_mb * 100
                if status.gpu_memory_total_mb > 0
                else 0
            )
            if gpu_memory_percent > self.alert_thresholds["gpu_memory_percent"]:
                issues.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")

        health_status = "healthy" if not issues else "unhealthy"

        return {
            "status": health_status,
            "message": "; ".join(issues) if issues else "All systems operational",
            "timestamp": status.timestamp.isoformat(),
            "details": {
                "cpu_percent": status.cpu_percent,
                "memory_percent": status.memory_percent,
                "disk_usage_percent": status.disk_usage_percent,
                "gpu_utilization": status.gpu_utilization,
            },
        }

    def export_status_history(self, output_path: Path, hours: int = 24) -> None:
        """ステータス履歴をエクスポート"""
        # メトリクス履歴から再構築
        summaries = self.metrics.get_all_summaries(window_seconds=hours * 3600)

        data = {"export_time": datetime.now().isoformat(), "window_hours": hours, "summaries": {}}

        for metric_name, summary in summaries.items():
            data["summaries"][metric_name] = {
                "count": summary.count,
                "mean": summary.mean,
                "min": summary.min,
                "max": summary.max,
                "std": summary.std,
                "percentiles": summary.percentiles,
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# グローバルインスタンス（遅延初期化）
_system_monitor = None


def get_system_monitor():
    """グローバルシステムモニターを取得"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


# 互換性のためのエイリアス
system_monitor = None
