"""
パフォーマンス最適化モジュール
システム全体のパフォーマンスを監視・最適化
"""

import gc
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""

    cpu_usage: float
    memory_usage: float
    memory_available: float
    gpu_usage: float | None
    gpu_memory: float | None
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    process_count: int
    thread_count: int
    timestamp: float


@dataclass
class OptimizationResult:
    """最適化結果"""

    success: bool
    optimization_type: str
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percentage: float
    recommendations: list[str]
    warnings: list[str]


class PerformanceOptimizer:
    """パフォーマンス最適化クラス"""

    def __init__(self, config_manager: ConfigManager):
        """
        初期化

        Args:
            config_manager: 設定管理オブジェクト
        """
        self.config = config_manager
        self.logger = get_logger(__name__)

        # 最適化設定
        self.optimization_config = self._load_optimization_config()

        # メトリクス履歴
        self.metrics_history: list[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None

        # 最適化状態
        self.optimizations_applied: list[str] = []
        self.baseline_metrics: PerformanceMetrics | None = None

        # GPU情報
        self.gpu_available = self._check_gpu_availability()

        self.logger.info("PerformanceOptimizer initialized")

    def _load_optimization_config(self) -> dict[str, Any]:
        """最適化設定を読み込み"""
        system_config = self.config.get_config().get("system", {})

        return {
            "max_workers": system_config.get("max_workers", mp.cpu_count()),
            "memory_limit": system_config.get("memory_limit", "8GB"),
            "gpu_enabled": system_config.get("gpu_enabled", True),
            "monitoring_interval": 1.0,  # 秒
            "optimization_thresholds": {
                "cpu_usage": 80.0,  # %
                "memory_usage": 85.0,  # %
                "gpu_usage": 90.0,  # %
            },
            "batch_size_limits": {"min": 1, "max": 32, "auto_adjust": True},
        }

    def _check_gpu_availability(self) -> bool:
        """GPU利用可能性をチェック"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf

                return len(tf.config.list_physical_devices("GPU")) > 0
            except ImportError:
                return False

    def get_current_metrics(self) -> PerformanceMetrics:
        """現在のパフォーマンスメトリクスを取得"""
        try:
            # CPU・メモリ情報
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # ディスクI/O
            disk_io = psutil.disk_io_counters()
            disk_read = disk_io.read_bytes if disk_io else 0
            disk_write = disk_io.write_bytes if disk_io else 0

            # ネットワークI/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent if net_io else 0
            net_recv = net_io.bytes_recv if net_io else 0

            # プロセス・スレッド数
            process_count = len(psutil.pids())
            thread_count = threading.active_count()

            # GPU情報
            gpu_usage = None
            gpu_memory = None

            if self.gpu_available:
                gpu_usage, gpu_memory = self._get_gpu_metrics()

            metrics = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory,
                disk_io_read=disk_read,
                disk_io_write=disk_write,
                network_io_sent=net_sent,
                network_io_recv=net_recv,
                process_count=process_count,
                thread_count=thread_count,
                timestamp=time.time(),
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0.0,
                gpu_usage=None,
                gpu_memory=None,
                disk_io_read=0,
                disk_io_write=0,
                network_io_sent=0,
                network_io_recv=0,
                process_count=0,
                thread_count=0,
                timestamp=time.time(),
            )

    def _get_gpu_metrics(self) -> tuple[float | None, float | None]:
        """GPU メトリクスを取得"""
        try:
            import torch

            if torch.cuda.is_available():
                # GPU使用率（簡易版）
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else None

                # GPU メモリ使用率
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )  # GB
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

                return gpu_usage, gpu_memory_percent
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # GPU使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = utilization.gpu

            # GPU メモリ使用率
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_percent = (memory_info.used / memory_info.total) * 100

            return gpu_usage, gpu_memory_percent

        except Exception:
            pass

        return None, None

    def start_monitoring(self):
        """パフォーマンス監視を開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """パフォーマンス監視を停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)

                # 履歴サイズ制限
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]

                # 自動最適化チェック
                self._check_auto_optimization(metrics)

                time.sleep(self.optimization_config["monitoring_interval"])

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)

    def _check_auto_optimization(self, metrics: PerformanceMetrics):
        """自動最適化をチェック"""
        thresholds = self.optimization_config["optimization_thresholds"]

        # CPU使用率チェック
        if metrics.cpu_usage > thresholds["cpu_usage"]:
            self.logger.warning(f"High CPU usage detected: {metrics.cpu_usage:.1f}%")
            self._optimize_cpu_usage()

        # メモリ使用率チェック
        if metrics.memory_usage > thresholds["memory_usage"]:
            self.logger.warning(f"High memory usage detected: {metrics.memory_usage:.1f}%")
            self._optimize_memory_usage()

        # GPU使用率チェック
        if metrics.gpu_usage and metrics.gpu_usage > thresholds["gpu_usage"]:
            self.logger.warning(f"High GPU usage detected: {metrics.gpu_usage:.1f}%")
            self._optimize_gpu_usage()

    def optimize_system(self) -> OptimizationResult:
        """システム全体を最適化"""
        self.logger.info("Starting system optimization...")

        before_metrics = self.get_current_metrics()
        if self.baseline_metrics is None:
            self.baseline_metrics = before_metrics

        recommendations = []
        warnings = []

        try:
            # メモリ最適化
            self._optimize_memory_usage()
            recommendations.append("Memory optimization applied")

            # CPU最適化
            self._optimize_cpu_usage()
            recommendations.append("CPU optimization applied")

            # GPU最適化（利用可能な場合）
            if self.gpu_available:
                self._optimize_gpu_usage()
                recommendations.append("GPU optimization applied")

            # ガベージコレクション
            gc.collect()
            recommendations.append("Garbage collection performed")

            # 少し待ってから測定
            time.sleep(1.0)
            after_metrics = self.get_current_metrics()

            # 改善率計算
            improvement = self._calculate_improvement(before_metrics, after_metrics)

            result = OptimizationResult(
                success=True,
                optimization_type="full_system",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement,
                recommendations=recommendations,
                warnings=warnings,
            )

            self.logger.info(f"System optimization completed. Improvement: {improvement:.1f}%")
            return result

        except Exception as e:
            self.logger.error(f"System optimization failed: {e}")
            warnings.append(f"Optimization error: {str(e)}")

            return OptimizationResult(
                success=False,
                optimization_type="full_system",
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                improvement_percentage=0.0,
                recommendations=recommendations,
                warnings=warnings,
            )

    def _optimize_memory_usage(self):
        """メモリ使用量を最適化"""
        try:
            # ガベージコレクション
            gc.collect()

            # NumPy配列のメモリ最適化
            if hasattr(np, "seterr"):
                np.seterr(all="ignore")

            self.optimizations_applied.append("memory_optimization")
            self.logger.debug("Memory optimization applied")

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")

    def _optimize_cpu_usage(self):
        """CPU使用率を最適化"""
        try:
            # 並列処理数を調整
            cpu_count = mp.cpu_count()
            current_workers = self.optimization_config["max_workers"]

            # CPU使用率が高い場合は並列数を減らす
            current_metrics = self.get_current_metrics()
            if current_metrics.cpu_usage > 80:
                new_workers = max(1, current_workers - 1)
                self.optimization_config["max_workers"] = new_workers
                self.logger.debug(f"Reduced max_workers to {new_workers}")

            self.optimizations_applied.append("cpu_optimization")

        except Exception as e:
            self.logger.error(f"CPU optimization failed: {e}")

    def _optimize_gpu_usage(self):
        """GPU使用率を最適化"""
        if not self.gpu_available:
            return

        try:
            import torch

            if torch.cuda.is_available():
                # GPU メモリキャッシュをクリア
                torch.cuda.empty_cache()

                # GPU メモリ使用量を最適化
                torch.cuda.synchronize()

                self.optimizations_applied.append("gpu_optimization")
                self.logger.debug("GPU optimization applied")

        except Exception as e:
            self.logger.error(f"GPU optimization failed: {e}")

    def _calculate_improvement(
        self, before: PerformanceMetrics, after: PerformanceMetrics
    ) -> float:
        """改善率を計算"""
        try:
            # メモリ使用率の改善
            memory_improvement = max(0, before.memory_usage - after.memory_usage)

            # CPU使用率の改善
            cpu_improvement = max(0, before.cpu_usage - after.cpu_usage)

            # 総合改善率（重み付き平均）
            total_improvement = memory_improvement * 0.6 + cpu_improvement * 0.4

            return total_improvement

        except Exception:
            return 0.0

    def optimize_batch_size(
        self, current_batch_size: int, processing_time: float, memory_usage: float
    ) -> int:
        """バッチサイズを最適化"""
        try:
            limits = self.optimization_config["batch_size_limits"]

            if not limits["auto_adjust"]:
                return current_batch_size

            # メモリ使用率に基づく調整
            if memory_usage > 85:
                # メモリ使用率が高い場合はバッチサイズを減らす
                new_size = max(limits["min"], current_batch_size // 2)
            elif memory_usage < 50 and processing_time < 1.0:
                # メモリに余裕があり処理が速い場合はバッチサイズを増やす
                new_size = min(limits["max"], current_batch_size * 2)
            else:
                new_size = current_batch_size

            if new_size != current_batch_size:
                self.logger.debug(f"Batch size optimized: {current_batch_size} -> {new_size}")

            return new_size

        except Exception as e:
            self.logger.error(f"Batch size optimization failed: {e}")
            return current_batch_size

    def get_optimization_recommendations(self) -> list[str]:
        """最適化推奨事項を取得"""
        recommendations = []

        if not self.metrics_history:
            return ["Start monitoring to get recommendations"]

        # 最新のメトリクス
        latest_metrics = self.metrics_history[-1]

        # CPU使用率チェック
        if latest_metrics.cpu_usage > 80:
            recommendations.append("CPU使用率が高いです。並列処理数を減らすことを検討してください")

        # メモリ使用率チェック
        if latest_metrics.memory_usage > 85:
            recommendations.append(
                "メモリ使用率が高いです。バッチサイズを減らすか、不要なデータを削除してください"
            )

        # GPU使用率チェック
        if latest_metrics.gpu_usage and latest_metrics.gpu_usage > 90:
            recommendations.append("GPU使用率が高いです。GPU メモリキャッシュをクリアしてください")

        # 利用可能メモリチェック
        if latest_metrics.memory_available < 1.0:  # 1GB未満
            recommendations.append(
                "利用可能メモリが少ないです。他のアプリケーションを終了することを検討してください"
            )

        # 履歴に基づく推奨
        if len(self.metrics_history) > 10:
            avg_cpu = np.mean([m.cpu_usage for m in self.metrics_history[-10:]])
            if avg_cpu > 70:
                recommendations.append(
                    "平均CPU使用率が高いです。システム設定の見直しを検討してください"
                )

        return recommendations if recommendations else ["システムは最適な状態で動作しています"]

    def export_metrics(self, output_path: str):
        """メトリクス履歴をエクスポート"""
        try:
            import json

            metrics_data = []
            for metrics in self.metrics_history:
                metrics_data.append(
                    {
                        "timestamp": metrics.timestamp,
                        "cpu_usage": metrics.cpu_usage,
                        "memory_usage": metrics.memory_usage,
                        "memory_available": metrics.memory_available,
                        "gpu_usage": metrics.gpu_usage,
                        "gpu_memory": metrics.gpu_memory,
                        "disk_io_read": metrics.disk_io_read,
                        "disk_io_write": metrics.disk_io_write,
                        "network_io_sent": metrics.network_io_sent,
                        "network_io_recv": metrics.network_io_recv,
                        "process_count": metrics.process_count,
                        "thread_count": metrics.thread_count,
                    }
                )

            export_data = {
                "optimization_config": self.optimization_config,
                "optimizations_applied": self.optimizations_applied,
                "metrics_history": metrics_data,
                "export_timestamp": time.time(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Metrics exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")

    def get_performance_summary(self) -> dict[str, Any]:
        """パフォーマンス要約を取得"""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        try:
            latest = self.metrics_history[-1]

            # 統計計算
            cpu_values = [m.cpu_usage for m in self.metrics_history]
            memory_values = [m.memory_usage for m in self.metrics_history]

            summary = {
                "current_metrics": {
                    "cpu_usage": latest.cpu_usage,
                    "memory_usage": latest.memory_usage,
                    "memory_available_gb": latest.memory_available,
                    "gpu_usage": latest.gpu_usage,
                    "gpu_memory": latest.gpu_memory,
                },
                "statistics": {
                    "avg_cpu_usage": np.mean(cpu_values),
                    "max_cpu_usage": np.max(cpu_values),
                    "avg_memory_usage": np.mean(memory_values),
                    "max_memory_usage": np.max(memory_values),
                    "monitoring_duration": len(self.metrics_history)
                    * self.optimization_config["monitoring_interval"],
                },
                "optimizations_applied": self.optimizations_applied,
                "recommendations": self.get_optimization_recommendations(),
                "system_info": {
                    "cpu_count": mp.cpu_count(),
                    "gpu_available": self.gpu_available,
                    "max_workers": self.optimization_config["max_workers"],
                },
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}
