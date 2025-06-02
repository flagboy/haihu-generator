"""
メモリ最適化モジュール
メモリ使用量の監視と最適化を提供
"""

import gc
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

from ..utils.logger import get_logger


@dataclass
class MemoryInfo:
    """メモリ情報"""

    total_memory: float  # GB
    available_memory: float  # GB
    used_memory: float  # GB
    memory_percent: float
    process_memory: float  # GB
    gc_count: dict[str, int]


class MemoryOptimizer:
    """メモリ最適化クラス"""

    def __init__(self):
        """初期化"""
        self.logger = get_logger(__name__)
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.memory_history: list[MemoryInfo] = []

        # 最適化設定
        self.optimization_config = {
            "gc_threshold": 85.0,  # GC実行のメモリ使用率閾値
            "warning_threshold": 90.0,  # 警告を出すメモリ使用率閾値
            "critical_threshold": 95.0,  # 緊急対応が必要なメモリ使用率閾値
            "monitoring_interval": 5.0,  # 監視間隔（秒）
            "history_limit": 100,  # 履歴保持数
        }

        self.logger.info("MemoryOptimizer initialized")

    def get_memory_info(self) -> MemoryInfo:
        """現在のメモリ情報を取得"""
        try:
            # システムメモリ情報
            memory = psutil.virtual_memory()

            # プロセスメモリ情報
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**3)  # GB

            # ガベージコレクション情報
            gc_count = {f"generation_{i}": gc.get_count()[i] for i in range(len(gc.get_count()))}

            return MemoryInfo(
                total_memory=memory.total / (1024**3),
                available_memory=memory.available / (1024**3),
                used_memory=memory.used / (1024**3),
                memory_percent=memory.percent,
                process_memory=process_memory,
                gc_count=gc_count,
            )

        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            return MemoryInfo(
                total_memory=0.0,
                available_memory=0.0,
                used_memory=0.0,
                memory_percent=0.0,
                process_memory=0.0,
                gc_count={},
            )

    def optimize_memory(self) -> dict[str, Any]:
        """メモリを最適化"""
        before_info = self.get_memory_info()

        try:
            # ガベージコレクション実行
            collected_objects = []
            for generation in range(3):
                collected = gc.collect(generation)
                collected_objects.append(collected)

            # NumPy配列のメモリ最適化
            self._optimize_numpy_memory()

            # Pythonオブジェクトの最適化
            self._optimize_python_objects()

            # 少し待ってから再測定
            time.sleep(0.5)
            after_info = self.get_memory_info()

            # 最適化結果
            memory_freed = before_info.process_memory - after_info.process_memory

            result = {
                "success": True,
                "memory_freed_gb": memory_freed,
                "memory_freed_mb": memory_freed * 1024,
                "before_memory_percent": before_info.memory_percent,
                "after_memory_percent": after_info.memory_percent,
                "gc_collected_objects": collected_objects,
                "optimization_actions": [
                    "garbage_collection",
                    "numpy_optimization",
                    "python_object_optimization",
                ],
            }

            self.logger.info(f"Memory optimization completed. Freed: {memory_freed:.2f}GB")
            return result

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"success": False, "error": str(e), "memory_freed_gb": 0.0}

    def _optimize_numpy_memory(self):
        """NumPy配列のメモリ最適化"""
        try:
            # NumPyのメモリプールをクリア（可能な場合）
            if hasattr(np, "clear_cache"):
                np.clear_cache()

            # NumPyの警告設定を最適化
            np.seterr(all="ignore")

        except Exception as e:
            self.logger.debug(f"NumPy memory optimization failed: {e}")

    def _optimize_python_objects(self):
        """Pythonオブジェクトの最適化"""
        try:
            # sys.intern()でよく使われる文字列を最適化
            # （ただし、動的に実行するのは危険なので、ここでは基本的な最適化のみ）

            # 未使用の参照をクリア
            gc.collect()

            # デバッグ情報をクリア（開発時のみ）
            if hasattr(sys, "_clear_type_cache"):
                sys._clear_type_cache()

        except Exception as e:
            self.logger.debug(f"Python object optimization failed: {e}")

    def start_monitoring(self):
        """メモリ監視を開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """メモリ監視を停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        self.logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                self.memory_history.append(memory_info)

                # 履歴サイズ制限
                if len(self.memory_history) > self.optimization_config["history_limit"]:
                    self.memory_history = self.memory_history[-50:]

                # 自動最適化チェック
                self._check_auto_optimization(memory_info)

                time.sleep(self.optimization_config["monitoring_interval"])

            except Exception as e:
                self.logger.error(f"Memory monitoring loop error: {e}")
                time.sleep(1.0)

    def _check_auto_optimization(self, memory_info: MemoryInfo):
        """自動最適化をチェック"""
        memory_percent = memory_info.memory_percent

        if memory_percent >= self.optimization_config["critical_threshold"]:
            self.logger.critical(f"Critical memory usage: {memory_percent:.1f}%")
            self.optimize_memory()

        elif memory_percent >= self.optimization_config["warning_threshold"]:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")

        elif memory_percent >= self.optimization_config["gc_threshold"]:
            self.logger.info(
                f"Running garbage collection due to memory usage: {memory_percent:.1f}%"
            )
            gc.collect()

    def get_memory_recommendations(self) -> list[str]:
        """メモリ最適化の推奨事項を取得"""
        if not self.memory_history:
            return ["Start monitoring to get recommendations"]

        latest_info = self.memory_history[-1]
        recommendations = []

        # メモリ使用率に基づく推奨
        if latest_info.memory_percent > 90:
            recommendations.append("メモリ使用率が非常に高いです。不要なプロセスを終了してください")
            recommendations.append("バッチサイズを減らすことを検討してください")

        elif latest_info.memory_percent > 80:
            recommendations.append(
                "メモリ使用率が高いです。定期的にガベージコレクションを実行してください"
            )

        # プロセスメモリに基づく推奨
        if latest_info.process_memory > 4.0:  # 4GB以上
            recommendations.append(
                "プロセスメモリ使用量が大きいです。データの分割処理を検討してください"
            )

        # 履歴に基づく推奨
        if len(self.memory_history) > 10:
            recent_usage = [info.memory_percent for info in self.memory_history[-10:]]
            avg_usage = np.mean(recent_usage)

            if avg_usage > 75:
                recommendations.append(
                    "平均メモリ使用率が高いです。システム設定の見直しを検討してください"
                )

        return recommendations if recommendations else ["メモリ使用量は適切な範囲内です"]

    def export_memory_report(self, output_path: str):
        """メモリレポートをエクスポート"""
        try:
            import json

            report_data = {
                "current_memory_info": {
                    "total_memory_gb": self.memory_history[-1].total_memory
                    if self.memory_history
                    else 0,
                    "available_memory_gb": self.memory_history[-1].available_memory
                    if self.memory_history
                    else 0,
                    "memory_percent": self.memory_history[-1].memory_percent
                    if self.memory_history
                    else 0,
                    "process_memory_gb": self.memory_history[-1].process_memory
                    if self.memory_history
                    else 0,
                },
                "memory_history": [
                    {
                        "timestamp": i,
                        "memory_percent": info.memory_percent,
                        "process_memory_gb": info.process_memory,
                        "available_memory_gb": info.available_memory,
                    }
                    for i, info in enumerate(self.memory_history)
                ],
                "optimization_config": self.optimization_config,
                "recommendations": self.get_memory_recommendations(),
                "export_timestamp": time.time(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Memory report exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export memory report: {e}")

    def estimate_memory_requirement(
        self, data_size_gb: float, processing_factor: float = 2.0
    ) -> dict[str, Any]:
        """メモリ要件を推定"""
        try:
            current_info = self.get_memory_info()

            # 推定メモリ要件
            estimated_requirement = data_size_gb * processing_factor

            # 利用可能メモリとの比較
            available_memory = current_info.available_memory

            result = {
                "data_size_gb": data_size_gb,
                "estimated_requirement_gb": estimated_requirement,
                "available_memory_gb": available_memory,
                "sufficient_memory": estimated_requirement <= available_memory,
                "memory_shortage_gb": max(0, estimated_requirement - available_memory),
                "recommended_batch_size": self._calculate_recommended_batch_size(
                    data_size_gb, available_memory
                ),
            }

            return result

        except Exception as e:
            self.logger.error(f"Failed to estimate memory requirement: {e}")
            return {"error": str(e)}

    def _calculate_recommended_batch_size(
        self, data_size_gb: float, available_memory_gb: float
    ) -> int:
        """推奨バッチサイズを計算"""
        try:
            # 安全マージンを考慮（利用可能メモリの70%を使用）
            usable_memory = available_memory_gb * 0.7

            # 単位データあたりのメモリ使用量を推定
            unit_memory = 0.1  # 100MB per unit (経験値)

            # 推奨バッチサイズ
            recommended_size = int(usable_memory / unit_memory)

            # 最小・最大値の制限
            return max(1, min(recommended_size, 32))

        except Exception:
            return 8  # デフォルト値
