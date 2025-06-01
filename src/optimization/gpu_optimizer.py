"""
GPU最適化モジュール
GPU使用率とメモリの最適化を提供
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger


@dataclass
class GPUInfo:
    """GPU情報"""

    gpu_id: int
    name: str
    total_memory: float  # GB
    used_memory: float  # GB
    free_memory: float  # GB
    utilization: float  # %
    temperature: float | None = None  # °C
    power_usage: float | None = None  # W


class GPUOptimizer:
    """GPU最適化クラス"""

    def __init__(self):
        """初期化"""
        self.logger = get_logger(__name__)

        # GPU利用可能性をチェック
        self.gpu_available = self._check_gpu_availability()
        self.gpu_framework = self._detect_gpu_framework()

        # 最適化設定
        self.optimization_config = {
            "memory_fraction": 0.8,  # 使用するGPUメモリの割合
            "allow_growth": True,  # メモリ使用量の動的増加を許可
            "clear_cache_threshold": 0.9,  # キャッシュクリアの閾値
            "monitoring_interval": 10.0,  # 監視間隔（秒）
        }

        # 監視状態
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.gpu_history: list[GPUInfo] = []

        if self.gpu_available:
            self.logger.info(f"GPU optimization initialized with {self.gpu_framework}")
        else:
            self.logger.info("GPU not available, GPU optimization disabled")

    def _check_gpu_availability(self) -> bool:
        """GPU利用可能性をチェック"""
        try:
            # PyTorchをチェック
            import torch

            if torch.cuda.is_available():
                return True
        except ImportError:
            pass

        try:
            # TensorFlowをチェック
            import tensorflow as tf

            if len(tf.config.list_physical_devices("GPU")) > 0:
                return True
        except ImportError:
            pass

        try:
            # CuPyをチェック
            import cupy

            if cupy.cuda.is_available():
                return True
        except ImportError:
            pass

        return False

    def _detect_gpu_framework(self) -> str:
        """使用可能なGPUフレームワークを検出"""
        frameworks = []

        try:
            import torch

            if torch.cuda.is_available():
                frameworks.append("pytorch")
        except ImportError:
            pass

        try:
            import tensorflow as tf

            if len(tf.config.list_physical_devices("GPU")) > 0:
                frameworks.append("tensorflow")
        except ImportError:
            pass

        try:
            import cupy

            if cupy.cuda.is_available():
                frameworks.append("cupy")
        except ImportError:
            pass

        return ",".join(frameworks) if frameworks else "none"

    def get_gpu_info(self) -> list[GPUInfo]:
        """GPU情報を取得"""
        if not self.gpu_available:
            return []

        gpu_infos = []

        try:
            # PyTorchを使用してGPU情報を取得
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)

                    # メモリ情報
                    total_memory = props.total_memory / (1024**3)  # GB
                    allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    cached_memory = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    free_memory = total_memory - cached_memory

                    # 使用率（簡易版）
                    utilization = (allocated_memory / total_memory) * 100 if total_memory > 0 else 0

                    gpu_info = GPUInfo(
                        gpu_id=i,
                        name=props.name,
                        total_memory=total_memory,
                        used_memory=allocated_memory,
                        free_memory=free_memory,
                        utilization=utilization,
                    )
                    gpu_infos.append(gpu_info)

        except Exception as e:
            self.logger.debug(f"PyTorch GPU info failed: {e}")

        # PyTorchで取得できない場合はnvidia-mlを試行
        if not gpu_infos:
            try:
                import pynvml

                pynvml.nvmlInit()

                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # デバイス名
                    name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                    # メモリ情報
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory = memory_info.total / (1024**3)  # GB
                    used_memory = memory_info.used / (1024**3)  # GB
                    free_memory = memory_info.free / (1024**3)  # GB

                    # 使用率
                    utilization_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = utilization_info.gpu

                    # 温度
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except:
                        temperature = None

                    # 電力使用量
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    except:
                        power_usage = None

                    gpu_info = GPUInfo(
                        gpu_id=i,
                        name=name,
                        total_memory=total_memory,
                        used_memory=used_memory,
                        free_memory=free_memory,
                        utilization=utilization,
                        temperature=temperature,
                        power_usage=power_usage,
                    )
                    gpu_infos.append(gpu_info)

            except Exception as e:
                self.logger.debug(f"NVML GPU info failed: {e}")

        return gpu_infos

    def optimize_gpu_memory(self) -> dict[str, Any]:
        """GPUメモリを最適化"""
        if not self.gpu_available:
            return {"success": False, "error": "GPU not available"}

        try:
            before_info = self.get_gpu_info()
            optimization_actions = []

            # PyTorchメモリ最適化
            try:
                import torch

                if torch.cuda.is_available():
                    # キャッシュをクリア
                    torch.cuda.empty_cache()
                    optimization_actions.append("pytorch_cache_cleared")

                    # メモリ統計をリセット
                    torch.cuda.reset_peak_memory_stats()
                    optimization_actions.append("pytorch_memory_stats_reset")

                    # 同期
                    torch.cuda.synchronize()
                    optimization_actions.append("pytorch_synchronized")

            except Exception as e:
                self.logger.debug(f"PyTorch optimization failed: {e}")

            # TensorFlowメモリ最適化
            try:
                import tensorflow as tf

                # メモリ成長を有効化
                gpus = tf.config.experimental.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    optimization_actions.append("tensorflow_memory_growth_enabled")

            except Exception as e:
                self.logger.debug(f"TensorFlow optimization failed: {e}")

            # CuPyメモリ最適化
            try:
                import cupy

                if cupy.cuda.is_available():
                    # メモリプールをクリア
                    mempool = cupy.get_default_memory_pool()
                    mempool.free_all_blocks()
                    optimization_actions.append("cupy_memory_pool_cleared")

            except Exception as e:
                self.logger.debug(f"CuPy optimization failed: {e}")

            # 少し待ってから再測定
            time.sleep(1.0)
            after_info = self.get_gpu_info()

            # 最適化結果を計算
            memory_freed = 0.0
            if before_info and after_info:
                for before, after in zip(before_info, after_info, strict=False):
                    memory_freed += before.used_memory - after.used_memory

            result = {
                "success": True,
                "memory_freed_gb": memory_freed,
                "optimization_actions": optimization_actions,
                "before_gpu_info": [
                    {
                        "gpu_id": info.gpu_id,
                        "used_memory_gb": info.used_memory,
                        "utilization": info.utilization,
                    }
                    for info in before_info
                ],
                "after_gpu_info": [
                    {
                        "gpu_id": info.gpu_id,
                        "used_memory_gb": info.used_memory,
                        "utilization": info.utilization,
                    }
                    for info in after_info
                ],
            }

            self.logger.info(f"GPU memory optimization completed. Freed: {memory_freed:.2f}GB")
            return result

        except Exception as e:
            self.logger.error(f"GPU memory optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def configure_gpu_settings(self) -> dict[str, Any]:
        """GPU設定を最適化"""
        if not self.gpu_available:
            return {"success": False, "error": "GPU not available"}

        try:
            configuration_actions = []

            # PyTorch設定
            try:
                import torch

                if torch.cuda.is_available():
                    # cuDNNベンチマークを有効化（固定サイズの入力に対して）
                    torch.backends.cudnn.benchmark = True
                    configuration_actions.append("cudnn_benchmark_enabled")

                    # cuDNN決定論的モードを無効化（パフォーマンス向上）
                    torch.backends.cudnn.deterministic = False
                    configuration_actions.append("cudnn_deterministic_disabled")

            except Exception as e:
                self.logger.debug(f"PyTorch configuration failed: {e}")

            # TensorFlow設定
            try:
                import tensorflow as tf

                # GPU設定
                gpus = tf.config.experimental.list_physical_devices("GPU")
                if gpus:
                    for gpu in gpus:
                        # メモリ成長を有効化
                        tf.config.experimental.set_memory_growth(gpu, True)

                        # メモリ制限を設定（オプション）
                        if self.optimization_config["memory_fraction"] < 1.0:
                            memory_limit = int(
                                8192 * self.optimization_config["memory_fraction"]
                            )  # MB
                            tf.config.experimental.set_memory_limit(gpu, memory_limit)

                    configuration_actions.append("tensorflow_gpu_configured")

            except Exception as e:
                self.logger.debug(f"TensorFlow configuration failed: {e}")

            result = {
                "success": True,
                "configuration_actions": configuration_actions,
                "settings": self.optimization_config,
            }

            self.logger.info("GPU settings configured successfully")
            return result

        except Exception as e:
            self.logger.error(f"GPU configuration failed: {e}")
            return {"success": False, "error": str(e)}

    def start_monitoring(self):
        """GPU監視を開始"""
        if not self.gpu_available or self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("GPU monitoring started")

    def stop_monitoring(self):
        """GPU監視を停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

        self.logger.info("GPU monitoring stopped")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                gpu_infos = self.get_gpu_info()
                self.gpu_history.extend(gpu_infos)

                # 履歴サイズ制限
                if len(self.gpu_history) > 1000:
                    self.gpu_history = self.gpu_history[-500:]

                # 自動最適化チェック
                self._check_auto_optimization(gpu_infos)

                time.sleep(self.optimization_config["monitoring_interval"])

            except Exception as e:
                self.logger.error(f"GPU monitoring loop error: {e}")
                time.sleep(1.0)

    def _check_auto_optimization(self, gpu_infos: list[GPUInfo]):
        """自動最適化をチェック"""
        for gpu_info in gpu_infos:
            # メモリ使用率が高い場合
            memory_usage_ratio = gpu_info.used_memory / gpu_info.total_memory
            if memory_usage_ratio >= self.optimization_config["clear_cache_threshold"]:
                self.logger.warning(
                    f"GPU {gpu_info.gpu_id} memory usage high: {memory_usage_ratio:.1%}"
                )
                self.optimize_gpu_memory()

            # 温度が高い場合（警告のみ）
            if gpu_info.temperature and gpu_info.temperature > 80:
                self.logger.warning(
                    f"GPU {gpu_info.gpu_id} temperature high: {gpu_info.temperature}°C"
                )

    def get_gpu_recommendations(self) -> list[str]:
        """GPU最適化の推奨事項を取得"""
        if not self.gpu_available:
            return ["GPU is not available"]

        recommendations = []

        if not self.gpu_history:
            return ["Start monitoring to get recommendations"]

        # 最新のGPU情報を分析
        recent_gpus = self.gpu_history[-len(self.get_gpu_info()) :]

        for gpu_info in recent_gpus:
            gpu_id = gpu_info.gpu_id

            # メモリ使用率チェック
            memory_usage_ratio = gpu_info.used_memory / gpu_info.total_memory
            if memory_usage_ratio > 0.9:
                recommendations.append(
                    f"GPU {gpu_id}: メモリ使用率が非常に高いです（{memory_usage_ratio:.1%}）"
                )
            elif memory_usage_ratio > 0.8:
                recommendations.append(
                    f"GPU {gpu_id}: メモリ使用率が高いです（{memory_usage_ratio:.1%}）"
                )

            # 使用率チェック
            if gpu_info.utilization > 95:
                recommendations.append(
                    f"GPU {gpu_id}: 使用率が非常に高いです（{gpu_info.utilization}%）"
                )

            # 温度チェック
            if gpu_info.temperature:
                if gpu_info.temperature > 85:
                    recommendations.append(
                        f"GPU {gpu_id}: 温度が高いです（{gpu_info.temperature}°C）"
                    )

        # 一般的な推奨
        if not recommendations:
            recommendations.append("GPU使用状況は良好です")

        return recommendations

    def export_gpu_report(self, output_path: str):
        """GPUレポートをエクスポート"""
        try:
            import json

            current_gpu_info = self.get_gpu_info()

            report_data = {
                "gpu_availability": self.gpu_available,
                "gpu_framework": self.gpu_framework,
                "current_gpu_info": [
                    {
                        "gpu_id": info.gpu_id,
                        "name": info.name,
                        "total_memory_gb": info.total_memory,
                        "used_memory_gb": info.used_memory,
                        "free_memory_gb": info.free_memory,
                        "utilization_percent": info.utilization,
                        "temperature_celsius": info.temperature,
                        "power_usage_watts": info.power_usage,
                    }
                    for info in current_gpu_info
                ],
                "optimization_config": self.optimization_config,
                "recommendations": self.get_gpu_recommendations(),
                "export_timestamp": time.time(),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"GPU report exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export GPU report: {e}")
