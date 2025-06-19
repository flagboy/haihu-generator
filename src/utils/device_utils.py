"""
デバイス自動検出ユーティリティ

PyTorchでGPU、MPS（Apple Silicon）、CPUを自動検出し、最適なデバイスを選択する
"""

import platform
from typing import Literal

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .logger import get_logger

logger = get_logger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def get_available_device(
    preferred_device: str | None = None, fallback_to_cpu: bool = True, log_device_info: bool = True
) -> torch.device | None:
    """
    利用可能な最適なデバイスを取得

    Args:
        preferred_device: 優先するデバイス ("cuda", "mps", "cpu", "auto")
        fallback_to_cpu: 優先デバイスが利用不可の場合にCPUにフォールバックするか
        log_device_info: デバイス情報をログに出力するか

    Returns:
        torch.device: 利用可能なデバイス
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorchがインストールされていません")
        return None

    # デバイス情報を収集
    device_info = get_device_info()

    # 優先デバイスの処理
    if preferred_device == "auto" or preferred_device is None:
        # 自動選択: CUDA > MPS > CPU の優先順位
        if device_info["cuda"]["available"]:
            device = torch.device("cuda")
        elif device_info["mps"]["available"]:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        # 指定されたデバイスを使用
        if preferred_device == "cuda" and device_info["cuda"]["available"]:
            device = torch.device("cuda")
        elif preferred_device == "mps" and device_info["mps"]["available"]:
            device = torch.device("mps")
        elif preferred_device == "cpu":
            device = torch.device("cpu")
        elif fallback_to_cpu:
            logger.warning(
                f"指定されたデバイス '{preferred_device}' は利用できません。CPUを使用します。"
            )
            device = torch.device("cpu")
        else:
            raise ValueError(f"指定されたデバイス '{preferred_device}' は利用できません")

    if log_device_info:
        logger.info(f"使用デバイス: {device}")
        _log_device_details(device, device_info)

    return device


def get_device_info() -> dict[str, dict[str, any]]:
    """
    システムのデバイス情報を取得

    Returns:
        各デバイスタイプの利用可能性と詳細情報
    """
    if not TORCH_AVAILABLE:
        return {
            "cuda": {"available": False},
            "mps": {"available": False},
            "cpu": {"available": True},
        }

    info = {
        "cuda": {"available": torch.cuda.is_available(), "device_count": 0, "devices": []},
        "mps": {"available": False, "built": False},
        "cpu": {"available": True, "threads": torch.get_num_threads()},
    }

    # CUDA情報
    if info["cuda"]["available"]:
        try:
            info["cuda"]["device_count"] = torch.cuda.device_count()
            info["cuda"]["current_device"] = torch.cuda.current_device()
            for i in range(info["cuda"]["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                info["cuda"]["devices"].append(
                    {
                        "index": i,
                        "name": device_props.name,
                        "total_memory": device_props.total_memory,
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "multi_processor_count": device_props.multi_processor_count,
                    }
                )
        except RuntimeError as e:
            # CUDAの初期化に失敗した場合
            logger.debug(f"CUDA情報の取得に失敗: {e}")
            info["cuda"]["available"] = False
            info["cuda"]["device_count"] = 0
            info["cuda"]["devices"] = []

    # MPS情報（Apple Silicon）
    if hasattr(torch.backends, "mps"):
        info["mps"]["built"] = torch.backends.mps.is_built()
        info["mps"]["available"] = torch.backends.mps.is_available()
        if info["mps"]["available"]:
            info["mps"]["platform"] = platform.platform()
            info["mps"]["processor"] = platform.processor()

    return info


def _log_device_details(device: torch.device, device_info: dict[str, dict[str, any]]) -> None:
    """デバイスの詳細情報をログに出力"""
    if device.type == "cuda":
        cuda_info = device_info["cuda"]
        if cuda_info["available"] and cuda_info["devices"]:
            current_device = cuda_info["devices"][cuda_info["current_device"]]
            logger.info(
                f"CUDA デバイス: {current_device['name']} "
                f"(メモリ: {current_device['total_memory'] / 1024**3:.1f} GB)"
            )
    elif device.type == "mps":
        logger.info(
            f"Apple Silicon MPS を使用 (プラットフォーム: {device_info['mps'].get('platform', 'Unknown')})"
        )
    elif device.type == "cpu":
        logger.info(f"CPU を使用 (スレッド数: {device_info['cpu']['threads']})")


def is_mps_available() -> bool:
    """MPSが利用可能かチェック"""
    if not TORCH_AVAILABLE:
        return False

    if hasattr(torch.backends, "mps"):
        return torch.backends.mps.is_built() and torch.backends.mps.is_available()
    return False


def get_device_memory_info(device: torch.device | None = None) -> dict[str, float] | None:
    """
    デバイスのメモリ情報を取得

    Args:
        device: 対象デバイス（Noneの場合は現在のデバイス）

    Returns:
        メモリ情報（allocated, reserved, free）またはNone
    """
    if not TORCH_AVAILABLE or device is None:
        return None

    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved(device) / 1024**3,  # GB
            "free": (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_allocated(device)
            )
            / 1024**3,  # GB
        }
    elif device.type == "mps":
        # MPSのメモリ情報取得は現在PyTorchでサポートされていない
        logger.debug("MPS デバイスのメモリ情報は現在取得できません")
        return None
    else:
        # CPUメモリ情報
        import psutil

        memory = psutil.virtual_memory()
        return {
            "allocated": (memory.total - memory.available) / 1024**3,  # GB
            "reserved": memory.total / 1024**3,  # GB
            "free": memory.available / 1024**3,  # GB
        }


def optimize_for_device(model: any, device: torch.device) -> any:
    """
    デバイスに応じてモデルを最適化

    Args:
        model: PyTorchモデル
        device: 対象デバイス

    Returns:
        最適化されたモデル
    """
    if not TORCH_AVAILABLE or model is None:
        return model

    # モデルをデバイスに移動
    model = model.to(device)

    # デバイス固有の最適化
    if device.type == "cuda":
        # CUDAの最適化
        if hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True
            logger.debug("CUDNN ベンチマークモードを有効化")
    elif device.type == "mps":
        # MPSの最適化
        # 現在、MPS特有の最適化設定は少ない
        logger.debug("MPS デバイス用に最適化")

    return model


# エクスポートする関数
__all__ = [
    "get_available_device",
    "get_device_info",
    "is_mps_available",
    "get_device_memory_info",
    "optimize_for_device",
    "DeviceType",
]
