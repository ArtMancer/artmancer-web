from __future__ import annotations

import gc
import logging
import os
from functools import lru_cache
from typing import Any, Dict

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.getLogger(__name__)


def is_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Kiểm tra xem Qwen pipeline đã được load chưa (cho bất kỳ task nào).
    
    Args:
        task_type: "insertion", "removal", "white-balance", hoặc None để kiểm tra bất kỳ.
    
    Returns:
        True nếu pipeline tương ứng đã được load.
    """
    from .qwen_loader import is_qwen_pipeline_loaded
    return is_qwen_pipeline_loaded(task_type)


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _resolve_device(name: str) -> torch.device | None:
    normalized = name.strip().lower()
    if normalized in {"cuda", "gpu", "nvidia"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"xpu", "intel", "arc"} and _xpu_available():
        return torch.device("xpu")
    if normalized == "mps" and torch.backends.mps.is_available():  # pragma: no cover - platform specific
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    return None


def get_device() -> torch.device:
    forced = os.getenv("ARTMANCER_DEVICE", "").strip().lower()
    if forced:
        device = _resolve_device(forced)
        if device is None:
            raise RuntimeError(f"Requested device '{forced}' is not available on this machine.")
        logger.info("⚙️  Forcing execution device to %s via ARTMANCER_DEVICE", device)
        return device

    for candidate in ("cuda", "xpu", "mps"):
        device = _resolve_device(candidate)
        if device is not None:
            return device
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_device_info() -> Dict[str, Any]:
    device = get_device()
    info: Dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),  # pragma: no cover - platform specific
        "xpu_available": _xpu_available(),
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
    elif _xpu_available():
        try:
            info["device_name"] = torch.xpu.get_device_name()
        except Exception:  # pragma: no cover - best effort
            info["device_name"] = "Intel XPU"
    elif torch.backends.mps.is_available():  # pragma: no cover - platform specific
        info["device_name"] = "Apple MPS"

    return info


def load_pipeline(task_type: str = "insertion") -> DiffusionPipeline:
    """
    Load Qwen pipeline cho task tương ứng.
    
    Args:
        task_type: "insertion", "removal", hoặc "white-balance"
    
    Returns:
        DiffusionPipeline đã load
    """
    # Cả 3 task đều dùng chung Qwen loader, khác nhau ở checkpoint và tham số.
    from .qwen_loader import load_qwen_pipeline
    return load_qwen_pipeline(task_type)


def clear_pipeline_cache() -> None:
    """Xoá cache toàn bộ Qwen pipelines."""
    from .qwen_loader import clear_qwen_cache
    
    clear_qwen_cache()
    gc.collect()
    logger.info("🧹 Cleared all Qwen pipeline caches")

