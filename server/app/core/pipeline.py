from __future__ import annotations

import gc
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Callable

# torch is imported lazily where needed to avoid requiring it in services that don't need it (e.g., JobManagerService, ImageUtilsService)

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.getLogger(__name__)


def is_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if Qwen pipeline is loaded.
    
    Args:
        task_type: "insertion", "removal", "white-balance", or None to check if pipeline is loaded.
    
    Returns:
        True if pipeline is loaded (and adapter is loaded if task_type specified), False otherwise.
    """
    from .qwen_loader import is_qwen_pipeline_loaded
    return is_qwen_pipeline_loaded(task_type)


def _xpu_available() -> bool:
    """
    Check if Intel XPU (GPU) is available.
    
    Returns:
        True if XPU is available, False otherwise
    """
    import torch  # Lazy import
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _resolve_device(name: str) -> Any:  # torch.device | None, but using Any to avoid torch import
    """
    Resolve device name string to torch.device.
    
    Args:
        name: Device name string (e.g., "cuda", "xpu", "mps", "cpu")
    
    Returns:
        torch.device if available, None otherwise
    """
    import torch  # Lazy import
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


def get_device() -> Any:  # torch.device, but using Any to avoid torch import
    """
    Get the best available device for model execution.
    
    Priority order:
    1. ARTMANCER_DEVICE environment variable (if set)
    2. CUDA (NVIDIA GPU)
    3. XPU (Intel GPU)
    4. MPS (Apple Silicon)
    5. CPU (fallback)
    
    Returns:
        torch.device instance for the selected device
    
    Raises:
        RuntimeError: If ARTMANCER_DEVICE is set but device is not available
    """
    import torch  # Lazy import
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
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary containing:
        - device: Current device name
        - cuda_available: Whether CUDA is available
        - mps_available: Whether MPS (Apple Silicon) is available
        - xpu_available: Whether XPU (Intel GPU) is available
        - device_name: Name of the current device (if available)
        - memory_total: Total GPU memory in bytes (if CUDA/XPU)
        - memory_allocated: Allocated GPU memory in bytes (if CUDA/XPU)
        - memory_reserved: Reserved GPU memory in bytes (if CUDA/XPU)
    """
    import torch  # Lazy import
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


def load_pipeline(
    task_type: str = "insertion",
    enable_flowmatch_scheduler: Optional[bool] = None,
    on_loading_progress: Optional[Callable[[str, float]]] = None,
) -> DiffusionPipeline:
    """
    Load Qwen pipeline for the specified task with optional optimization flags.
    
    Uses sync def because:
    - Runs on Heavy Worker (A100) where model loading is blocking
    - Modal automatically handles threading for sync functions
    
    Args:
        task_type: "insertion", "removal", or "white-balance"
        enable_flowmatch_scheduler: Override config setting (None = use config)
    
    Returns:
        Loaded DiffusionPipeline
    """
    # All 3 tasks use the same Qwen loader, differing only in checkpoint and parameters.
    from .qwen_loader import load_qwen_pipeline
    return load_qwen_pipeline(
        task_type=task_type,
        enable_flowmatch_scheduler=enable_flowmatch_scheduler,
        on_loading_progress=on_loading_progress,
    )


def clear_pipeline_cache() -> None:
    """Clear cache for Qwen pipeline and all adapters."""
    from .qwen_loader import clear_qwen_cache
    
    clear_qwen_cache()
    gc.collect()
    logger.info("🧹 Cleared Qwen pipeline cache")

