from __future__ import annotations

import gc
import logging
from functools import lru_cache
from typing import Any, Dict

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.getLogger(__name__)


def is_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if pipeline is loaded without forcing a load.
    
    Args:
        task_type: "insertion", "removal", "white-balance", or None to check any pipeline
    
    Returns:
        True if pipeline is loaded, False otherwise
    """
    if task_type == "white-balance":
        from .pix2pix_loader import is_pix2pix_pipeline_loaded
        return is_pix2pix_pipeline_loaded()
    else:
        from .qwen_loader import is_qwen_pipeline_loaded
        return is_qwen_pipeline_loaded(task_type)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - platform specific
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_device_info() -> Dict[str, Any]:
    device = get_device()
    info: Dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()

    return info


def load_pipeline(task_type: str = "insertion") -> DiffusionPipeline:
    """
    Load pipeline for the specified task type.
    Dispatches to appropriate loader based on task type.
    
    Args:
        task_type: "insertion", "removal", or "white-balance"
    
    Returns:
        Loaded DiffusionPipeline
    """
    if task_type == "white-balance":
        from .pix2pix_loader import load_pix2pix_pipeline
        return load_pix2pix_pipeline()
    else:
        # insertion or removal - use Qwen loader
        from .qwen_loader import load_qwen_pipeline
        return load_qwen_pipeline(task_type)


def clear_pipeline_cache() -> None:
    """Clear all cached pipelines."""
    from .qwen_loader import clear_qwen_cache
    from .pix2pix_loader import clear_pix2pix_cache
    
    clear_qwen_cache()
    clear_pix2pix_cache()
    
    gc.collect()
    logger.info("🧹 Cleared all pipeline caches")

