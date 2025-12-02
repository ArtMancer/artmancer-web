from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import QwenImageEditPlusPipeline # type: ignore

from .config import settings

logger = logging.getLogger(__name__)

_pipeline_insertion: Optional[DiffusionPipeline] = None
_pipeline_removal: Optional[DiffusionPipeline] = None
_pipeline_wb: Optional[DiffusionPipeline] = None  # white-balance


def _ensure_file(path: str | Path, env_name: str) -> Path:
    """Đảm bảo file tồn tại, nếu không thì báo lỗi rõ ràng."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{env_name} trỏ tới file không tồn tại: {model_path}. "
            f"Hãy kiểm tra đường dẫn trong .env ({env_name})."
        )
    return model_path


def get_qwen_pipeline(task_type: str) -> Optional[DiffusionPipeline]:
    """Lấy pipeline Qwen từ cache nếu đã load."""
    if task_type == "removal":
        return _pipeline_removal
    if task_type == "white-balance":
        return _pipeline_wb
    # Mặc định dùng pipeline insertion cho các task còn lại (insertion)
    return _pipeline_insertion


def load_qwen_pipeline(task_type: str = "insertion") -> DiffusionPipeline:
    """
    Load QwenImageEditPlusPipeline theo đúng style code tham chiếu:
    - Dùng from_pretrained(model_name, torch_dtype=bfloat16)
    - Load LoRA từ MODEL_FILE_INSERTION / MODEL_FILE_REMOVAL / MODEL_FILE_WHITE_BALANCE
    - Không fallback sang pipeline khác; lỗi sẽ raise thẳng để dễ debug.
    """
    global _pipeline_insertion, _pipeline_removal, _pipeline_wb
    
    # Check cache
    cached = get_qwen_pipeline(task_type)
    if cached is not None:
        return cached
    
    # Base model từ Hugging Face (có thể sau này cho vào config nếu cần)
    base_model_id = "Qwen/Qwen-Image-Edit-2509"

    # Chọn file LoRA theo task (mỗi task một checkpoint riêng)
    if task_type == "removal":
        lora_path = _ensure_file(settings.model_file_removal, "MODEL_FILE_REMOVAL")
    elif task_type == "white-balance":
        lora_path = _ensure_file(
            settings.model_file_white_balance,
            "MODEL_FILE_WHITE_BALANCE",
        )
    else:  # insertion (mặc định)
        lora_path = _ensure_file(settings.model_file_insertion, "MODEL_FILE_INSERTION")
    
    # Lấy device & dtype
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(
        "🔄 Loading QwenImageEditPlusPipeline base model '%s' on %s (dtype=%s)",
        base_model_id,
        device,
        dtype,
    )

    try:
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            "Không thể load QwenImageEditPlusPipeline từ "
            f"'{base_model_id}': {exc}. "
            "Hãy kiểm tra lại diffusers version và quyền truy cập Hugging Face."
        ) from exc

    pipe = pipe.to(device)

    # Load LoRA weights
    logger.info("📥 Loading LoRA weights from %s cho task=%s", lora_path, task_type)
    try:
        pipe.load_lora_weights(str(lora_path), adapter_name="lora")
        pipe.set_adapters("lora")
        pipe.fuse_lora(adapter_names=["lora"])
    except Exception as exc:
        raise RuntimeError(
            f"Không thể load/fuse LoRA từ {lora_path}: {exc}. "
            "Đảm bảo đây là checkpoint LoRA tương thích với QwenImageEditPlus."
        ) from exc
    
    logger.info("✅ QwenImageEditPlusPipeline + LoRA đã sẵn sàng cho task=%s", task_type)
    
    # Cache theo task
    if task_type == "removal":
        _pipeline_removal = pipe
    elif task_type == "white-balance":
        _pipeline_wb = pipe
    else:
        _pipeline_insertion = pipe
    
    return pipe


def clear_qwen_cache() -> None:
    """Clear cached Qwen pipelines."""
    global _pipeline_insertion, _pipeline_removal, _pipeline_wb
    if _pipeline_insertion is not None:
        del _pipeline_insertion
        _pipeline_insertion = None
    if _pipeline_removal is not None:
        del _pipeline_removal
        _pipeline_removal = None
    if _pipeline_wb is not None:
        del _pipeline_wb
        _pipeline_wb = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Cleared Qwen pipeline caches")


def is_qwen_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if Qwen pipeline is loaded without forcing a load.
    
    Args:
        task_type: "insertion", "removal", "white-balance", or None to check any pipeline
    
    Returns:
        True if pipeline is loaded, False otherwise
    """
    if task_type == "removal":
        return _pipeline_removal is not None
    if task_type == "white-balance":
        return _pipeline_wb is not None
    if task_type == "insertion":
        return _pipeline_insertion is not None
    # task_type is None: check bất kỳ pipeline nào
    return (
        (_pipeline_insertion is not None)
        or (_pipeline_removal is not None)
        or (_pipeline_wb is not None)
    )

