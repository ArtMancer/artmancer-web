from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os

import torch

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .config import settings

logger = logging.getLogger(__name__)

# Check if fastsafetensors is available (optional dependency)
_FASTSAFETENSORS_AVAILABLE = False
try:
    from fastsafetensors import fastsafe_open, SingleGroup  # type: ignore
    _FASTSAFETENSORS_AVAILABLE = True
    logger.info("✅ fastsafetensors is available - will use for faster LoRA loading")
except ImportError:
    logger.debug("fastsafetensors not available - using standard safetensors loader")

# Suppress expected PEFT warnings when working with multiple LoRA adapters
warnings.filterwarnings("ignore", message="Already found a `peft_config` attribute in the model")
warnings.filterwarnings("ignore", message="Already unmerged. Nothing to do.")
warnings.filterwarnings("ignore", message="Adapter cannot be set when the model is merged")

# Single pipeline instance for all tasks
_pipeline: Optional[DiffusionPipeline] = None
# Track which adapters are loaded
_loaded_adapters: set[str] = set()
# Current active adapter
_current_adapter: Optional[str] = None
# Track if adapters are currently fused (merged into base model)
_is_fused: bool = False


def _ensure_file(path: str | Path, env_name: str) -> Path:
    """Ensure file exists, raise clear error if not."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{env_name} points to non-existent file: {model_path}. "
            f"Please check the path in .env ({env_name})."
        )
    return model_path


def _flush_memory() -> None:
    """Flush GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("🧹 Memory flushed (gc.collect + torch.cuda.empty_cache)")


def get_qwen_pipeline(task_type: str | None = None) -> Optional[DiffusionPipeline]:
    """
    Get Qwen pipeline from cache if already loaded.
    
    Args:
        task_type: Optional task type (not used, kept for backward compatibility)
    
    Returns:
        Cached pipeline instance or None if not loaded
    """
    return _pipeline


def load_qwen_pipeline(task_type: str = "insertion") -> DiffusionPipeline:
    """
    Load QwenImageEditPlusPipeline following reference code style:
    - Use single pipeline instance for all tasks
    - Load multiple LoRA adapters (one per task) without fusing
    - Switch adapters when needed
    
    Uses sync def because:
    - Runs on Heavy Worker (A100) where model loading is blocking
    - Modal automatically handles threading for sync functions
    - PyTorch operations are CPU/GPU bound, not I/O bound
    """
    global _pipeline, _loaded_adapters, _current_adapter, _is_fused
    
    # Map task_type to adapter name
    adapter_name = task_type
    
    # Select LoRA file by task (each task has its own checkpoint)
    if task_type == "removal":
        lora_path = _ensure_file(settings.model_file_removal, "MODEL_FILE_REMOVAL")
    elif task_type == "white-balance":
        lora_path = _ensure_file(
            settings.model_file_white_balance,
            "MODEL_FILE_WHITE_BALANCE",
        )
    else:  # insertion (default)
        lora_path = _ensure_file(settings.model_file_insertion, "MODEL_FILE_INSERTION")
    
    # Get device & dtype
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load base pipeline if not already loaded
    if _pipeline is None:
        # Track model load time for container state
        import time
        model_load_start = time.time()
        
        # Lazy import to avoid triton import issues at module level
        from diffusers import QwenImageEditPlusPipeline  # type: ignore
        
        # Apply memory optimizations if enabled
        if settings.enable_memory_optimizations:
            # Set TF32 for faster matmul (only on Ampere+ GPUs)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("✅ Enabled TF32 matmul for faster computation")
        
        # Flush memory before loading
        _flush_memory()
        
        # Allow overriding base model path via environment (for Modal Volume + local SSD cache)
        base_model_id = os.getenv("MODEL_PATH") or "Qwen/Qwen-Image-Edit-2509"
        logger.info(
            "🔄 Loading QwenImageEditPlusPipeline base model '%s' on %s (dtype=%s)",
            base_model_id,
            device,
            dtype,
        )
        
        # Load 4-bit text encoder if enabled (for low-end GPUs)
        text_encoder = None
        if settings.enable_4bit_text_encoder and device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
                
                logger.info("📦 Loading Text Encoder with 4-bit quantization (saves ~4GB VRAM)...")
                _flush_memory()
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                
                # Load text encoder (sync - Modal handles threading)
                text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base_model_id,
                    subfolder="text_encoder",
                    quantization_config=bnb_config,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                logger.info("✅ Text Encoder loaded with 4-bit quantization")
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to load 4-bit text encoder, falling back to default: %s", exc
                )
                text_encoder = None

        try:
            # Build pipeline kwargs
            pipeline_kwargs = {
                "torch_dtype": dtype,
                "device_map": "cuda" if device.type == "cuda" else None,
                "trust_remote_code": True,
            }
            
            # Add text_encoder if 4-bit loaded
            if text_encoder is not None:
                pipeline_kwargs["text_encoder"] = text_encoder
            
            # Add memory optimization flags if enabled
            if settings.enable_memory_optimizations:
                pipeline_kwargs["use_safetensors"] = True
                pipeline_kwargs["low_cpu_mem_usage"] = True
                logger.info("✅ Enabled memory optimizations (safetensors + low_cpu_mem_usage)")
            
            # Load pipeline (sync - Modal handles threading)
            _pipeline = QwenImageEditPlusPipeline.from_pretrained(
                base_model_id,
                **pipeline_kwargs
            )
            
            # Configure FlowMatch scheduler if enabled
            if settings.enable_flowmatch_scheduler:
                try:
                    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                        FlowMatchEulerDiscreteScheduler,
                    )
                    logger.info(
                        f"🔄 Configuring FlowMatchEulerDiscreteScheduler with shift={settings.scheduler_shift}"
                    )
                    _pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                        _pipeline.scheduler.config,
                        shift=settings.scheduler_shift,
                    )
                    logger.info("✅ FlowMatch scheduler configured")
                except ImportError:
                    logger.warning(
                        "⚠️ FlowMatchEulerDiscreteScheduler not available in this diffusers version. "
                        "Skipping scheduler configuration."
                    )
                except Exception as exc:
                    logger.warning("⚠️ Failed to configure FlowMatch scheduler: %s", exc)
            
            # Apply CPU offload if enabled (for low-end GPUs)
            if settings.enable_cpu_offload and device.type == "cuda":
                try:
                    try:
                        from accelerate import cpu_offload  # type: ignore
                        logger.info("📉 Configuring CPU offload for transformer and VAE...")
                        
                        # Offload transformer (heaviest component ~8GB BF16)
                        cpu_offload(_pipeline.transformer, device)
                        
                        # Offload VAE (lighter)
                        cpu_offload(_pipeline.vae, device)
                        
                        # Text encoder 4-bit stays on GPU (already quantized)
                        logger.info("✅ CPU offload configured (transformer + VAE)")
                    except ImportError:
                        # Fallback to diffusers built-in method
                        logger.info("📉 Using diffusers enable_model_cpu_offload (accelerate.cpu_offload not available)...")
                        _pipeline.enable_model_cpu_offload()
                        logger.info("✅ CPU offload configured via diffusers")
                except Exception as exc:
                    logger.warning("⚠️ Failed to configure CPU offload: %s", exc)
            
            # Flush memory after loading
            _flush_memory()
            
            # Calculate model load time and mark container state
            model_load_time_ms = (time.time() - model_load_start) * 1000
            logger.info(f"✅ Pipeline loaded successfully in {model_load_time_ms:.2f}ms")
            
            # Mark container state as warm (model loaded)
            try:
                from ..services.container_state import mark_model_loaded
                mark_model_loaded(model_load_time_ms)
            except ImportError:
                # Container state service not available (e.g., in tests)
                pass
            
        except Exception as exc:
            # Clear cache on error too
            _flush_memory()
            raise RuntimeError(
                f"Failed to load QwenImageEditPlusPipeline from "
                f"'{base_model_id}': {exc}. "
                "Please check diffusers version and Hugging Face access permissions."
            ) from exc
    
    # At this point, _pipeline should be loaded
    if _pipeline is None:
        raise RuntimeError("Pipeline failed to load")
    
    # Load LoRA adapter if not already loaded
    if adapter_name not in _loaded_adapters:
        logger.info("📥 Loading LoRA weights from %s for task=%s (adapter=%s)", lora_path, task_type, adapter_name)
        logger.debug("Currently loaded adapters: %s", list(_loaded_adapters))
        
        # Check if pipeline already has peft_config (may trigger warning on first load)
        has_peft = hasattr(_pipeline, 'peft_config') and _pipeline.peft_config is not None
        if has_peft:
            logger.debug("Pipeline already has peft_config, loading additional adapter '%s'", adapter_name)
        
        try:
            # Try using fastsafetensors for faster loading if available
            if _FASTSAFETENSORS_AVAILABLE and lora_path.suffix == ".safetensors":
                logger.info("🚀 Using fastsafetensors for faster LoRA loading...")
                try:
                    # Load with fastsafetensors (experimental - may need custom integration)
                    # For now, fallback to standard loader as diffusers may not support fastsafetensors directly
                    # TODO: Investigate if diffusers/peft can use fastsafetensors tensors
                    logger.debug("fastsafetensors available but using standard loader (diffusers integration pending)")
                    _pipeline.load_lora_weights(
                        str(lora_path),
                        adapter_name=adapter_name
                    )
                except Exception as fast_exc:
                    logger.warning(f"⚠️ fastsafetensors failed, falling back to standard loader: {fast_exc}")
                    _pipeline.load_lora_weights(
                        str(lora_path),
                        adapter_name=adapter_name
                    )
            else:
                # Standard safetensors loader
                _pipeline.load_lora_weights(
                    str(lora_path),
                    adapter_name=adapter_name
                )
            _loaded_adapters.add(adapter_name)
            logger.info("✅ Loaded adapter '%s', total adapters: %d", adapter_name, len(_loaded_adapters))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LoRA from {lora_path}: {exc}. "
                "Ensure this is a LoRA checkpoint compatible with QwenImageEditPlus."
            ) from exc
    else:
        logger.debug("Adapter '%s' already loaded, skipping", adapter_name)
    
    # Switch to the requested adapter if not already active
    if _current_adapter != adapter_name:
        logger.info("🔄 Switching from adapter '%s' to '%s' for task=%s", _current_adapter, adapter_name, task_type)
        # Unfuse previous adapter if it was fused (to allow switching)
        if _is_fused and _current_adapter is not None:
            try:
                logger.debug("Unfusing adapter '%s' before switch", _current_adapter)
                _pipeline.unfuse_lora()
                _is_fused = False
            except Exception as exc:
                logger.warning("Failed to unfuse adapter: %s", exc)
        # Set the new adapter as active
        _pipeline.set_adapters(adapter_name)
        _current_adapter = adapter_name
        logger.info("✅ Switched to adapter '%s'", adapter_name)
    else:
        logger.debug("Adapter '%s' is already active, skipping switch", adapter_name)
    
    logger.info("✅ QwenImageEditPlusPipeline ready with adapter '%s' for task=%s", adapter_name, task_type)
    
    return _pipeline


def clear_qwen_cache() -> None:
    """Clear cached Qwen pipeline and adapters."""
    global _pipeline, _loaded_adapters, _current_adapter, _is_fused
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
    _loaded_adapters.clear()
    _current_adapter = None
    _is_fused = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Cleared Qwen pipeline cache")


def is_qwen_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if Qwen pipeline is loaded without forcing a load.
    
    Args:
        task_type: "insertion", "removal", "white-balance", or None to check if pipeline is loaded
    
    Returns:
        True if pipeline is loaded (and adapter is loaded if task_type specified), False otherwise
    """
    if _pipeline is None:
        return False
    if task_type is None:
        return True
    # Check if adapter for this task is loaded
    return task_type in _loaded_adapters

