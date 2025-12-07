from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any
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
# Track optimization flags used for current pipeline
_pipeline_cache_key: Optional[str] = None


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
    """Flush GPU and CPU memory aggressively."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
        torch.cuda.ipc_collect()  # Force IPC cleanup
    # Force multiple GC passes to ensure cleanup
    for _ in range(3):
        gc.collect()
    logger.debug("🧹 Memory flushed (gc.collect + torch.cuda.empty_cache + synchronize)")


def get_qwen_pipeline(task_type: str | None = None) -> Optional[DiffusionPipeline]:
    """
    Get Qwen pipeline from cache if already loaded.
    
    Args:
        task_type: Optional task type (not used, kept for backward compatibility)
    
    Returns:
        Cached pipeline instance or None if not loaded
    """
    return _pipeline


def load_qwen_pipeline(
    task_type: str = "insertion",
    enable_flowmatch_scheduler: Optional[bool] = None,
    on_loading_progress: Optional[Callable[[str, float]]] = None,
) -> DiffusionPipeline:
    """
    Load QwenImageEditPlusPipeline following reference code style:
    - Use single pipeline instance for all tasks
    - Load multiple LoRA adapters (one per task) without fusing
    - Switch adapters when needed
    - Support optimization flags from request (override config if provided)
    
    Args:
        task_type: "insertion", "removal", or "white-balance"
        enable_flowmatch_scheduler: Override config setting (None = use config)
    
    Uses sync def because:
    - Runs on Heavy Worker (A100) where model loading is blocking
    - Modal automatically handles threading for sync functions
    - PyTorch operations are CPU/GPU bound, not I/O bound
    """
    global _pipeline, _loaded_adapters, _current_adapter, _is_fused, _pipeline_cache_key
    
    # Resolve flags: request flags > config settings
    use_flowmatch = enable_flowmatch_scheduler if enable_flowmatch_scheduler is not None else settings.enable_flowmatch_scheduler
    
    # Create cache key based on flags combination
    cache_key = f"{task_type}_{use_flowmatch}"
    
    # Check if pipeline needs to be reloaded due to different flags
    if _pipeline is not None and _pipeline_cache_key != cache_key:
        logger.info(
            f"🔄 Optimization flags changed (old: {_pipeline_cache_key}, new: {cache_key}). "
            "Reloading pipeline with new flags..."
        )
        clear_qwen_cache()
        _pipeline_cache_key = None
        # Wait a bit for memory to be fully released
        import time
        time.sleep(1)
    
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
        
        # Stage 1: Check cache and prepare (0-10%)
        if on_loading_progress:
            on_loading_progress("Checking cache...", 0.05)

        try:
            # Build pipeline kwargs
            # Note: QwenImageEditPlusPipeline accepts torch_dtype, not dtype
            # Using "cuda" to load directly to GPU (more efficient than loading to RAM first)
            pipeline_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,  # Use torch_dtype instead of dtype
                "device_map": "cuda" if device.type == "cuda" else None,
                # Note: trust_remote_code is not expected by QwenImageEditPlusPipeline
                "low_cpu_mem_usage": True,
            }
            
            # Stage 2: Loading weights to RAM (10-40%)
            if on_loading_progress:
                on_loading_progress("Loading checkpoint shards to RAM...", 0.20)
            
            # Clear GPU memory before loading to avoid OOM
            _flush_memory()
            
            # Load pipeline (sync - Modal handles threading)
            # This is the heavy I/O operation - loading checkpoint shards
            # Use try-except to handle OOM and retry after clearing cache
            try:
                _pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    base_model_id,
                    **pipeline_kwargs
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                    logger.warning("⚠️ OOM during pipeline load, clearing cache and retrying...")
                    # Clear any existing pipeline that might be in memory
                    clear_qwen_cache()
                    import time
                    time.sleep(3)  # Wait for memory to be fully released
                    # Retry once
                    _flush_memory()
                    _pipeline = QwenImageEditPlusPipeline.from_pretrained(
                        base_model_id,
                        **pipeline_kwargs
                    )
                else:
                    raise
            
            # Stage 3: Checkpoint loaded, moving to GPU (40-80%)
            if on_loading_progress:
                on_loading_progress("Moving weights to GPU...", 0.60)
            
            # Configure FlowMatch scheduler if enabled
            if use_flowmatch:
                try:
                    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                        FlowMatchEulerDiscreteScheduler,
                    )
                    logger.info(
                        f"🔄 Configuring FlowMatchEulerDiscreteScheduler with shift={settings.scheduler_shift}"
                    )
                    if on_loading_progress:
                        on_loading_progress("Configuring scheduler...", 0.70)
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
            
            # Stage 4: Warming up (80-100%)
            if on_loading_progress:
                on_loading_progress("Warming up pipeline...", 0.85)
            
            # Flush memory after loading
            _flush_memory()
            
            # Calculate model load time and mark container state
            model_load_time_ms = (time.time() - model_load_start) * 1000
            logger.info(f"✅ Pipeline loaded successfully in {model_load_time_ms:.2f}ms")
            
            # Stage 5: Complete
            if on_loading_progress:
                on_loading_progress("Pipeline ready", 1.0)
            
            # Mark container state as warm (model loaded)
            try:
                from ..services.container_state import mark_model_loaded
                mark_model_loaded(model_load_time_ms)
            except ImportError:
                # Container state service not available (e.g., in tests)
                pass
            
            # Store cache key for this pipeline configuration
            _pipeline_cache_key = cache_key
            logger.info(f"📝 Pipeline cache key: {cache_key}")
            
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
    """Clear cached Qwen pipeline and adapters aggressively."""
    global _pipeline, _loaded_adapters, _current_adapter, _is_fused, _pipeline_cache_key
    
    if _pipeline is not None:
        # Move pipeline to CPU first to free GPU memory
        try:
            # Check if pipeline has device_map and reset it first
            if hasattr(_pipeline, 'hf_device_map') and _pipeline.hf_device_map:
                # Pipeline was loaded with device_map, need to reset it first
                if hasattr(_pipeline, 'reset_device_map'):
                    _pipeline.reset_device_map()
                elif hasattr(_pipeline, 'disable_model_cpu_offload'):
                    # Alternative method if reset_device_map doesn't exist
                    _pipeline.disable_model_cpu_offload()
            
            # Now we can safely move to CPU
            if hasattr(_pipeline, 'to'):
                _pipeline.to('cpu')
            if hasattr(_pipeline, 'unet') and _pipeline.unet is not None:
                _pipeline.unet.to('cpu')
            if hasattr(_pipeline, 'vae') and _pipeline.vae is not None:
                _pipeline.vae.to('cpu')
            if hasattr(_pipeline, 'text_encoder') and _pipeline.text_encoder is not None:
                _pipeline.text_encoder.to('cpu')
        except Exception as e:
            logger.warning(f"⚠️ Error moving pipeline to CPU: {e}")
            # If moving to CPU fails, try to reset device_map and delete directly
            try:
                if hasattr(_pipeline, 'reset_device_map'):
                    _pipeline.reset_device_map()
            except Exception:
                pass
        
        # Delete pipeline
        del _pipeline
        _pipeline = None
    
    _loaded_adapters.clear()
    _current_adapter = None
    _is_fused = False
    _pipeline_cache_key = None
    
    # Aggressive memory cleanup
    _flush_memory()
    
    logger.info("🧹 Cleared Qwen pipeline cache (aggressive cleanup)")


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

