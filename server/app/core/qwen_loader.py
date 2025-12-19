"""
Qwen Model Loader

This module manages the loading and caching of QwenImageEditPlusPipeline models.
It implements a single-pipeline architecture with multiple LoRA adapters and hot-swapping.

Key Features:
- Single pipeline instance shared across all tasks
- Multiple LoRA adapters (insertion, removal, white-balance) loaded without fusing
- Hot-swapping: Pre-load adapters to RAM for fast switching (avoids disk I/O)
- Dynamic adapter switching based on task type
- Aggressive memory cleanup and cache management
- Support for FlowMatch scheduler optimization
- Progress callbacks for model loading

Architecture:
- Base model: Qwen/Qwen-Image-Edit-2509 (loaded to GPU once)
- Adapters: Task-specific LoRA checkpoints (insertion_cp.safetensors, etc.)
  - Pre-loaded to RAM cache when base model loads
  - Hot-swapped from RAM to GPU when switching tasks (fast)
  - Falls back to disk loading if RAM cache unavailable
- Device: Automatically detects CUDA/XPU/MPS/CPU
- Memory: Aggressive cleanup to prevent OOM errors

Hot-Swapping Strategy:
1. Load base model structure to GPU once
2. Pre-load all 3 adapters (.safetensors) into system RAM (CPU) when base model loads
3. When switching tasks: Copy tensors from RAM directly to GPU memory addresses
4. Fallback to disk loading if hot-swap fails or adapter not in RAM cache

Usage:
    from app.core.qwen_loader import load_qwen_pipeline
    
    pipeline = load_qwen_pipeline(
        task_type="insertion",
        enable_flowmatch_scheduler=False,
        on_loading_progress=lambda msg, pct: print(f"{msg}: {pct}%")
    )
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any

import torch

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .config import settings

logger = logging.getLogger(__name__)

# Try to import safetensors for loading LoRA checkpoints (hot-swapping)
try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning(
        "safetensors not available - hot-swapping will fallback to disk loading"
    )

# Suppress expected PEFT warnings when working with multiple LoRA adapters
warnings.filterwarnings(
    "ignore", message="Already found a `peft_config` attribute in the model"
)
warnings.filterwarnings("ignore", message="Already unmerged. Nothing to do.")
warnings.filterwarnings(
    "ignore", message="Adapter cannot be set when the model is merged"
)

# ============================================================================
# Global State (Pipeline and Adapter Tracking)
# ============================================================================

# Single pipeline instance for all tasks
_pipeline: Optional[DiffusionPipeline] = None
# Track which adapters are loaded
_loaded_adapters: set[str] = set()
# Track which adapters are loaded in GPU (for fast switching)
_gpu_loaded_adapters: set[str] = set()
# Track adapter usage times for LRU eviction (if memory-aware unloading enabled)
_adapter_usage_times: Dict[str, float] = {}
# Current active adapter
_current_adapter: Optional[str] = None
# Track if adapters are currently fused (merged into base model)
_is_fused: bool = False
# Track optimization flags used for current pipeline
_pipeline_cache_key: Optional[str] = None
# RAM cache for LoRA adapters (CPU tensors) - Hot-swapping optimization
_adapter_ram_cache: Dict[str, Dict[str, torch.Tensor]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def _ensure_file(path: str | Path, env_name: str) -> Path:
    """
    Ensure file exists, raise clear error if not.
    
    Args:
        path: File path to check
        env_name: Environment variable name (for error message)
    
    Returns:
        Path object if file exists
    
    Raises:
        FileNotFoundError: If file does not exist
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{env_name} points to non-existent file: {model_path}. "
            f"Please check the path in .env ({env_name})."
        )
    return model_path


def _flush_memory() -> None:
    """
    Flush GPU and CPU memory aggressively.
    
    Performs multiple garbage collection passes and CUDA cache clearing
    to free up memory. Used before/after large operations to prevent OOM.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
        torch.cuda.ipc_collect()  # Force IPC cleanup
    # Force multiple GC passes to ensure cleanup
    for _ in range(3):
        gc.collect()
    logger.debug("🧹 Memory flushed (gc.collect + torch.cuda.empty_cache + synchronize)")


def _calculate_tensor_size_mb(tensors: Dict[str, torch.Tensor]) -> float:
    """
    Calculate total size of tensors in MB.
    
    Args:
        tensors: Dictionary of tensors
    
    Returns:
        Total size in MB
    """
    return sum(
        t.numel() * t.element_size() for t in tensors.values()
    ) / (1024 ** 2)


def _resolve_lora_path(task_type: str) -> Path:
    """
    Resolve LoRA checkpoint path based on task type.
    
    Args:
        task_type: "insertion", "removal", or "white-balance"
    
    Returns:
        Path to LoRA checkpoint file
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist
    """
    if task_type == "removal":
        return _ensure_file(settings.model_file_removal, "MODEL_FILE_REMOVAL")
    elif task_type == "white-balance":
        return _ensure_file(
            settings.model_file_white_balance,
            "MODEL_FILE_WHITE_BALANCE",
        )
    else:  # insertion (default)
        return _ensure_file(settings.model_file_insertion, "MODEL_FILE_INSERTION")


def _load_adapter_to_ram(lora_path: Path, adapter_name: str) -> Dict[str, torch.Tensor]:
    """
    Load LoRA checkpoint into RAM (CPU) for hot-swapping.
    
    Args:
        lora_path: Path to .safetensors file
        adapter_name: Name of the adapter
    
    Returns:
        Dictionary mapping tensor names to CPU tensors
    
    Raises:
        RuntimeError: If safetensors is not available or loading fails
        FileNotFoundError: If checkpoint file does not exist
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError(
            "safetensors library not available - cannot load adapter to RAM"
        )
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
    
    logger.info(f"📥 Loading adapter '{adapter_name}' into RAM from {lora_path}")
    
    adapter_tensors: Dict[str, torch.Tensor] = {}
    
    # Load all tensors from safetensors file to CPU memory
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Ensure tensor is on CPU (safetensors may load to different device)
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            adapter_tensors[key] = tensor
    
    # Log total size
    total_size_mb = _calculate_tensor_size_mb(adapter_tensors)
    logger.info(
        f"✅ Loaded adapter '{adapter_name}' into RAM: "
        f"{len(adapter_tensors)} tensors, {total_size_mb:.2f} MB"
    )
    
    return adapter_tensors


def _hot_swap_adapter_from_ram(
    pipeline: DiffusionPipeline,
    adapter_name: str,
    adapter_tensors: Dict[str, torch.Tensor]
) -> None:
    """
    Hot-swap adapter from RAM cache to GPU using temporary file.
    
    Strategy: Save RAM-cached tensors to temp file, then load via diffusers API.
    Much faster than disk I/O since data is already in RAM.
    
    Args:
        pipeline: QwenImageEditPlusPipeline instance
        adapter_name: Adapter name to swap in
        adapter_tensors: CPU tensors from RAM cache
    
    Raises:
        RuntimeError: If hot-swap fails (fallback to disk loading)
    """
    logger.info(f"🔥 Hot-swapping adapter '{adapter_name}' from RAM to GPU...")
    
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # Convert tensors to correct dtype (keep on CPU, diffusers will move to GPU)
    gpu_tensors = {
        key: cpu_tensor.to(dtype=dtype) if cpu_tensor.dtype != dtype else cpu_tensor
        for key, cpu_tensor in adapter_tensors.items()
    }
    
    # Save to temp file and load via diffusers (fast since data is in RAM)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', suffix='.safetensors', delete=False
        ) as f:
            temp_file = f.name
        
        save_file(gpu_tensors, temp_file)
        pipeline.load_lora_weights(temp_file, adapter_name=adapter_name)
        logger.info(f"✅ Hot-swapped adapter '{adapter_name}' from RAM cache")
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def _preload_all_adapters_to_ram() -> None:
    """
    Pre-load all 3 LoRA adapters into RAM cache for fast hot-swapping.
    
    Loads: insertion, removal, white-balance adapters to CPU memory.
    Enables instant switching between tasks without disk I/O.
    """
    global _adapter_ram_cache
    
    # Skip if already pre-loaded
    if len(_adapter_ram_cache) >= 3:
        logger.debug("All adapters already pre-loaded into RAM, skipping")
        return
    
    logger.info("🔄 Pre-loading all adapters into RAM for hot-swapping...")
    
    adapters_to_load = [
        ("insertion", settings.model_file_insertion, "MODEL_FILE_INSERTION"),
        ("removal", settings.model_file_removal, "MODEL_FILE_REMOVAL"),
        ("white-balance", settings.model_file_white_balance, "MODEL_FILE_WHITE_BALANCE"),
    ]
    
    loaded_count = 0
    failed_count = 0
    
    for adapter_name, model_file, env_name in adapters_to_load:
        if adapter_name in _adapter_ram_cache:
            continue  # Already cached
        
        try:
            lora_path = _ensure_file(model_file, env_name)
            _adapter_ram_cache[adapter_name] = _load_adapter_to_ram(
                lora_path, adapter_name
            )
            loaded_count += 1
        except Exception as exc:
            logger.warning(
                f"⚠️ Failed to pre-load adapter '{adapter_name}' into RAM: {exc}. "
                "Will fallback to disk loading."
            )
            failed_count += 1
    
    # Calculate total cache size
    total_size_mb = sum(
        _calculate_tensor_size_mb(tensors)
        for tensors in _adapter_ram_cache.values()
    )
    
    logger.info(
        f"✅ Pre-loading complete: {loaded_count} adapters loaded into RAM "
        f"({failed_count} failed, {total_size_mb:.2f} MB total)"
    )


def _is_adapter_in_gpu(adapter_name: str) -> bool:
    """
    Check if adapter is loaded in GPU pipeline.
    
    Verifies both tracking set and pipeline's peft_config for accuracy.
    
    Args:
        adapter_name: Adapter name to check
    
    Returns:
        True if adapter is loaded in GPU, False otherwise
    """
    global _pipeline, _gpu_loaded_adapters
    
    if _pipeline is None or adapter_name not in _gpu_loaded_adapters:
        return False
    
    # Verify adapter is actually in pipeline's peft_config
    if hasattr(_pipeline, 'peft_config') and _pipeline.peft_config is not None:
        if adapter_name in _pipeline.peft_config:
            return True
        # Adapter was removed from pipeline but still in tracking - sync tracking
        _gpu_loaded_adapters.discard(adapter_name)
        return False
    
    return True


def _get_gpu_memory_free_mb() -> Optional[float]:
    """
    Get free GPU memory in MB.
    
    Returns:
        Free GPU memory in MB, or None if CUDA not available
    """
    if not torch.cuda.is_available():
        return None
    
    try:
        free_memory = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_allocated(0)
        )
        return free_memory / (1024 ** 2)  # MB
    except Exception:
        return None


def _unload_lru_adapter_if_needed(requested_adapter: str) -> None:
    """
    Unload least-recently-used adapter if GPU memory is low.
    
    Only runs if enable_memory_aware_unloading is True.
    Note: diffusers keeps adapters in memory, we only track them as "unloaded".
    Adapter can be quickly reloaded from RAM cache if needed.
    
    Args:
        requested_adapter: Adapter that will be loaded (don't unload this one)
    """
    global _pipeline, _gpu_loaded_adapters, _adapter_usage_times, _current_adapter
    
    if not settings.enable_memory_aware_unloading or _pipeline is None:
        return
    
    free_memory_mb = _get_gpu_memory_free_mb()
    if free_memory_mb is None or free_memory_mb >= settings.gpu_memory_unload_threshold_mb:
        return  # Can't check or memory sufficient
    
    # Find LRU adapter (exclude requested and current)
    candidates = _gpu_loaded_adapters - {requested_adapter, _current_adapter}
    if not candidates:
        return  # No candidates
    
    lru_adapter = min(candidates, key=lambda a: _adapter_usage_times.get(a, 0.0))
    logger.info(
        f"🧹 Unloading LRU adapter '{lru_adapter}' to free GPU memory "
        f"(free: {free_memory_mb:.0f} MB < threshold: "
        f"{settings.gpu_memory_unload_threshold_mb} MB)"
    )
    
    _gpu_loaded_adapters.discard(lru_adapter)
    logger.info(
        f"✅ Unloaded adapter '{lru_adapter}' from GPU "
        "(can be reloaded from RAM cache if needed)"
    )


def _load_base_pipeline(
    base_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    on_loading_progress: Optional[Callable[[str, float]]] = None
) -> DiffusionPipeline:
    """
    Load base QwenImageEditPlusPipeline with OOM retry logic.
    
    Args:
        base_model_id: Model ID or path
        device: Target device
        dtype: Target dtype
        on_loading_progress: Optional progress callback
    
    Returns:
        Loaded pipeline instance
    
    Raises:
        RuntimeError: If loading fails after retry
    """
    # Lazy import to avoid triton import issues at module level
    from diffusers import QwenImageEditPlusPipeline  # type: ignore
    
    _flush_memory()
    
    logger.info(
        f"🔄 Loading QwenImageEditPlusPipeline base model '{base_model_id}' "
        f"on {device} (dtype={dtype})"
    )
    
    if on_loading_progress:
        on_loading_progress("Checking cache...", 0.05)
    
    pipeline_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "cuda" if device.type == "cuda" else None,
        "low_cpu_mem_usage": True,
    }
    
    if on_loading_progress:
        on_loading_progress("Loading checkpoint shards to RAM...", 0.20)
    
    _flush_memory()
    
    # Load pipeline with OOM retry logic
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            base_model_id, **pipeline_kwargs
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
            logger.warning(
                "⚠️ OOM during pipeline load, clearing cache and retrying..."
            )
            clear_qwen_cache()
            time.sleep(3)  # Wait for memory release
            _flush_memory()
            pipeline = QwenImageEditPlusPipeline.from_pretrained(
                base_model_id, **pipeline_kwargs
            )
        else:
            raise
    
    if on_loading_progress:
        on_loading_progress("Moving weights to GPU...", 0.60)
    
    return pipeline


def _configure_flowmatch_scheduler(
    pipeline: DiffusionPipeline,
    on_loading_progress: Optional[Callable[[str, float]]] = None
) -> None:
    """
    Configure FlowMatch scheduler if enabled.
    
    Args:
        pipeline: Pipeline instance to configure
        on_loading_progress: Optional progress callback
    """
    try:
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        logger.info(
            f"🔄 Configuring FlowMatchEulerDiscreteScheduler "
            f"with shift={settings.scheduler_shift}"
        )
        if on_loading_progress:
            on_loading_progress("Configuring scheduler...", 0.70)
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipeline.scheduler.config, shift=settings.scheduler_shift
        )
        logger.info("✅ FlowMatch scheduler configured")
    except ImportError:
        logger.warning(
            "⚠️ FlowMatchEulerDiscreteScheduler not available. "
            "Skipping scheduler configuration."
        )
    except Exception as exc:
        logger.warning(f"⚠️ Failed to configure FlowMatch scheduler: {exc}")


def _move_pipeline_to_cpu(pipeline: DiffusionPipeline) -> None:
    """
    Move pipeline components to CPU to free GPU memory.
    
    Args:
        pipeline: Pipeline instance to move
    """
    try:
        # Reset device_map if pipeline was loaded with it
        if hasattr(pipeline, 'hf_device_map') and pipeline.hf_device_map:
            if hasattr(pipeline, 'reset_device_map'):
                pipeline.reset_device_map()
            elif hasattr(pipeline, 'disable_model_cpu_offload'):
                pipeline.disable_model_cpu_offload()
        
        # Move components to CPU
        if hasattr(pipeline, 'to'):
            pipeline.to('cpu')
        for component_name in ['unet', 'vae', 'text_encoder']:
            component = getattr(pipeline, component_name, None)
            if component is not None and hasattr(component, 'to'):
                component.to('cpu')
    except Exception as e:
        logger.warning(f"⚠️ Error moving pipeline to CPU: {e}")
        try:
            if hasattr(pipeline, 'reset_device_map'):
                pipeline.reset_device_map()
        except Exception:
            pass


def _load_adapter_with_fallback(
    pipeline: DiffusionPipeline,
    adapter_name: str,
    lora_path: Path
) -> None:
    """
    Load adapter with hot-swap fallback to disk loading.
    
    Args:
        pipeline: Pipeline instance
        adapter_name: Adapter name
        lora_path: Path to LoRA checkpoint
    """
    global _loaded_adapters, _gpu_loaded_adapters, _adapter_usage_times, _adapter_ram_cache
    
    logger.info(
        f"📥 Loading LoRA weights for adapter={adapter_name}"
    )
    
    # Try hot-swap from RAM cache first (fastest path)
    adapter_loaded = False
    if adapter_name in _adapter_ram_cache:
        try:
            logger.info(
                f"🔥 Using hot-swap from RAM cache for adapter '{adapter_name}'"
            )
            _hot_swap_adapter_from_ram(
                pipeline, adapter_name, _adapter_ram_cache[adapter_name]
            )
            _loaded_adapters.add(adapter_name)
            _gpu_loaded_adapters.add(adapter_name)
            _adapter_usage_times[adapter_name] = time.time()
            adapter_loaded = True
            logger.info(
                f"✅ Hot-swapped adapter '{adapter_name}' from RAM to GPU, "
                f"total GPU adapters: {len(_gpu_loaded_adapters)}"
            )
        except Exception as hot_swap_exc:
            logger.warning(
                f"⚠️ Hot-swap failed for adapter '{adapter_name}': {hot_swap_exc}. "
                "Falling back to disk loading..."
            )
    
    # Fallback to disk loading if hot-swap failed or adapter not in RAM cache
    if not adapter_loaded:
        logger.info(f"📂 Loading adapter '{adapter_name}' from disk: {lora_path}")
        pipeline.load_lora_weights(str(lora_path), adapter_name=adapter_name)
        _loaded_adapters.add(adapter_name)
        _gpu_loaded_adapters.add(adapter_name)
        _adapter_usage_times[adapter_name] = time.time()
        logger.info(
            f"✅ Loaded adapter '{adapter_name}' from disk to GPU, "
            f"total GPU adapters: {len(_gpu_loaded_adapters)}"
        )
        
        # Cache to RAM for future hot-swapping (if not already cached)
        if adapter_name not in _adapter_ram_cache:
            try:
                logger.debug(
                    f"📥 Caching adapter '{adapter_name}' to RAM "
                    "for future hot-swapping..."
                )
                _adapter_ram_cache[adapter_name] = _load_adapter_to_ram(
                    lora_path, adapter_name
                )
            except Exception:
                pass  # Non-critical: caching failed, will use disk loading next time


# ============================================================================
# Public API
# ============================================================================

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
        on_loading_progress: Optional callback for loading progress
    
    Returns:
        Loaded pipeline instance
    
    Uses sync def because:
    - Runs on Heavy Worker (A100) where model loading is blocking
    - Modal automatically handles threading for sync functions
    - PyTorch operations are CPU/GPU bound, not I/O bound
    """
    global _pipeline, _loaded_adapters, _current_adapter, _is_fused, _pipeline_cache_key
    
    # Resolve flags: request flags override config settings
    use_flowmatch = (
        enable_flowmatch_scheduler
        if enable_flowmatch_scheduler is not None
        else settings.enable_flowmatch_scheduler
    )
    # Base model path: Modal Volume (via MODEL_PATH env) or HuggingFace
    base_model_id = os.getenv("MODEL_PATH") or "Qwen/Qwen-Image-Edit-2509"
    
    # Check if pipeline needs reload due to different optimization flags/model path
    # Cache key should NOT depend on task_type; tasks only swap LoRA adapters.
    cache_key = f"base={base_model_id}|flowmatch={use_flowmatch}"
    if _pipeline is not None and _pipeline_cache_key != cache_key:
        logger.info(
            f"🔄 Optimization flags changed (old: {_pipeline_cache_key}, new: {cache_key}). "
            "Reloading pipeline with new flags..."
        )
        clear_qwen_cache()
        _pipeline_cache_key = None
        time.sleep(1)  # Wait for memory release
    
    # Map task_type to adapter name
    adapter_name = task_type
    
    # Resolve LoRA path
    lora_path = _resolve_lora_path(task_type)
    
    # Get device & dtype
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load base pipeline if not already loaded
    if _pipeline is None:
        model_load_start = time.time()

        # Load base pipeline
        _pipeline = _load_base_pipeline(
            base_model_id, device, dtype, on_loading_progress
        )

        # Configure FlowMatch scheduler if enabled
        if use_flowmatch:
            _configure_flowmatch_scheduler(_pipeline, on_loading_progress)

        if on_loading_progress:
            on_loading_progress("Warming up pipeline...", 0.85)

        _flush_memory()

        # Mark container state and pre-load adapters
        model_load_time_ms = (time.time() - model_load_start) * 1000
        logger.info(
            f"✅ Pipeline loaded successfully in {model_load_time_ms:.2f}ms"
        )

        if on_loading_progress:
            on_loading_progress("Pipeline ready", 1.0)

        try:
            from ..services.container_state import mark_model_loaded
            mark_model_loaded(model_load_time_ms)
        except ImportError:
            # Container state service not available (e.g., in tests)
            pass

        _pipeline_cache_key = cache_key
        logger.info(f"📝 Pipeline cache key: {cache_key}")

        # Pre-load all adapters into RAM for hot-swapping
        _preload_all_adapters_to_ram()
    
    if _pipeline is None:
        raise RuntimeError("Pipeline failed to load")
    
    # Fast path: Adapter already in GPU (instant switch, no loading needed)
    if _is_adapter_in_gpu(adapter_name):
        logger.info(
            f"⚡ Adapter '{adapter_name}' already loaded in GPU "
            "(instant switch), skipping load"
        )
        _adapter_usage_times[adapter_name] = time.time()
        if adapter_name not in _loaded_adapters:
            _loaded_adapters.add(adapter_name)
        if _current_adapter != adapter_name:
            logger.info(
                f"🔄 Switching to adapter '{adapter_name}' (already in GPU)"
            )
            _current_adapter = adapter_name
        return _pipeline
    
    # Check if adapter is in pipeline (container warm/hot but tracking was reset)
    adapter_already_in_pipeline = False
    if hasattr(_pipeline, 'peft_config') and _pipeline.peft_config is not None:
        if adapter_name in _pipeline.peft_config:
            adapter_already_in_pipeline = True
            logger.info(
                f"✅ Adapter '{adapter_name}' already loaded in pipeline "
                "(container warm/hot), skipping load"
            )
            if adapter_name not in _loaded_adapters:
                _loaded_adapters.add(adapter_name)
            if adapter_name not in _gpu_loaded_adapters:
                _gpu_loaded_adapters.add(adapter_name)
    
    # Unload LRU adapter if memory is low (only if loading new adapter)
    if not adapter_already_in_pipeline:
        _unload_lru_adapter_if_needed(adapter_name)
    
    # Load adapter if not already loaded
    if adapter_name not in _loaded_adapters and not adapter_already_in_pipeline:
        _load_adapter_with_fallback(_pipeline, adapter_name, lora_path)
    
    # Switch to requested adapter if not already active
    if _current_adapter != adapter_name:
        logger.info(
            f"🔄 Switching from adapter '{_current_adapter}' to '{adapter_name}' "
            f"for task={task_type}"
        )
        _adapter_usage_times[adapter_name] = time.time()
        
        # Unfuse previous adapter if fused (required for switching)
        if _is_fused and _current_adapter is not None:
            try:
                _pipeline.unfuse_lora()
                _is_fused = False
            except Exception as exc:
                logger.warning(f"Failed to unfuse adapter: {exc}")
        
        _pipeline.set_adapters(adapter_name)
        _current_adapter = adapter_name
        logger.info(f"✅ Switched to adapter '{adapter_name}'")
    
    logger.info(
        f"✅ QwenImageEditPlusPipeline ready with adapter '{adapter_name}' "
        f"for task={task_type}"
    )
    return _pipeline


def clear_qwen_cache() -> None:
    """
    Clear cached Qwen pipeline and adapters aggressively.
    
    Moves pipeline to CPU, clears all tracking sets, and flushes memory.
    Used when pipeline needs to be reloaded (e.g., optimization flags changed).
    """
    global _pipeline, _loaded_adapters, _gpu_loaded_adapters, _adapter_usage_times
    global _current_adapter, _is_fused, _pipeline_cache_key, _adapter_ram_cache
    
    if _pipeline is not None:
        _move_pipeline_to_cpu(_pipeline)
        del _pipeline
        _pipeline = None
    
    # Clear all tracking
    _loaded_adapters.clear()
    _gpu_loaded_adapters.clear()
    _adapter_usage_times.clear()
    _current_adapter = None
    _is_fused = False
    _pipeline_cache_key = None
    
    # Clear RAM cache
    if _adapter_ram_cache:
        cache_size_mb = sum(
            _calculate_tensor_size_mb(tensors)
            for tensors in _adapter_ram_cache.values()
        )
        logger.info(
            f"🧹 Clearing RAM cache: {len(_adapter_ram_cache)} adapters "
            f"({cache_size_mb:.2f} MB)"
        )
        _adapter_ram_cache.clear()
    
    _flush_memory()
    logger.info("🧹 Cleared Qwen pipeline cache (aggressive cleanup)")


def is_qwen_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if Qwen pipeline is loaded without forcing a load.
    
    Args:
        task_type: "insertion", "removal", "white-balance", or None to check
            if pipeline is loaded
    
    Returns:
        True if pipeline is loaded (and adapter is loaded if task_type specified),
        False otherwise
    """
    if _pipeline is None:
        return False
    if task_type is None:
        return True
    # Check if adapter for this task is loaded
    return task_type in _loaded_adapters
