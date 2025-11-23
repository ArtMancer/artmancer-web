from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
from diffusers import DiffusionPipeline

from .config import settings

logger = logging.getLogger(__name__)

_pipeline_white_balance: Optional[DiffusionPipeline] = None


def get_pix2pix_pipeline() -> Optional[DiffusionPipeline]:
    """
    Get cached Pix2Pix pipeline if available.
    
    Returns:
        Cached pipeline or None
    """
    return _pipeline_white_balance


def load_pix2pix_pipeline() -> DiffusionPipeline:
    """
    Load Pix2Pix pipeline for white-balance task.
    
    Returns:
        Loaded DiffusionPipeline from HuggingFace
    """
    global _pipeline_white_balance
    
    # Check cache first
    if _pipeline_white_balance is not None:
        logger.info("♻️ Reusing existing Pix2Pix white-balance pipeline")
        return _pipeline_white_balance
    
    # Load Pix2Pix model for white balance from HuggingFace
    logger.info("🎨 Loading Pix2Pix white balance model from HuggingFace: ArtMancer/Pix2Pix_wb")
    
    # Import get_device from pipeline module to avoid circular import
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # Use device_map for CUDA, "mps" for Apple devices
    device_map = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
    
    logger.info(f"📥 Loading DiffusionPipeline from HuggingFace with device_map={device_map}, dtype={dtype}")
    
    # Suppress warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*safetensors.*")
        warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        warnings.filterwarnings("ignore", message=".*deprecated.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        try:
            # Load from HuggingFace using DiffusionPipeline
            # According to reference code: DiffusionPipeline.from_pretrained("ArtMancer/Pix2Pix_wb", dtype=torch.bfloat16, device_map="cuda")
            # Note: Some pipelines may not accept dtype directly, so we try both methods
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    "ArtMancer/Pix2Pix_wb",
                    dtype=dtype,
                    device_map=device_map,
                )
            except (TypeError, ValueError):
                # Fallback: load without dtype, then set it manually
                pipe = DiffusionPipeline.from_pretrained(
                    "ArtMancer/Pix2Pix_wb",
                    device_map=device_map,
                )
                # Set dtype after loading if needed
                if hasattr(pipe, "to") and device.type == "cuda":
                    pipe = pipe.to(dtype=dtype)
        except Exception as e:
            # Final fallback for older diffusers versions
            logger.warning(f"Failed to load with dtype, trying torch_dtype: {e}")
            pipe = DiffusionPipeline.from_pretrained(
                "ArtMancer/Pix2Pix_wb",
                torch_dtype=dtype,
                device_map=device_map,
            )
    
    logger.info("✅ Pix2Pix white balance model loaded from HuggingFace")
    
    # Apply optimizations
    if settings.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        logger.info("✅ Attention slicing enabled for Pix2Pix model")
    if settings.enable_xformers and hasattr(
        pipe, "enable_xformers_memory_efficient_attention"
    ):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("✅ xFormers attention enabled for Pix2Pix model")
        except Exception as exc:
            if "xformers" not in str(exc).lower():
                logger.warning("⚠️ Could not enable xFormers attention: %s", exc)
    
    # VAE slicing for white-balance (Pix2Pix) pipeline
    if settings.enable_vae_slicing_white_balance and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled for Pix2Pix white-balance pipeline")
        except Exception as exc:
            logger.warning("⚠️ Could not enable VAE slicing for Pix2Pix: %s", exc)
    
    _pipeline_white_balance = pipe
    logger.info("✅ Pix2Pix white balance pipeline loaded on %s", device)
    return pipe


def clear_pix2pix_cache() -> None:
    """Clear cached Pix2Pix pipeline."""
    global _pipeline_white_balance
    if _pipeline_white_balance is not None:
        del _pipeline_white_balance
        _pipeline_white_balance = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Cleared Pix2Pix pipeline cache")


def is_pix2pix_pipeline_loaded() -> bool:
    """
    Check if Pix2Pix pipeline is loaded without forcing a load.
    
    Returns:
        True if pipeline is loaded, False otherwise
    """
    return _pipeline_white_balance is not None

