from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline

from .config import settings

logger = logging.getLogger(__name__)

_pipeline_insertion: Optional[DiffusionPipeline] = None
_pipeline_removal: Optional[DiffusionPipeline] = None


def _ensure_model_file(path: str | Path) -> Path:
    """Ensure model file exists."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please download the safetensors checkpoint and update MODEL_FILE_INSERTION/MODEL_FILE_REMOVAL in .env."
        )
    return model_path


def _is_qwen_model(model_path: Path) -> bool:
    """
    Detect if model file is a QwenImageEditPlus model based on filename.
    
    Args:
        model_path: Path to model file
    
    Returns:
        True if model appears to be QwenImageEditPlus
    """
    filename_lower = model_path.name.lower()
    # Check for Qwen model indicators in filename
    qwen_indicators = ["qwen", "2509"]
    return any(indicator in filename_lower for indicator in qwen_indicators)


def get_qwen_pipeline(task_type: str) -> Optional[DiffusionPipeline]:
    """
    Get cached Qwen pipeline if available.
    
    Args:
        task_type: "insertion" or "removal"
    
    Returns:
        Cached pipeline or None
    """
    if task_type == "removal":
        return _pipeline_removal
    elif task_type == "insertion":
        return _pipeline_insertion
    return None


def load_qwen_pipeline(task_type: str = "insertion") -> DiffusionPipeline:
    """
    Load Qwen pipeline for the specified task type.
    
    Args:
        task_type: "insertion" or "removal"
    
    Returns:
        Loaded DiffusionPipeline (QwenImageEditPlusCustomPipeline or StableDiffusionInpaintPipeline)
    """
    global _pipeline_insertion, _pipeline_removal
    
    # Check cache first
    cached = get_qwen_pipeline(task_type)
    if cached is not None:
        return cached
    
    # Determine model path
    if task_type == "removal":
        model_path = _ensure_model_file(settings.model_file_removal)
    else:  # insertion (default)
        model_path = _ensure_model_file(settings.model_file_insertion)
    
    logger.info("🔄 Loading Qwen pipeline for %s from %s", task_type, model_path)
    
    # Import get_device from pipeline module to avoid circular import
    from .pipeline import get_device
    
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Check if this is a QwenImageEditPlus model
    is_qwen = _is_qwen_model(model_path)
    
    if is_qwen:
        logger.info("🎯 Detected QwenImageEditPlus model, loading QwenImageEditPlusPipeline...")
        try:
            # Import QwenImageEditPlusPipeline from diffusers (new version)
            from diffusers import QwenImageEditPlusPipeline
            logger.info("✅ Found QwenImageEditPlusPipeline in diffusers")
            
            # Try to get constants if available
            CONDITION_IMAGE_SIZE = 512
            VAE_IMAGE_SIZE = 512
            try:
                from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
                    CONDITION_IMAGE_SIZE,
                    VAE_IMAGE_SIZE,
                )
            except ImportError:
                # Constants might be in the pipeline class
                if hasattr(QwenImageEditPlusPipeline, 'CONDITION_IMAGE_SIZE'):
                    CONDITION_IMAGE_SIZE = QwenImageEditPlusPipeline.CONDITION_IMAGE_SIZE
                if hasattr(QwenImageEditPlusPipeline, 'VAE_IMAGE_SIZE'):
                    VAE_IMAGE_SIZE = QwenImageEditPlusPipeline.VAE_IMAGE_SIZE
            
            # Use QwenImageEditPlusPipeline as the custom pipeline
            QwenImageEditPlusCustomPipeline = QwenImageEditPlusPipeline
            
            # Import required components
            from diffusers import (
                QwenImageTransformer2DModel,
                AutoencoderKLQwenImage,
            )
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            
            logger.info("📥 Loading QwenImageEditPlusPipeline components...")
            
            # Load components individually to avoid downloading unnecessary transformer weights
            # We have custom transformer weights in safetensors file, so we don't need to download from HuggingFace
            logger.info("📥 Loading components individually (to avoid downloading transformer from HuggingFace)...")
            
            # Load scheduler
            scheduler = FlowMatchEulerDiscreteScheduler()
            
            # Load text encoder and tokenizer
            logger.info("📥 Loading text encoder and tokenizer from HuggingFace...")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=dtype,
                device_map=device if device.type == "cuda" else None,
            )
            tokenizer = Qwen2Tokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct"
            )
            
            # Load VAE
            logger.info("📥 Loading VAE from HuggingFace...")
            vae = AutoencoderKLQwenImage.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                subfolder="vae",
                torch_dtype=dtype,
            )
            vae = vae.to(device)
            
            # Load transformer from safetensors (NOT from HuggingFace to save bandwidth)
            logger.info("📥 Loading transformer from local safetensors file...")
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
            
            # Filter transformer weights
            transformer_weights = {}
            for key, value in state_dict.items():
                if "transformer" in key.lower() or "model" in key.lower():
                    # Remove common prefixes
                    clean_key = key.replace("transformer.", "").replace("model.", "")
                    transformer_weights[clean_key] = value
            
            # Load transformer base model structure from HuggingFace (small config files only)
            logger.info("📥 Loading transformer base structure from HuggingFace (config only, no weights)...")
            transformer = QwenImageTransformer2DModel.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                subfolder="transformer",
                torch_dtype=dtype,
            )
            
            # Load custom transformer weights from safetensors
            if transformer_weights:
                logger.info(f"📥 Loading {len(transformer_weights)} custom transformer weights from safetensors...")
                transformer.load_state_dict(transformer_weights, strict=False)
                logger.info("✅ Custom transformer weights loaded from safetensors")
            else:
                logger.warning("⚠️ No transformer weights found in safetensors file, using base transformer")
            
            transformer = transformer.to(device)
            
            # Create pipeline with components
            logger.info("📥 Creating QwenImageEditPlusPipeline with components...")
            try:
                pipe = QwenImageEditPlusCustomPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    transformer=transformer,
                )
                pipe = pipe.to(device)
                logger.info("✅ QwenImageEditPlusPipeline created with components")
            except Exception as e:
                logger.error(f"❌ Failed to create pipeline with components: {e}")
                raise
                # Fallback: load components individually
                # Load scheduler
                scheduler = FlowMatchEulerDiscreteScheduler()
                
                # Load text encoder and tokenizer
                logger.info("📥 Loading text encoder and tokenizer...")
                text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    torch_dtype=dtype,
                    device_map=device if device.type == "cuda" else None,
                )
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct"
                )
                
                # Load VAE
                logger.info("📥 Loading VAE...")
                vae = AutoencoderKLQwenImage.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    subfolder="vae",
                    torch_dtype=dtype,
                )
                vae = vae.to(device)
                
                # Load transformer from safetensors
                logger.info("📥 Loading transformer from safetensors...")
                from safetensors.torch import load_file
                state_dict = load_file(str(model_path))
                
                # Filter transformer weights
                transformer_weights = {}
                for key, value in state_dict.items():
                    if "transformer" in key.lower() or "model" in key.lower():
                        # Remove common prefixes
                        clean_key = key.replace("transformer.", "").replace("model.", "")
                        transformer_weights[clean_key] = value
                
                # Load transformer
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    subfolder="transformer",
                    torch_dtype=dtype,
                )
                
                # Load weights into transformer
                if transformer_weights:
                    logger.info(f"📥 Loading {len(transformer_weights)} transformer weights...")
                    transformer.load_state_dict(transformer_weights, strict=False)
                
                transformer = transformer.to(device)
                
                # Create pipeline with components
                logger.info("📥 Creating QwenImageEditPlusPipeline with components...")
                pipe = QwenImageEditPlusCustomPipeline(
                    scheduler=scheduler,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae,
                    transformer=transformer,
                )
                pipe = pipe.to(device)
                logger.info("✅ QwenImageEditPlusPipeline created with components")
            
        except ImportError as e:
            logger.warning(f"⚠️ Failed to import QwenImageEditPlus components: {e}")
            logger.warning("⚠️ QwenImageEditPlusCustomPipeline requires diffusers >= 0.30.0 with Qwen support")
            logger.warning("⚠️ To use QwenImageEditPlus, please update diffusers:")
            logger.warning("   pip install --upgrade diffusers")
            logger.warning("   or: pip install git+https://github.com/huggingface/diffusers.git")
            logger.info("📥 Falling back to standard StableDiffusionInpaintPipeline loading...")
            logger.info("   Note: This will work but may not have optimal quality for Qwen models")
            is_qwen = False
        except Exception as e:
            logger.error(f"❌ Failed to load QwenImageEditPlus pipeline: {e}")
            logger.info("📥 Falling back to standard pipeline loading...")
            is_qwen = False
    
    if not is_qwen:
        # Fallback to standard StableDiffusionInpaintPipeline loading
        logger.info("🔄 Loading standard pipeline from %s", model_path)
        
        # Load base pipeline from pretrained model first
        from diffusers import StableDiffusionInpaintPipeline
        
        logger.info("📥 Loading base pipeline from pretrained model...")
        # Suppress all warnings during model loading to reduce noise
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*safetensors.*")
            warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
            warnings.filterwarnings("ignore", message=".*torch_dtype.*")
            warnings.filterwarnings("ignore", message=".*deprecated.*")
            warnings.filterwarnings("ignore", category=UserWarning)
            # Try using dtype first (new API), fallback to torch_dtype if needed
            try:
                base_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            except TypeError:
                # Fallback for older diffusers versions
                base_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
        logger.info("✅ Base pipeline loaded")
        
        # Now try to load weights from safetensors file
        try:
            # Load all required components from pretrained first
            from transformers import CLIPTextModel
            from diffusers import UNet2DConditionModel, AutoencoderKL
            
            logger.info("📥 Loading components from pretrained...")
            # Suppress all warnings during component loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*safetensors.*")
                warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                warnings.filterwarnings("ignore", message=".*deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning)
            text_encoder = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                subfolder="text_encoder",
                dtype=dtype,
            )
            text_encoder = text_encoder.to(device)
            
            # Try dtype first, fallback to torch_dtype
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    subfolder="unet",
                    dtype=dtype,
                )
            except TypeError:
                unet = UNet2DConditionModel.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    subfolder="unet",
                    torch_dtype=dtype,
                )
            unet = unet.to(device)
            
            try:
                vae = AutoencoderKL.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    subfolder="vae",
                    dtype=dtype,
                )
            except TypeError:
                vae = AutoencoderKL.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    subfolder="vae",
                    torch_dtype=dtype,
                )
            vae = vae.to(device)
            
            logger.info("✅ Components loaded, loading weights from safetensors...")
            # Try to load from safetensors with all components
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*safetensors.*")
                warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                warnings.filterwarnings("ignore", message=".*deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning)
                # Try dtype first, fallback to torch_dtype
                try:
                    pipe = StableDiffusionInpaintPipeline.from_single_file(
                        str(model_path),
                        text_encoder=text_encoder,
                        unet=unet,
                        vae=vae,
                        dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )
                except TypeError:
                    pipe = StableDiffusionInpaintPipeline.from_single_file(
                        str(model_path),
                        text_encoder=text_encoder,
                        unet=unet,
                        vae=vae,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )
            logger.info("✅ Pipeline loaded with weights from safetensors")
        except Exception as e:
            logger.warning("⚠️ Failed to load with components from safetensors: %s", e)
            logger.info("📥 Trying to load weights manually from safetensors...")
            try:
                from safetensors.torch import load_file
                from diffusers import UNet2DConditionModel
                
                # Load safetensors weights
                state_dict = load_file(str(model_path))
                logger.info(f"✅ Loaded {len(state_dict)} weight tensors from safetensors")
                
                # Try to load weights into UNet (most common component in safetensors)
                try:
                    # Filter for UNet weights (usually prefixed with "model.diffusion_model." or "unet.")
                    unet_weights = {}
                    for key, value in state_dict.items():
                        # Check if this is a UNet weight
                        if "unet" in key.lower() or "diffusion_model" in key.lower():
                            # Remove prefix if present
                            clean_key = key.replace("model.diffusion_model.", "").replace("unet.", "")
                            unet_weights[clean_key] = value
                    
                    if unet_weights:
                        logger.info(f"📥 Loading {len(unet_weights)} UNet weights into pipeline...")
                        # Load into base pipeline's UNet
                        base_pipe.unet.load_state_dict(unet_weights, strict=False)
                        logger.info("✅ UNet weights loaded into base pipeline")
                    else:
                        logger.warning("⚠️ No UNet weights found in safetensors")
                except Exception as unet_error:
                    logger.warning("⚠️ Failed to load UNet weights: %s", unet_error)
                
                pipe = base_pipe
            except ImportError:
                logger.warning("⚠️ safetensors not available, using base pipeline only")
                pipe = base_pipe
            except Exception as e2:
                logger.warning("⚠️ Failed to load weights from safetensors: %s", e2)
                logger.info("📥 Using base pipeline without safetensors weights")
                pipe = base_pipe
        
        pipe = pipe.to(device)
        
        # Apply standard optimizations for non-Qwen pipelines
        if settings.attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            logger.info("✅ Attention slicing enabled")
        if settings.enable_xformers and hasattr(
            pipe, "enable_xformers_memory_efficient_attention"
        ):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers attention enabled")
            except Exception as exc:  # pragma: no cover - optional feature
                # Only log if it's not a known issue (e.g., xformers not installed)
                if "xformers" not in str(exc).lower():
                    logger.warning("⚠️ Could not enable xFormers attention: %s", exc)
                # Otherwise silently continue (xformers is optional)
    
    # VAE slicing for memory optimization (Qwen pipelines only)
    if settings.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
            logger.info("✅ VAE slicing enabled for Qwen pipeline")
        except Exception as exc:
            logger.warning("⚠️ Could not enable VAE slicing: %s", exc)
    
    # If QwenImageEditPlus pipeline was loaded, skip standard optimizations
    if is_qwen:
        logger.info("✅ QwenImageEditPlus pipeline loaded, skipping standard optimizations")
    
    # Store in the appropriate global variable
    if task_type == "removal":
        _pipeline_removal = pipe
    else:
        _pipeline_insertion = pipe
    
    logger.info("✅ Qwen pipeline loaded for %s on %s", task_type, device)
    return pipe


def clear_qwen_cache() -> None:
    """Clear cached Qwen pipelines."""
    global _pipeline_insertion, _pipeline_removal
    if _pipeline_insertion is not None:
        del _pipeline_insertion
        _pipeline_insertion = None
    if _pipeline_removal is not None:
        del _pipeline_removal
        _pipeline_removal = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Cleared Qwen pipeline caches")


def is_qwen_pipeline_loaded(task_type: str | None = None) -> bool:
    """
    Check if Qwen pipeline is loaded without forcing a load.
    
    Args:
        task_type: "insertion", "removal", or None to check any pipeline
    
    Returns:
        True if pipeline is loaded, False otherwise
    """
    if task_type == "removal":
        return _pipeline_removal is not None
    elif task_type == "insertion":
        return _pipeline_insertion is not None
    else:
        return (_pipeline_insertion is not None) or (_pipeline_removal is not None)

