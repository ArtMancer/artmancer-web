import os
import base64
import time
import traceback
import logging
import io
import gc
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Try to import QwenImageEditPlusPipeline first
    from diffusers import QwenImageEditPlusPipeline
    PIPELINE_TYPE = "qwen_plus"
    logger.info("✅ Using QwenImageEditPlusPipeline")
except ImportError:
    try:
        # Fallback to original Qwen
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline as QwenImageEditPlusPipeline
        PIPELINE_TYPE = "qwen"
        logger.info("✅ Using QwenImageEditPipeline (fallback)")
    except ImportError:
        try:
            # Fallback to Stable Diffusion Inpainting (more memory efficient)
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline as QwenImageEditPlusPipeline
            PIPELINE_TYPE = "sd_inpaint"
            logger.info("✅ Using StableDiffusionInpaintPipeline (fallback)")
        except ImportError:
            # Last resort - use general AutoPipeline
            from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting as QwenImageEditPlusPipeline
            PIPELINE_TYPE = "auto"
            logger.info("✅ Using AutoPipelineForInpainting (fallback)")

from dotenv import load_dotenv
from model_config import MODEL_PRESETS, AVAILABLE_MODELS, DEFAULT_SETTINGS, MODEL_INFO

# Load environment variables
load_dotenv()

# Global pipeline variable - will be loaded lazily
pipeline = None

class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of desired changes")
    input_image: Optional[str] = Field(None, description="Base64 encoded input image for editing")
    input_images: Optional[List[str]] = Field(None, description="List of base64 encoded images for multi-image input")
    mask_image: Optional[str] = Field(None, description="Base64 encoded mask image")
    width: Optional[int] = Field(512, ge=256, le=2048, description="Output width")
    height: Optional[int] = Field(512, ge=256, le=2048, description="Output height") 
    num_inference_steps: Optional[int] = Field(20, ge=1, le=100, description="Number of inference steps")
    guidance_scale: Optional[float] = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    true_cfg_scale: Optional[float] = Field(6.0, ge=1.0, le=15.0, description="True CFG scale for Qwen")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed for reproducible results")
    model: str = Field("qwen-image-edit", description="Model to use")

class ModelSettings(BaseModel):
    """Model configuration settings"""
    model_name: str
    display_name: str
    description: str
    num_inference_steps: int
    guidance_scale: float
    true_cfg_scale: float
    max_width: int
    max_height: int
    supports_mask: bool
    supports_multi_image: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_type: str
    available_models: List[str]
    gpu_available: bool
    device: str
    memory_info: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app"""
    logger.info("🚀 Starting ArtMancer Server...")
    
    # Startup - pipeline will be loaded lazily when first requested
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down ArtMancer Server...")
    global pipeline
    if pipeline:
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del pipeline
            gc.collect()
            logger.info("✅ Pipeline cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="ArtMancer API",
    description="AI-powered image editing with Qwen models",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_device_info():
    """Get device information"""
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
    }
    
    if torch.cuda.is_available():
        try:
            device_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            device_info["memory_allocated"] = torch.cuda.memory_allocated()
            device_info["memory_reserved"] = torch.cuda.memory_reserved()
        except Exception as e:
            logger.warning(f"Could not get CUDA memory info: {e}")
    
    return device_info

def get_optimal_device():
    """Determine the best device to use"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🎮 Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("🖥️ Using CPU device")
    return device

def load_pipeline_with_fallback():
    """Load pipeline with device and memory fallbacks"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    logger.info("🔄 Loading Qwen Image Edit pipeline...")
    
    device = get_optimal_device()
    
    try:
        # Determine model path based on pipeline type
        if PIPELINE_TYPE == "qwen_plus":
            model_path = "Qwen/Qwen-Image-Edit-2509"
        elif PIPELINE_TYPE == "qwen":
            model_path = "Qwen/Qwen-Image-Edit"
        else:
            # For fallback pipelines, use a stable diffusion model
            model_path = "runwayml/stable-diffusion-inpainting"
        
        # Load with proper options
        load_options = {
            "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
            "device_map": None,  # Don't use automatic device mapping
        }
        
        # Add safety checker disable for Stable Diffusion
        if PIPELINE_TYPE in ["sd_inpaint", "auto"]:
            load_options["safety_checker"] = None
            load_options["requires_safety_checker"] = False
        
        logger.info(f"📦 Loading model: {model_path}")
        logger.info(f"🎛️ Load options: {load_options}")
        
        # Load pipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            **load_options
        )
        
        # Explicitly move to device after loading
        pipeline = pipeline.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        
        if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("✅ Enabled xFormers memory efficient attention")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable xFormers: {e}")
        
        logger.info(f"✅ Pipeline loaded successfully on {device}")
        return pipeline
        
    except Exception as e:
        error_message = str(e).lower()
        
        # Handle specific error types
        if "out of memory" in error_message or "cuda out of memory" in error_message:
            logger.error("❌ GPU out of memory - trying CPU fallback")
            try:
                # Force CPU mode
                pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                pipeline = pipeline.to(torch.device("cpu"))
                logger.info("✅ Pipeline loaded on CPU as fallback")
                return pipeline
            except Exception as cpu_error:
                logger.error(f"❌ CPU fallback also failed: {cpu_error}")
                raise HTTPException(
                    status_code=500, 
                    detail="🔥 Failed to load pipeline on both GPU and CPU. Please restart the server."
                )
        
        elif "device" in error_message or "cuda" in error_message:
            logger.error(f"❌ Device error: {e}")
            raise HTTPException(
                status_code=500, 
                detail="🎮 Device configuration error. Please check CUDA installation or restart the server."
            )
        else:
            logger.error(f"❌ Failed to load pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"❌ Error converting base64 to image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"❌ Error converting image to base64: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting image: {str(e)}")

def get_generation_params(request: GenerationRequest, pipeline_type: str) -> Dict[str, Any]:
    """Get generation parameters based on pipeline type"""
    base_params = {
        "prompt": request.prompt,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
    }
    
    # Add negative prompt if supported
    if request.negative_prompt:
        base_params["negative_prompt"] = request.negative_prompt
    
    # Add Qwen-specific parameters
    if pipeline_type in ["qwen_plus", "qwen"] and hasattr(request, 'true_cfg_scale'):
        base_params["true_cfg_scale"] = request.true_cfg_scale
    
    # Handle generator with proper device placement
    if request.seed is not None:
        device = next(pipeline.parameters()).device
        base_params["generator"] = torch.Generator(device=device).manual_seed(request.seed)
    
    return base_params

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        device_info = get_device_info()
        
        return HealthResponse(
            status="healthy",
            model_loaded=pipeline is not None,
            pipeline_type=PIPELINE_TYPE,
            available_models=list(AVAILABLE_MODELS.keys()),
            gpu_available=device_info["cuda_available"],
            device=device_info["device_name"],
            memory_info=device_info
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            pipeline_type=PIPELINE_TYPE,
            available_models=[],
            gpu_available=False,
            device="unknown",
            memory_info=None
        )

@app.get("/models", response_model=Dict[str, Any])
async def get_models():
    """Get available models and their settings"""
    return {
        "available_models": AVAILABLE_MODELS,
        "model_presets": MODEL_PRESETS,
        "default_settings": DEFAULT_SETTINGS,
        "model_info": MODEL_INFO
    }

@app.get("/models/{model_name}", response_model=ModelSettings)
async def get_model_info(model_name: str):
    """Get specific model information"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_info = MODEL_INFO[model_name]
    preset = MODEL_PRESETS.get(model_name, DEFAULT_SETTINGS)
    
    return ModelSettings(
        model_name=model_name,
        display_name=model_info["display_name"],
        description=model_info["description"],
        num_inference_steps=preset["num_inference_steps"],
        guidance_scale=preset["guidance_scale"],
        true_cfg_scale=preset.get("true_cfg_scale", 6.0),
        max_width=preset["max_width"],
        max_height=preset["max_height"],
        supports_mask=model_info["supports_mask"],
        supports_multi_image=model_info["supports_multi_image"]
    )

def handle_generation_error(e: Exception) -> HTTPException:
    """Handle generation errors with specific error types"""
    error_message = str(e)
    logger.error(f"❌ Generation error: {error_message}")
    
    try:
        if "CUDA out of memory" in error_message:
            error_type = "memory_error"
            error_message = "🔥 Out of GPU memory. Try:\n• Using a smaller image\n• Reducing num_inference_steps\n• Restarting the server"
        elif "different from other tensors" in error_message or "device" in error_message.lower():
            error_type = "device_mismatch_error"
            error_message = "🎮 Device mismatch error. Try:\n• Restarting the server\n• Using CPU-only mode\n• Checking CUDA installation"
        elif "torch" in error_message.lower():
            error_type = "torch_error"
            error_message = f"🔧 PyTorch error: {str(e)}"
        else:
            error_type = "generation_error"
            error_message = f"❌ Generation failed: {str(e)}"
        
        return HTTPException(
            status_code=500,
            detail={
                "error": error_type,
                "message": error_message,
                "full_error": str(e) if logger.level <= logging.DEBUG else None
            }
        )
    except Exception as handle_error:
        logger.error(f"Error in error handler: {handle_error}")
        return HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate or edit an image using Qwen models"""
    start_time = time.time()
    
    try:
        # Load pipeline if not already loaded
        current_pipeline = load_pipeline_with_fallback()
        
        logger.info(f"🎨 Starting generation with prompt: '{request.prompt[:50]}...'")
        
        # Handle input images
        input_image = None
        if request.input_image:
            input_image = base64_to_image(request.input_image)
            logger.info(f"📷 Input image: {input_image.size}")
        elif request.input_images and len(request.input_images) > 0:
            # For multi-image support, use the first image as primary
            input_image = base64_to_image(request.input_images[0])
            logger.info(f"📷 Multi-image input: {input_image.size} (using first of {len(request.input_images)})")
        
        # Handle mask
        mask_image = None
        if request.mask_image:
            mask_image = base64_to_image(request.mask_image)
            logger.info(f"🎭 Mask image: {mask_image.size}")
        
        # Get generation parameters
        generation_params = get_generation_params(request, PIPELINE_TYPE)
        
        # Add images to parameters based on pipeline type
        if PIPELINE_TYPE in ["qwen_plus", "qwen"]:
            if input_image:
                generation_params["image"] = input_image
            if mask_image:
                generation_params["mask_image"] = mask_image
        elif PIPELINE_TYPE in ["sd_inpaint", "auto"]:
            if input_image and mask_image:
                generation_params["image"] = input_image
                generation_params["mask_image"] = mask_image
                generation_params["width"] = request.width
                generation_params["height"] = request.height
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Stable Diffusion inpainting requires both input image and mask"
                )
        
        logger.info(f"🎛️ Generation params: {list(generation_params.keys())}")
        
        # Generate image with proper error handling
        try:
            with torch.no_grad():
                result = current_pipeline(**generation_params)
                
                # Handle different result formats
                if hasattr(result, 'images'):
                    generated_image = result.images[0]
                elif isinstance(result, list):
                    generated_image = result[0]
                else:
                    generated_image = result
        except RuntimeError as e:
            if "different from other tensors" in str(e) or "device" in str(e).lower():
                logger.warning(f"⚠️ Device mismatch detected: {e}")
                # Try to move all tensors to the same device
                try:
                    device = next(current_pipeline.parameters()).device
                    logger.info(f"🔧 Attempting to fix device mismatch on {device}")
                    
                    # Recreate generator on correct device
                    if 'generator' in generation_params:
                        generation_params['generator'] = torch.Generator(device=device).manual_seed(request.seed)
                    
                    # Retry generation
                    with torch.no_grad():
                        result = current_pipeline(**generation_params)
                        
                        if hasattr(result, 'images'):
                            generated_image = result.images[0]
                        elif isinstance(result, list):
                            generated_image = result[0]
                        else:
                            generated_image = result
                            
                except Exception as retry_error:
                    logger.error(f"❌ Device fix retry failed: {retry_error}")
                    raise handle_generation_error(e)
            else:
                raise handle_generation_error(e)
        
        # Convert result to base64
        result_base64 = image_to_base64(generated_image)
        
        # Log success
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s")
        
        return {
            "success": True,
            "image": result_base64,
            "generation_time": round(generation_time, 2),
            "model_used": PIPELINE_TYPE,
            "parameters_used": {
                "prompt": request.prompt,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "true_cfg_scale": generation_params.get("true_cfg_scale"),
                "seed": request.seed,
                "image_size": f"{request.width}x{request.height}" if request.width and request.height else "auto"
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Handle any other errors
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise handle_generation_error(e)

@app.post("/clear-cache")
async def clear_cache():
    """Clear CUDA cache and reload pipeline"""
    try:
        global pipeline
        
        # Clear pipeline
        if pipeline:
            del pipeline
            pipeline = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ Cache cleared successfully")
        
        return {
            "success": True,
            "message": "Cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )