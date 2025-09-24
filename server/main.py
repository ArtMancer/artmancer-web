import os
import base64
import time
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from model_config import MODEL_PRESETS, AVAILABLE_MODELS, DEFAULT_SETTINGS, MODEL_INFO

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global client

    # Startup
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required")

    try:
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        raise

    yield

    # Shutdown
    print("🔄 Shutting down server...")


app = FastAPI(
    title="ArtMancer Web API",
    description="FastAPI server for Gemini-powered image generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response schemas
class ModelSettings(BaseModel):
    """Customizable model settings for Gemini API"""

    model: str = Field(
        default="gemini-2.5-flash-image-preview",
        description="Gemini model to use for generation",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Controls randomness in generation (0.0-2.0)",
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter (0.0-1.0)"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100, description="Top-k sampling parameter (1-100)"
    )
    max_output_tokens: Optional[int] = Field(
        default=None, ge=1, le=8192, description="Maximum number of tokens to generate"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Sequences where generation should stop"
    )


class GenerationRequest(BaseModel):
    """Request model for image generation"""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text prompt for image generation",
    )
    settings: Optional[ModelSettings] = Field(
        default=None, description="Optional model settings to customize generation"
    )
    return_format: str = Field(
        default="base64",
        regex="^(base64|url|both)$",
        description="Response format: 'base64', 'url', or 'both'",
    )


class GenerationResponse(BaseModel):
    """Response model for successful generation"""

    success: bool
    prompt: str
    model_used: str
    settings_used: Dict[str, Any]
    generated_text: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    generation_time: float


class ErrorResponse(BaseModel):
    """Error response model"""

    success: bool = False
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None


# Available models endpoint
@app.get("/api/models")
async def get_available_models():
    """Get list of available Gemini models with capabilities"""
    try:
        models_with_info = []
        for model in AVAILABLE_MODELS:
            model_data = {"name": model, "info": MODEL_INFO.get(model, {})}
            models_with_info.append(model_data)

        return {
            "success": True,
            "models": models_with_info,
            "default": DEFAULT_SETTINGS["model"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "model_fetch_error",
            },
        )


# Model presets endpoint
@app.get("/api/presets")
async def get_model_presets():
    """Get available model presets for different use cases"""
    return {"success": True, "presets": MODEL_PRESETS, "default_preset": "balanced"}


# Apply preset endpoint
@app.post("/api/presets/{preset_name}")
async def apply_preset(preset_name: str):
    """Get settings for a specific preset"""
    if preset_name not in MODEL_PRESETS:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"Preset '{preset_name}' not found",
                "error_type": "preset_not_found",
            },
        )

    return {
        "success": True,
        "preset_name": preset_name,
        "settings": MODEL_PRESETS[preset_name],
    }


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "success": True,
        "status": "healthy",
        "service": "ArtMancer Web API",
        "gemini_client": "connected" if client else "disconnected",
    }


# Generation with preset endpoint
@app.post("/api/generate/preset/{preset_name}")
async def generate_with_preset(preset_name: str, request: GenerationRequest):
    """Generate image using a predefined preset"""
    if preset_name not in MODEL_PRESETS:
        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": f"Preset '{preset_name}' not found",
                "error_type": "preset_not_found",
            },
        )

    # Override settings with preset
    preset_settings = MODEL_PRESETS[preset_name].copy()
    preset_settings.pop("description", None)  # Remove description field

    # Apply preset to request
    request.settings = ModelSettings(**preset_settings)

    # Call the main generation function
    result = await generate_image(request)

    # Add preset info to response
    if hasattr(result, "dict"):
        result_dict = result.dict()
        result_dict["preset_used"] = preset_name
        result_dict["preset_description"] = MODEL_PRESETS[preset_name].get(
            "description", ""
        )
        return result_dict

    return result


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate image using Gemini API with customizable settings"""
    import time

    start_time = time.time()

    if not client:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Gemini client not initialized",
                "error_type": "client_error",
            },
        )

    try:
        # Prepare generation config
        generation_config = {}
        settings_used = {
            "model": request.settings.model
            if request.settings
            else "gemini-2.5-flash-image-preview"
        }

        if request.settings:
            if request.settings.temperature is not None:
                generation_config["temperature"] = request.settings.temperature
                settings_used["temperature"] = request.settings.temperature

            if request.settings.top_p is not None:
                generation_config["top_p"] = request.settings.top_p
                settings_used["top_p"] = request.settings.top_p

            if request.settings.top_k is not None:
                generation_config["top_k"] = request.settings.top_k
                settings_used["top_k"] = request.settings.top_k

            if request.settings.max_output_tokens is not None:
                generation_config["max_output_tokens"] = (
                    request.settings.max_output_tokens
                )
                settings_used["max_output_tokens"] = request.settings.max_output_tokens

            if request.settings.stop_sequences is not None:
                generation_config["stop_sequences"] = request.settings.stop_sequences
                settings_used["stop_sequences"] = request.settings.stop_sequences

        # Make API call to Gemini
        model_name = (
            request.settings.model
            if request.settings
            else "gemini-2.5-flash-image-preview"
        )

        generation_args = {
            "model": model_name,
            "contents": [request.prompt],
        }

        if generation_config:
            generation_args["generation_config"] = types.GenerationConfig(
                **generation_config
            )

        response = client.models.generate_content(**generation_args)

        # Process response
        generated_text = None
        image_base64 = None
        image_url = None

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text = part.text
            elif part.inline_data is not None:
                # Convert image data to base64
                if request.return_format in ["base64", "both"]:
                    image_base64 = base64.b64encode(part.inline_data.data).decode(
                        "utf-8"
                    )

                # Save image and create URL (optional)
                if request.return_format in ["url", "both"]:
                    # For now, we'll include the base64 data URL
                    # In production, you might save to cloud storage and return actual URL
                    image_url = f"data:image/png;base64,{base64.b64encode(part.inline_data.data).decode('utf-8')}"

        generation_time = time.time() - start_time

        return GenerationResponse(
            success=True,
            prompt=request.prompt,
            model_used=model_name,
            settings_used=settings_used,
            generated_text=generated_text,
            image_base64=image_base64,
            image_url=image_url,
            generation_time=generation_time,
        )

    except Exception as e:
        generation_time = time.time() - start_time
        error_details = {
            "generation_time": generation_time,
            "model_attempted": request.settings.model
            if request.settings
            else "gemini-2.5-flash-image-preview",
        }

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "generation_error",
                "details": error_details,
            },
        )


# Batch generation endpoint
@app.post("/api/generate/batch")
async def generate_batch(
    requests: List[GenerationRequest], background_tasks: BackgroundTasks
):
    """Generate multiple images in batch (async processing)"""
    batch_id = f"batch_{int(time.time() * 1000)}"

    # In a production environment, you'd typically use a task queue like Celery
    # For now, we'll process them sequentially
    results = []

    for i, request in enumerate(requests):
        try:
            result = await generate_image(request)
            results.append({"index": i, "success": True, "result": result})
        except HTTPException as e:
            results.append({"index": i, "success": False, "error": e.detail})

    return {
        "success": True,
        "batch_id": batch_id,
        "total_requests": len(requests),
        "results": results,
    }


# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get current API configuration"""
    return {
        "success": True,
        "config": {
            "available_formats": ["base64", "url", "both"],
            "max_prompt_length": 2000,
            "supported_models": AVAILABLE_MODELS,
            "model_info": MODEL_INFO,
            "available_presets": list(MODEL_PRESETS.keys()),
            "default_settings": DEFAULT_SETTINGS,
            "preset_descriptions": {
                name: preset.get("description", "")
                for name, preset in MODEL_PRESETS.items()
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
