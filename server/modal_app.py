"""
Modal app for ArtMancer backend.

This file defines all Modal services, images, and helper functions for deployment.

Architecture:
- API Gateway (CPU): Single entry point, routes requests to backend services.
- Job Manager (CPU): Coordinates async generation jobs, dispatches to H100 workers.
- Segmentation Service (T4): Smart mask segmentation.
- Image Utils Service (CPU): Image processing utilities.
- H100 Worker Function: Async image generation (spawned by Job Manager).

Features:
- Modal Volume for checkpoints + base model snapshot.
- Load model trực tiếp từ Volume (không copy) để giảm cold-start từ 20-60s xuống 1-3s.
- Cold boot enabled for cost optimization (scales to zero when idle).

Table of Contents:
1. CONFIGURATION - Constants and paths
2. INFRASTRUCTURE - Modal app, volume, dictio setup
3. IMAGE DEFINITIONS - Docker images for different services
4. HELPER FUNCTIONS - Utilities for job state, TORCH_HOME, pipeline reuse
5. FASTAPI APP FACTORY - Factory function for FastAPI apps
6. SERVICE CLASSES - Modal service definitions (4 endpoints)
7. WORKER FUNCTIONS - Background tasks
8. VOLUME MANAGEMENT - Helper functions for Volume operations
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

# ==========================================
# 1. CONFIGURATION
# ==========================================
APP_NAME = "artmancer"
VOLUME_NAME = "artmancer-checkpoints"
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

# Volume mount + layout
VOL_MOUNT_PATH = "/checkpoints"
VOL_MODEL_PATH = f"{VOL_MOUNT_PATH}/base_model"
VOL_TASKS_PATH = VOL_MOUNT_PATH
VOL_SEGMENTATION_PATH = f"{VOL_MOUNT_PATH}/segmentation"
VOL_BIREFNET_PATH = f"{VOL_MOUNT_PATH}/birefnet"
VOL_REFINEMENT_PATH = f"{VOL_MOUNT_PATH}/stable-diffusion-inpainting"
VOL_TORCH_CACHE_PATH = f"{VOL_MOUNT_PATH}/torch_cache"

# Common ignore patterns for local dirs
IGNORE_PATTERNS = ["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"]

# ==========================================
# 2. INFRASTRUCTURE
# ==========================================
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
job_state_dictio = modal.Dict.from_name("artmancer-job-state", create_if_missing=True)

# ==========================================
# 3. IMAGE DEFINITIONS
# ==========================================
# Docker images for different services. Each image includes only necessary dependencies
# to minimize image size and cold-start time.

def _base_image_setup(python_version: str = "3.12") -> modal.Image:
    """Base CPU image setup: pip upgrade, uv install, common env."""
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install(
            "zlib1g-dev",  # Pillow build deps
            "libjpeg-dev",  # Pillow build deps
        )
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
        .env({"PYTHONPATH": "/root"})
    )

def _add_common_dirs(image: modal.Image) -> modal.Image:
    """Add common local directories to image."""
    return (
        image.add_local_dir("app", "/root/app", ignore=IGNORE_PATTERNS)
        .add_local_dir("api_gateway", "/root/api_gateway", ignore=IGNORE_PATTERNS)
        .add_local_dir("shared", "/root/shared", ignore=IGNORE_PATTERNS)
    )

# 3.1 CPU IMAGE (Job Manager & API Gateway) - lightweight, no torch/IOPaint
cpu_image = (
    _base_image_setup()
    .run_commands(
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "python-dotenv "
        "httpx>=0.27.0 "
        "pillow>=10.3.0 "
        "numpy>=1.26.4 "
        "opencv-python-headless>=4.8.0 "
        "scikit-image>=0.25.2"
    )
    .env({"PYTHONPATH": "/root"})
)
cpu_image = _add_common_dirs(cpu_image)

# 3.2 HEAVY IMAGE (H100) - prebuilt CUDA/Torch, hosts LaMa model
heavy_image = (
    modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --no-cache-dir "
        "transformers==4.57.3 "
        "accelerate==1.12.0 "
        "safetensors>=0.4.3 "
        "fastsafetensors>=0.1.0 "
        "peft>=0.10.0 "
        "diffusers>=0.36.0 "
        "opencv-python-headless "
        "scikit-image "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "python-multipart "
        "python-dotenv "
        "huggingface-hub>=1.0.0 "
        "hf-transfer "
        "requests "
        "tqdm "
        "psutil "
        "pillow>=10.1.0 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "numpy>=1.26.4 "
        "boto3 "
        "awscli"
    )
    .env(
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
            "PYTHONPATH": "/root",
        }
    )
)
heavy_image = _add_common_dirs(heavy_image)

# 3.3 SEGMENTATION IMAGE (T4) - prebuilt CUDA/Torch
segmentation_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --no-cache-dir "
        "ultralytics==8.3.235 "
        "transformers==4.57.3 "
        "kornia "
        "opencv-python-headless "
        "scikit-image "
        "huggingface-hub==0.35.1 "
        "psutil "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "pillow>=10.3.0 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "numpy>=1.26.4 "
        "einops "
        "timm"
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics", "PYTHONPATH": "/root"})
)
segmentation_image = _add_common_dirs(segmentation_image)

# 3.4 IMAGE UTILS IMAGE (CPU-only) - lightweight, no torch/IOPaint
imageutils_image = (
    _base_image_setup()
    .run_commands(
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "pillow>=10.3.0 "
        "numpy>=1.26.4 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "opencv-python-headless>=4.8.0 "
        "scikit-image>=0.25.2"
    )
    .env({"PYTHONPATH": "/root"})
)
imageutils_image = _add_common_dirs(imageutils_image)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def update_job_state(
    task_id: str,
    status: str,
    progress: float,
    error: str | None = None,
    result: str | None = None,
    debug_info: dict | None = None,
    **kwargs,
) -> None:
    """
    Update job state in job_state_dictio, preserving created_at timestamp.
    
    Args:
        task_id: Task identifier
        status: Job status (e.g., "queued", "processing", "done", "error")
        progress: Progress value (0.0-1.0)
        error: Error message if any
        result: Result data if any
        debug_info: Debug information dict
        **kwargs: Additional fields to include in state
    """
    existing_state = job_state_dictio.get(task_id, {})
    job_state_dictio[task_id] = {
        "status": status,
        "progress": progress,
        "error": error,
        "result": result,
        "debug_info": debug_info,
        "created_at": existing_state.get("created_at", time.time()),
        **kwargs,
    }

def check_pipeline_reuse(task_id: str, task_type: str) -> bool:
    """
    Check if pipeline can be reused (container warm/hot optimization).
    
    Decision tree:
    1. If base pipeline not loaded → return False (cold container)
    2. If adapter for task_type not loaded → return True (fast adapter switch)
    3. If pipeline exists but corrupted → return False (need reload)
    4. If pipeline valid → return True (reuse)
    
    Args:
        task_id: Task identifier for logging
        task_type: Task type (insertion, removal, wb)
    
    Returns:
        True if pipeline can be reused, False if reload needed
    """
    try:
        from app.core.qwen_loader import (
            get_qwen_pipeline,
            is_qwen_pipeline_loaded,
            _flush_memory,
            _pipeline as global_pipeline,
        )
        
        # Cold container: no base pipeline loaded
        if global_pipeline is None:
            print(f"🧹 [H100 Worker] Task {task_id}: No pipeline loaded (container cold), will load from scratch...")
            from app.core.qwen_loader import clear_qwen_cache
            clear_qwen_cache()
            time.sleep(1)
            _flush_memory()
            print(f"✅ [H100 Worker] Task {task_id}: Cache cleared and memory flushed, ready for fresh load")
            return False
        
        # Base pipeline loaded, check adapter
        if not is_qwen_pipeline_loaded(task_type):
            print(f"🔄 [H100 Worker] Task {task_id}: Base pipeline loaded but adapter for {task_type} not loaded, will switch adapter...")
            _flush_memory()
            return True  # Adapter switch is fast, no reload needed
        
        # Adapter loaded, verify pipeline is usable
        existing_pipeline = get_qwen_pipeline(task_type)
        if existing_pipeline is None:
            print(f"🔄 [H100 Worker] Task {task_id}: Pipeline state inconsistent, will reload...")
            return False
        
        # Sanity check: verify pipeline has required attributes
        if hasattr(existing_pipeline, "unet") and existing_pipeline.unet is not None:
            print(f"♻️ [H100 Worker] Task {task_id}: Pipeline already loaded for task_type={task_type}, REUSING (container warm/hot)")
            _flush_memory()
            return True
        
        # Pipeline exists but appears corrupted
        print(f"⚠️ [H100 Worker] Task {task_id}: Pipeline exists but appears corrupted, will reload...")
        return False
        
    except Exception as e:
        print(f"⚠️ [H100 Worker] Task {task_id}: Failed to check/reuse pipeline: {e}")
        # Fallback: clear cache to ensure clean state
        try:
            from app.core.qwen_loader import clear_qwen_cache, _flush_memory
            clear_qwen_cache()
            time.sleep(1)
            _flush_memory()
        except Exception:
            pass
        return False

def _get_modal_module():
    """Get modal_app module, handling import edge cases."""
    import sys
    if "modal_app" not in sys.modules:
        import modal_app
        return modal_app
    return sys.modules["modal_app"]

# ==========================================
# 5. FASTAPI APP FACTORY
# ==========================================

def create_fastapi_app(
    job_manager_mode: bool = False,
    include_segmentation: bool = False,
    include_image_utils: bool = False,
) -> FastAPI:
    """
    Create FastAPI app instance with appropriate routers.
    
    This function is used by all Modal services to create their FastAPI apps.
    Each service includes only the routers it needs.
    
    Args:
        job_manager_mode: If True, includes generation_async router (Job Manager).
        include_segmentation: If True, includes smart_mask router (Segmentation Service).
        include_image_utils: If True, includes image_utils router (Image Utils Service).
    
    Returns:
        FastAPI app instance with configured routers and CORS middleware.
    """
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Parse allowed origins from environment
    allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://localhost:3000")
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events."""
        cleanup_task = None
        if job_manager_mode:
            import asyncio
            from app.services.job_cleanup import cleanup_expired_jobs
            
            def cleanup_wrapper():
                """Wrapper to get job_state_dictio and call cleanup."""
                try:
                    modal_app = _get_modal_module()
                    job_dictio = getattr(modal_app, "job_state_dictio", None)
                    if job_dictio is not None:
                        cleanup_expired_jobs(job_dictio)
                except Exception as e:
                    print(f"⚠️ [JobManager] Cleanup error: {e}")
            
            async def periodic_cleanup():
                """Periodic cleanup task (runs every 5 minutes)."""
                try:
                    while True:
                        await asyncio.sleep(300)
                        cleanup_wrapper()
                except asyncio.CancelledError:
                    print("🛑 [JobManager] Background cleanup task cancelled")
                    raise
            
            cleanup_task = asyncio.create_task(periodic_cleanup())
            print("✅ [JobManager] Background cleanup scheduler started (runs every 5 minutes)")
        
        yield
        
        # Shutdown: cancel cleanup task if exists
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
            print("✅ [JobManager] Background cleanup task stopped")
    
    fastapi_app = FastAPI(
        title="ArtMancer API",
        description="AI-powered image editing with Qwen models",
        version="2.1.0",
        lifespan=lifespan,
    )

    # Configure CORS with security best practices
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["Authorization", "Content-Type"],
    )
    
    # Remove server header for security
    @fastapi_app.middleware("http")
    async def remove_server_header(request, call_next):
        response = await call_next(request)
        if "server" in response.headers:
            del response.headers["server"]
        return response

    # Import common routers
    from app.api.endpoints import system, debug
    fastapi_app.include_router(system.router)
    fastapi_app.include_router(debug.router)

    # Conditionally include service-specific routers
    if job_manager_mode:
        from app.api.endpoints import generation_async
        fastapi_app.include_router(generation_async.router)

    if include_segmentation:
        try:
            from app.api.endpoints import smart_mask
            fastapi_app.include_router(smart_mask.router)
            print(f"✅ [FastAPI] Included smart_mask router with prefix: {smart_mask.router.prefix}")
            # Log all routes for debugging
            for route in fastapi_app.routes:
                route_path = getattr(route, "path", None)
                route_methods = getattr(route, "methods", None)
                if route_path and route_methods:
                    print(f"   📍 Route: {list(route_methods)} {route_path}")
        except Exception as e:
            print(f"❌ [FastAPI] Failed to include smart_mask router: {e}")
            import traceback
            traceback.print_exc()
            raise

    if include_image_utils:
        from app.api.endpoints import image_utils
        fastapi_app.include_router(image_utils.router)

    # Add ping endpoint for health checks
    @fastapi_app.get("/ping")
    async def ping():
        return {"status": "healthy"}

    return fastapi_app

# ==========================================
# 6. SERVICE CLASSES
# ==========================================
# Each service is a Modal class that defines a container with lifecycle hooks.
# Services can be deployed independently and scale based on demand.

def _service_lifecycle_mixin(service_name: str):
    """
    Mixin for common service lifecycle patterns (startup/shutdown tracking).
    
    Why: All services share similar lifecycle hooks. This reduces duplication
    while maintaining flexibility for service-specific initialization.
    """
    class ServiceLifecycleMixin:
        container_start_time: float | None = None
        is_ready: bool = False

        @modal.enter()
        def prepare(self):
            """Container startup hook."""
            self.container_start_time = time.time()
            self.is_ready = False
            print(f"🚀 [{service_name}] Container starting up...")
            # Service-specific initialization happens here (override if needed)
            self.is_ready = True
            uptime = time.time() - self.container_start_time
            print(f"✅ [{service_name}] Container ready! Startup took {uptime:.2f}s")

        @modal.exit()
        def cleanup(self):
            """Container shutdown hook."""
            if self.container_start_time:
                uptime = time.time() - self.container_start_time
                print(
                    f"🛑 [{service_name}] Container shutting down after {uptime:.2f}s of uptime"
                )
            self.is_ready = False
            print(f"✅ [{service_name}] Cleanup complete")
    
    return ServiceLifecycleMixin

# 6.1 JOB MANAGER SERVICE (CPU-only, Cold boot enabled)
@app.cls(
    image=cpu_image,
    cpu=1,
    min_containers=0,
    timeout=1800,
    scaledown_window=1200,
    volumes={VOL_MOUNT_PATH: volume},
)
class JobManagerService(_service_lifecycle_mixin("JobManagerService")):
    """CPU-based job manager for coordinating H100 inference jobs."""

    @modal.asgi_app(label="job-manager")
    def serve(self):
        """Job manager endpoint for async generation coordination."""
        return create_fastapi_app(job_manager_mode=True)

# 6.2 API GATEWAY SERVICE (CPU-only, Cold boot enabled)
@app.cls(
    image=cpu_image,
    cpu=1,
    min_containers=0,
    timeout=1800,
    scaledown_window=1200,
)
class APIGatewayService(_service_lifecycle_mixin("APIGatewayService")):
    """API Gateway - Single entry point for all requests."""

    @modal.enter()
    def prepare(self):
        """Container startup hook with service URL configuration."""
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [APIGatewayService] Container starting up...")
        
        # Set service URLs from environment or defaults
        service_urls = {
            "SEGMENTATION_SERVICE_URL": "https://nxan2911--segmentation.modal.run",
            "IMAGE_UTILS_SERVICE_URL": "https://nxan2911--image-utils.modal.run",
            "JOB_MANAGER_SERVICE_URL": "https://nxan2911--job-manager.modal.run",
        }
        
        for env_key, default_url in service_urls.items():
            os.environ.setdefault(env_key, default_url)
        
        print("📋 [APIGatewayService] Service URLs configured:")
        for env_key in service_urls:
            print(f"   - {env_key.replace('_', ' ').title()}: {os.environ[env_key]}")
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [APIGatewayService] Container ready! Startup took {uptime:.2f}s")

    @modal.asgi_app(label="api-gateway")
    def serve(self):
        """API Gateway endpoint - Single entry point for all requests."""
        from api_gateway.main import create_app
        return create_app()

# 6.3 SEGMENTATION SERVICE (T4)
@app.cls(
    image=segmentation_image,
    gpu="T4",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=1800,
    min_containers=0,
    max_containers=5,
    scaledown_window=1200,  # 20 minutes idle before shutdown (was 120s)
)
@modal.concurrent(max_inputs=20)
class SegmentationService(_service_lifecycle_mixin("SegmentationService")):
    """Segmentation service for smart mask generation."""

    segmentation_model: object | None = None

    @modal.enter()
    def warmup(self):
        """Container startup hook: pre-load segmentation model."""
        self.container_start_time = time.time()
        self.is_ready = False
        self.segmentation_model = None
        print("🚀 [SegmentationService] Container starting up...")
        
        import torch
        print(f"🔧 [SegmentationService] CUDA available: {torch.cuda.is_available()}")
        
        # Pre-load FastSAM model from Volume if available
        try:
            from ultralytics import FastSAM  # type: ignore[attr-defined]
            
            vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
            segmentation_model_path = vol_segmentation_file if os.path.exists(vol_segmentation_file) else "FastSAM-s.pt"
            
            if segmentation_model_path == vol_segmentation_file:
                print(f"🚀 FastSAM-s.pt loaded directly from Volume: {vol_segmentation_file}")
            else:
                print("⚠️ FastSAM-s.pt not in Volume, will download from GitHub (run setup_volume to cache)")
            
            print(f"🔄 Loading FastSAM-s from: {segmentation_model_path}")
            self.segmentation_model = FastSAM(segmentation_model_path)
            print("✅ FastSAM-s model loaded")
        except Exception as e:
            print(f"⚠️ Segmentation model warmup failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [SegmentationService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """Container shutdown hook."""
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [SegmentationService] Container shutting down after {uptime:.2f}s of uptime")
        else:
            print("🛑 [SegmentationService] Container shutting down")
        self.segmentation_model = None
        self.is_ready = False
        print("✅ [SegmentationService] Cleanup complete")

    @modal.asgi_app(label="segmentation")
    def serve(self):
        """Segmentation service endpoint for smart mask generation."""
        app = create_fastapi_app(include_segmentation=True)
        print(f"🚀 [SegmentationService] FastAPI app created with {len(app.routes)} routes")
        for route in app.routes:
            route_path = getattr(route, "path", None)
            route_methods = getattr(route, "methods", None)
            if route_path and route_methods:
                print(f"   📍 Route: {list(route_methods)} {route_path}")
        return app

# 6.4 IMAGE UTILS SERVICE (CPU-only)
@app.cls(
    image=imageutils_image,
    cpu=1,
    timeout=1800,
    min_containers=0,
    scaledown_window=900,  # 15 minutes idle before shutdown (was 60s)
)
@modal.concurrent(max_inputs=50)
class ImageUtilsService(_service_lifecycle_mixin("ImageUtilsService")):
    """Image utilities service for resize, crop, encode, etc."""

    @modal.asgi_app(label="image-utils")
    def serve(self):
        """Image utils service endpoint for image processing operations."""
        return create_fastapi_app(include_image_utils=True)

# ==========================================
# 7. WORKER FUNCTIONS
# ==========================================

@app.function(
    image=heavy_image,
    gpu="H100",
    timeout=1800,
    volumes={VOL_MOUNT_PATH: volume},
    max_containers=1,
    scaledown_window=1800,
)
def qwen_worker(task_id: str, payload: dict):
    """
    H100 worker function for async image generation.
    
    Workflow:
    1. Updates job state to "initializing_h100"
    2. Checks if pipeline is already loaded (container reuse optimization)
    3. Loads Qwen pipeline from Volume only if needed
    4. Updates job state to "processing" with progress callbacks
    5. Runs generation with progress tracking
    6. Updates job state to "done" with result (base64 image)
    
    Args:
        task_id: Unique task identifier
        payload: Generation request payload (same as GenerationRequest schema)
    
    Returns:
        dict with status and result (for logging/debugging)
    """
    try:
        # Initialize job state
        update_job_state(task_id, "initializing_h100", 0.1)
        print(f"🚀 [H100 Worker] Task {task_id}: Initializing H100 container...")
        
        # Setup environment: load model trực tiếp từ Volume (không copy)
        os.environ.setdefault("TASK_CHECKPOINTS_PATH", VOL_TASKS_PATH)
        
        if Path(VOL_MODEL_PATH).exists():
            os.environ["MODEL_PATH"] = VOL_MODEL_PATH
            print(f"🚀 MODEL_PATH set to {VOL_MODEL_PATH} (loaded directly from Volume)")
        else:
            print(f"⚠️ Base model snapshot not found at {VOL_MODEL_PATH}. qwen_loader will load from Hugging Face ({BASE_MODEL_ID}).")
        
        # Import here to avoid loading on module import
        from app.models.schemas import GenerationRequest
        from app.services.generation_service import GenerationService
        
        # Check if pipeline can be reused (container reuse optimization)
        request = GenerationRequest(**payload)
        task_type = request.task_type or "insertion"
        pipeline_already_loaded = check_pipeline_reuse(task_id, task_type)
        
        if pipeline_already_loaded:
            print(f"✅ [H100 Worker] Task {task_id}: Pipeline ready, skipping load (container reuse)")
        else:
            print(f"🔄 [H100 Worker] Task {task_id}: Pipeline will be loaded by GenerationService...")
        
        # Update state: loading_pipeline
        update_job_state(task_id, "loading_pipeline", 0.2)
        
        # Create callbacks for progress tracking
        def on_loading_progress(message: str, progress: float):
            """Callback to update loading progress during pipeline loading."""
            # Map progress (0.0-1.0) to overall progress (0.2-0.3)
            overall_progress = 0.2 + (progress * 0.1)
            update_job_state(
                task_id,
                "loading_pipeline",
                overall_progress,
                loading_message=message,
            )
            print(f"📊 [H100 Worker] Task {task_id}: Loading progress - {message} ({progress*100:.1f}%)")
        
        def on_pipeline_loaded():
            """Callback to update status when pipeline finishes loading."""
            print(f"📢 [H100 Worker] Task {task_id}: on_pipeline_loaded callback called, updating status to processing...")
            update_job_state(
                task_id,
                "processing",
                0.3,
                current_step=None,
                total_steps=None,
                loading_message=None,
            )
            print(f"✅ [H100 Worker] Task {task_id}: Status updated to 'processing', pipeline ready for generation")
        
        def progress_callback(step: int, timestep: int, total_steps: int):
            """Callback to update progress during generation with cancellation check."""
            try:
                current_state = job_state_dictio.get(task_id, {})
                
                # Check cancellation flag
                if current_state.get("cancelled", False):
                    print(f"⚠️ [H100 Worker] Task {task_id} cancelled at step {step}/{total_steps}")
                    update_job_state(
                        task_id,
                        "cancelled",
                        0.3 + (step / total_steps) * 0.65,
                        error="Generation cancelled by user",
                    )
                    raise ValueError(f"Task {task_id} was cancelled")
                
                # Calculate progress: 0.3 (pipeline loaded) to 0.95 (before final processing)
                # Note: diffusers callback provides step as 0-based, we add 1 for 1-based display
                generation_progress = 0.3 + (step / total_steps) * 0.65
                update_job_state(
                    task_id,
                    "processing",
                    generation_progress,
                    current_step=step + 1,
                    total_steps=total_steps,
                )
            except ValueError as e:
                if "cancelled" in str(e).lower():
                    raise
                print(f"⚠️ [H100 Worker] Failed to update progress: {e}")
            except Exception as e:
                print(f"⚠️ [H100 Worker] Failed to update progress: {e}")
        
        # Run generation with progress callbacks
        print(f"🎨 [H100 Worker] Task {task_id}: Running generation...")
        generation_service = GenerationService()
        result = generation_service.generate(
            request, 
            progress_callback=progress_callback, 
            on_pipeline_loaded=on_pipeline_loaded,
            on_loading_progress=on_loading_progress,
        )
        
        # Extract and encode result image
        generated_pil = result.get("generated_pil")
        if not generated_pil:
            raise ValueError("Generation result missing generated_pil image")
        
        # Encode to base64 for job_state_dictio
        from app.services.image_processing import image_to_base64
        image_base64 = image_to_base64(generated_pil)
        
        # Update state: done
        update_job_state(
            task_id,
            "done",
            1.0,
            result=image_base64,
            debug_info=result.get("debug_info"),
        )
        
        print(f"✅ [H100 Worker] Task {task_id}: Generation completed")
        
        # Keep pipeline in memory for container reuse (flush GPU cache only)
        try:
            from app.core.qwen_loader import _flush_memory, _pipeline as global_pipeline
            if global_pipeline is not None:
                print(f"🧹 [H100 Worker] Task {task_id}: Flushing GPU memory cache (keeping model in memory for container reuse)...")
                _flush_memory()
                print(f"✅ [H100 Worker] Task {task_id}: GPU memory flushed (model kept in memory for reuse)")
        except Exception as e:
            print(f"⚠️ [H100 Worker] Task {task_id}: Memory flush warning: {e}")
        
        return {"status": "done", "task_id": task_id, "result": image_base64}
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [H100 Worker] Task {task_id}: Error - {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Update state: error
        update_job_state(task_id, "error", 0.0, error=error_msg)
        
        # Cleanup on critical errors only (OOM, corruption)
        try:
            from app.core.qwen_loader import clear_qwen_cache
            if "out of memory" in error_msg.lower() or "corrupt" in error_msg.lower():
                print(f"🧹 [H100 Worker] Task {task_id}: Cleaning up pipeline cache after critical error...")
                clear_qwen_cache()
            else:
                print(f"⚠️ [H100 Worker] Task {task_id}: Non-critical error, keeping pipeline in memory for reuse")
        except Exception:
            pass
        
        return {"status": "error", "task_id": task_id, "error": error_msg}

# ==========================================
# 8. VOLUME MANAGEMENT
# ==========================================

@app.function(volumes={VOL_MOUNT_PATH: volume}, timeout=60)
def remove_checkpoint_file(filename: str) -> bool:
    """
    Remove a specific checkpoint file from Modal Volume.
    
    Args:
        filename: Name of the checkpoint file to remove
    
    Returns:
        True if file was removed, False otherwise
    """
    file_path = f"{VOL_MOUNT_PATH}/{filename}"
    
    if not os.path.exists(file_path):
        print(f"  ℹ️ File not found: {filename}")
        return False

    try:
        os.remove(file_path)
        volume.commit()
        print(f"  🗑️ Removed: {filename}")
        return True
    except Exception as e:
        print(f"  ⚠️ Failed to remove {file_path}: {e}")
        return False

@app.function(volumes={VOL_MOUNT_PATH: volume}, timeout=60)
def clear_checkpoints() -> list[str]:
    """
    Remove all .safetensors files from Modal Volume.
    
    Usage:
        modal run modal_app.py::clear_checkpoints
    
    Returns:
        List of removed filenames
    """
    import glob
    
    print(f"🗑️ Clearing all checkpoints from Volume '{VOLUME_NAME}' ...")
    
    pattern = f"{VOL_MOUNT_PATH}/*.safetensors"
    files = glob.glob(pattern)
    
    if not files:
        print("  No checkpoint files found to remove.")
        return []
    
    removed = []
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            os.remove(file_path)
            print(f"  🗑️ Removed: {filename}")
            removed.append(filename)
        except Exception as e:
            print(f"  ⚠️ Failed to remove {file_path}: {e}")
    
    if removed:
        volume.commit()
    
    print(f"✅ Removed {len(removed)} checkpoint file(s).")
    return removed

@app.local_entrypoint()
def upload_checkpoints(local_checkpoints_dir: str = "./checkpoints", force: bool = False):
    """
    Upload LoRA checkpoint files (.safetensors) from local to Volume root.
    
    Expected checkpoint files for 3 tasks:
    - insertion_cp.safetensors (for insertion task)
    - removal_cp.safetensors (for removal task)
    - wb_cp.safetensors (for white-balance task)

    Usage:
        modal run modal_app.py::upload_checkpoints --local-checkpoints-dir ./checkpoints
        modal run modal_app.py::upload_checkpoints --force
    """
    from pathlib import Path

    local_path = Path(local_checkpoints_dir)
    if not local_path.exists():
        print(f"Error: Local checkpoints directory not found: {local_path}")
        return

    checkpoint_files = list(local_path.glob("*.safetensors"))
    if not checkpoint_files:
        print(f"❌ No .safetensors files found in {local_path}")
        return

    # Expected checkpoint files for 3 tasks
    expected_files = {
        "insertion_cp.safetensors": "Insertion",
        "removal_cp.safetensors": "Removal",
        "wb_cp.safetensors": "White-balance",
    }
    
    found_files = {f.name for f in checkpoint_files}
    missing_files = set(expected_files.keys()) - found_files
    extra_files = found_files - set(expected_files.keys())
    
    # Print file summary
    print(f"\n📋 Checkpoint files found in {local_path}:")
    for checkpoint_file in sorted(checkpoint_files):
        task_name = expected_files.get(checkpoint_file.name, "Unknown")
        print(f"   ✅ {checkpoint_file.name} ({task_name})")
    
    if missing_files:
        print("\n⚠️  Missing expected checkpoint files:")
        for missing in sorted(missing_files):
            print(f"   ❌ {missing} ({expected_files[missing]})")
        print("\n💡 Note: Generation will still work, but missing tasks will use base model only.")
    
    if extra_files:
        print("\nℹ️  Additional checkpoint files found (will also be uploaded):")
        for extra in sorted(extra_files):
            print(f"   📦 {extra}")
    
    print(f"\n📤 Uploading {len(checkpoint_files)} checkpoint file(s) to Volume '{VOLUME_NAME}' ...")
    
    # Clear existing checkpoints if force mode
    if force:
        print("🔄 Force mode: clearing existing checkpoints first...")
        clear_checkpoints.remote()
        print("✅ Existing checkpoints cleared.")
    else:
        print("🔍 Checking for existing files...")
        for checkpoint_file in checkpoint_files:
            filename = checkpoint_file.name
            print(f"  Checking: {filename}")
            if remove_checkpoint_file.remote(filename):
                print(f"  ✅ Removed existing {filename}")

    # Upload files
    print("\n📤 Starting upload...")
    uploaded_count = 0
    with volume.batch_upload() as upload:
        for checkpoint_file in checkpoint_files:
            remote_path = f"/{checkpoint_file.name}"
            task_name = expected_files.get(checkpoint_file.name, "Additional")
            print(f"  📤 Uploading {checkpoint_file.name} ({task_name}) -> {remote_path}")
            try:
                upload.put_file(checkpoint_file, remote_path)
                uploaded_count += 1
                print(f"  ✅ Successfully uploaded {checkpoint_file.name}")
            except FileExistsError:
                print(f"  ⚠️ File already exists: {checkpoint_file.name}")
                print("  💡 Tip: Use --force flag to overwrite all files")
                print("  🔄 Attempting to remove and re-upload...")
                remove_checkpoint_file.remote(checkpoint_file.name)
                upload.put_file(checkpoint_file, remote_path)
                uploaded_count += 1
                print(f"  ✅ Successfully uploaded {checkpoint_file.name}")
            except Exception as e:
                print(f"  ❌ Failed to upload {checkpoint_file.name}: {e}")

    print(f"\n✅ Upload complete: {uploaded_count}/{len(checkpoint_files)} files uploaded.")
    print(f"   Files are now available at {VOL_MOUNT_PATH}/ in Modal containers.")
    
    # Summary
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} expected checkpoint file(s) missing.")
        print(f"   Tasks will work but may use base model only for: {', '.join(expected_files[f] for f in missing_files)}")
    else:
        print("\n✅ All 3 task checkpoints are present: Insertion, Removal, White-balance")

@app.local_entrypoint()
def upload_depth_model(local_checkpoints_dir: str = "./checkpoints", force: bool = False):
    """
    Upload depth_anything_v2.pth model file to Modal Volume root.
    
    Usage:
        modal run modal_app.py::upload_depth_model
        modal run modal_app.py::upload_depth_model --local-checkpoints-dir ./checkpoints --force
    """
    from pathlib import Path
    
    local_path = Path(local_checkpoints_dir)
    if not local_path.exists():
        print(f"❌ Error: Local checkpoints directory not found: {local_path}")
        return
    
    model_file = local_path / "depth_anything_v2.pth"
    if not model_file.exists():
        print(f"❌ Error: depth_anything_v2.pth not found in {local_path}")
        return
    
    print(f"📤 Uploading depth_anything_v2.pth to Volume '{VOLUME_NAME}' ...")
    
    # Remove existing file if force mode or if exists
    remote_path = "/depth_anything_v2.pth"
    if force:
        print("🔄 Force mode: removing existing file if any...")
        try:
            remove_checkpoint_file.remote("depth_anything_v2.pth")
            print("✅ Existing file removed.")
        except Exception:
            pass  # File might not exist
    
    # Upload file
    try:
        with volume.batch_upload() as upload:
            print(f"  📤 Uploading {model_file.name} -> {remote_path}")
            upload.put_file(model_file, remote_path)
            print(f"  ✅ Successfully uploaded {model_file.name}")
        
        # Commit changes
        volume.commit()
        print(f"\n✅ Upload complete: {model_file.name} is now available at {VOL_MOUNT_PATH}/ in Modal containers.")
    except FileExistsError:
        print(f"  ⚠️ File already exists: {model_file.name}")
        if not force:
            print("  💡 Tip: Use --force flag to overwrite existing file")
            print("  🔄 Attempting to remove and re-upload...")
            remove_checkpoint_file.remote("depth_anything_v2.pth")
            with volume.batch_upload() as upload:
                upload.put_file(model_file, remote_path)
            volume.commit()
            print(f"  ✅ Successfully uploaded {model_file.name}")
    except Exception as e:
        print(f"  ❌ Failed to upload {model_file.name}: {e}")

def _calculate_directory_size(directory: str) -> float:
    """Calculate total size of directory in MB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / 1024 / 1024

def _check_model_exists(model_path: str, check_files: list[str] | None = None) -> bool:
    """
    Check if a model already exists in the given path.
    
    Args:
        model_path: Path to model directory
        check_files: List of files to check for existence. If None, checks for model_index.json
    
    Returns:
        True if model exists, False otherwise
    """
    if check_files is None:
        check_files = ["model_index.json"]
    
    if not os.path.exists(model_path):
        return False
    
    # Check if at least one of the required files exists
    for check_file in check_files:
        file_path = os.path.join(model_path, check_file)
        if os.path.exists(file_path):
            return True
    
    return False

@app.function(image=heavy_image, volumes={VOL_MOUNT_PATH: volume}, timeout=3600)
def setup_volume(force: bool = False):
    """
    Download Qwen base model, FastSAM model, BiRefNet model, and Stable Diffusion Inpainting model into Modal Volume.

    This is optional but recommended. If not run, the models will be downloaded
    on first use (slower cold-start).

    Args:
        force: If True, re-download models even if they already exist. Default: False (skip if exists).

    Usage:
        modal run modal_app.py::setup_volume
        modal run modal_app.py::setup_volume --force  # Force re-download all models
    """
    from huggingface_hub import snapshot_download
    import urllib.request

    print("🚀 Starting setup_volume: downloading models to Volume...")
    
    # 1. Download Qwen base model
    print("\n📦 [1/4] Checking Qwen base model...")
    os.makedirs(VOL_MODEL_PATH, exist_ok=True)

    if _check_model_exists(VOL_MODEL_PATH, ["model_index.json", "tokenizer_config.json"]) and not force:
        print(f"⏭️  Qwen base model already exists in {VOL_MODEL_PATH}. Skipping download.")
        print("   Use --force to re-download.")
    else:
        if force:
            print("🔄 Force mode: Re-downloading Qwen base model...")
        try:
            snapshot_download(
                repo_id=BASE_MODEL_ID,
                local_dir=VOL_MODEL_PATH,
                ignore_patterns=["*.msgpack", "*.bin", ".git*"],
            )
            print(f"✅ Base model downloaded to {VOL_MODEL_PATH}")
            print("📂 Current contents of base model directory:")
            try:
                print(os.listdir(VOL_MODEL_PATH))
            except Exception:
                # Listing contents is best-effort; ignore errors here
                pass
        except Exception as exc:
            print(f"❌ Failed to download base model: {exc}")
            return

    # 2. Download FastSAM model
    print("\n📦 [2/4] Checking FastSAM-s.pt...")
    os.makedirs(VOL_SEGMENTATION_PATH, exist_ok=True)
    
    fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"
    vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
    
    if os.path.exists(vol_segmentation_file) and not force:
        file_size = os.path.getsize(vol_segmentation_file) / 1024 / 1024
        print(f"⏭️  FastSAM-s.pt already exists ({file_size:.1f} MB). Skipping download.")
        print("   Use --force to re-download.")
    else:
        if force:
            print("🔄 Force mode: Re-downloading FastSAM-s.pt...")
        try:
            print(f"📥 Downloading FastSAM-s.pt from {fastsam_url}...")
            urllib.request.urlretrieve(fastsam_url, vol_segmentation_file)
            file_size = os.path.getsize(vol_segmentation_file) / 1024 / 1024
            print(f"✅ FastSAM-s.pt downloaded to {vol_segmentation_file} ({file_size:.1f} MB)")
            print("✅ FastSAM-s.pt cached in Volume. SegmentationService will use this for faster cold starts.")
        except Exception as exc:
            print(f"❌ Failed to download FastSAM-s.pt: {exc}")
            import traceback
            traceback.print_exc()
            return
    
    # 3. Download BiRefNet model
    print("\n📦 [3/4] Checking BiRefNet model...")
    os.makedirs(VOL_BIREFNET_PATH, exist_ok=True)
    
    birefnet_repo_id = "zhengpeng7/BiRefNet"
    
    if _check_model_exists(VOL_BIREFNET_PATH, ["model_index.json", "config.json"]) and not force:
        total_size = _calculate_directory_size(VOL_BIREFNET_PATH)
        print(f"⏭️  BiRefNet model already exists in {VOL_BIREFNET_PATH} ({total_size:.1f} MB). Skipping download.")
        print("   Use --force to re-download.")
    else:
        if force:
            print("🔄 Force mode: Re-downloading BiRefNet model...")
        try:
            print(f"📥 Downloading BiRefNet from {birefnet_repo_id}...")
            snapshot_download(
                repo_id=birefnet_repo_id,
                local_dir=VOL_BIREFNET_PATH,
                ignore_patterns=["*.msgpack", ".git*"],
            )
            total_size = _calculate_directory_size(VOL_BIREFNET_PATH)
            print(f"✅ BiRefNet model downloaded to {VOL_BIREFNET_PATH} ({total_size:.1f} MB)")
            print("✅ BiRefNet cached in Volume. Will be loaded directly from Volume for faster cold starts.")
        except Exception as exc:
            print(f"❌ Failed to download BiRefNet model: {exc}")
            import traceback
            traceback.print_exc()
            return
    
    # 4. Download Stable Diffusion Inpainting model (for LaMa refinement)
    print("\n📦 [4/4] Checking Stable Diffusion Inpainting model...")
    os.makedirs(VOL_REFINEMENT_PATH, exist_ok=True)
    
    refinement_repo_id = "runwayml/stable-diffusion-inpainting"
    
    if _check_model_exists(VOL_REFINEMENT_PATH, ["model_index.json", "scheduler/scheduler_config.json"]) and not force:
        total_size = _calculate_directory_size(VOL_REFINEMENT_PATH)
        print(f"⏭️  Stable Diffusion Inpainting model already exists in {VOL_REFINEMENT_PATH} ({total_size:.1f} MB). Skipping download.")
        print("   Use --force to re-download.")
    else:
        if force:
            print("🔄 Force mode: Re-downloading Stable Diffusion Inpainting model...")
        try:
            print(f"📥 Downloading Stable Diffusion Inpainting from {refinement_repo_id}...")
            snapshot_download(
                repo_id=refinement_repo_id,
                local_dir=VOL_REFINEMENT_PATH,
                ignore_patterns=["*.msgpack", ".git*"],
            )
            total_size = _calculate_directory_size(VOL_REFINEMENT_PATH)
            print(f"✅ Stable Diffusion Inpainting model downloaded to {VOL_REFINEMENT_PATH} ({total_size:.1f} MB)")
            print("✅ Stable Diffusion Inpainting cached in Volume. Will be loaded directly from Volume for LaMa refinement.")
        except Exception as exc:
            print(f"❌ Failed to download Stable Diffusion Inpainting model: {exc}")
            import traceback
            traceback.print_exc()
            return
    
    # Commit all changes to Volume
    print("\n💾 Committing changes to Volume...")
    volume.commit()
    print("✅ All models cached in Volume successfully!")

# ==========================================
# 9. DEPLOYMENT & USAGE
# ==========================================
# Deploy all services:
#   modal deploy modal_app.py
#
# This deploys 4 public endpoints:
# - API Gateway: https://<username>--api-gateway.modal.run
# - Job Manager: https://<username>--job-manager.modal.run
# - Segmentation Service: https://<username>--segmentation.modal.run
# - Image Utils Service: https://<username>--image-utils.modal.run
#
# Note: H100 worker (qwen_worker) is not a public endpoint - it's spawned internally by Job Manager.
#
# All services support cold boot (scale to zero when idle) for cost optimization.
#
# For local testing with hot-reload:
#   modal serve modal_app.py
#
# Setup Volume (recommended before first deployment):
#   modal run modal_app.py::setup_volume
#   modal run modal_app.py::upload_checkpoints
