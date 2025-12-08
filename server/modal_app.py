"""
Modal app for ArtMancer backend.

This file defines all Modal services, images, and helper functions for deployment.

Architecture:
- API Gateway (CPU): Single entry point, routes requests to backend services.
- Job Manager (CPU): Coordinates async generation jobs, dispatches to H200 workers.
- Segmentation Service (T4): Smart mask segmentation.
- Image Utils Service (CPU): Image processing utilities.
- H200 Worker Function: Async image generation (spawned by Job Manager).

Features:
- Modal Volume for checkpoints + base model snapshot.
- Load model trực tiếp từ Volume (không copy) để giảm cold-start từ 20-60s xuống 1-3s.
- Cold boot enabled for cost optimization (scales to zero when idle).

Table of Contents:
1. CONFIGURATION - Constants and paths
2. INFRASTRUCTURE - Modal app, volume, dictio setup
3. IMAGE DEFINITIONS - Docker images for different services
   - cpu_image: Job Manager & API Gateway
   - heavy_image: Qwen generation (H200)
   - segmentation_image: Mask segmentation (T4)
   - imageutils_image: Image utilities (CPU)
4. create_fastapi_app() - Factory function for FastAPI apps
5. SERVICE CLASSES - Modal service definitions (4 endpoints)
   - APIGatewayService: Single entry point (https://<username>--api-gateway.modal.run)
   - JobManagerService: Async job coordination (https://<username>--job-manager.modal.run)
   - SegmentationService: Smart mask segmentation (https://<username>--segmentation.modal.run)
   - ImageUtilsService: Image processing utilities (https://<username>--image-utils.modal.run)
6. WORKER FUNCTIONS - Background tasks
   - qwen_worker(): H200 worker for async generation (spawned by Job Manager, not a public endpoint)
7. HELPER FUNCTIONS - Volume management utilities
   - remove_checkpoint_file(): Remove single checkpoint
   - clear_checkpoints(): Remove all checkpoints
   - upload_checkpoints(): Upload checkpoints from local
   - setup_volume(): Download Qwen base model, FastSAM model, and BiRefNet model
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
# CONFIGURATION
# ==========================================
APP_NAME = "artmancer"
VOLUME_NAME = "artmancer-checkpoints"
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

# Volume mount + layout
VOL_MOUNT_PATH = "/checkpoints"          # Mounted in functions
VOL_MODEL_PATH = f"{VOL_MOUNT_PATH}/base_model"
VOL_TASKS_PATH = VOL_MOUNT_PATH         # Keep LoRA checkpoints at /checkpoints/*.safetensors
VOL_SEGMENTATION_PATH = f"{VOL_MOUNT_PATH}/segmentation"  # Segmentation models cache (FastSAM-s.pt, load trực tiếp từ Volume, không copy)
VOL_BIREFNET_PATH = f"{VOL_MOUNT_PATH}/birefnet"  # BiRefNet model cache (load trực tiếp từ Volume, không copy)

# ==========================================
# INFRASTRUCTURE
# ==========================================
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Job state storage using Modal Dictio (persistent across container restarts)
job_state_dictio = modal.Dict.from_name("artmancer-job-state", create_if_missing=True)

# ==========================================
# 3. IMAGE DEFINITIONS
# ==========================================
# Docker images for different services. Each image includes only necessary dependencies
# to minimize image size and cold-start time.

# 3.1 CPU IMAGE (Job Manager & API Gateway) - Minimal, no GPU
# ==========================================
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Minimal dependencies for job management and API Gateway
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
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
    .add_local_dir("api_gateway", "/root/api_gateway", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
    .add_local_dir("shared", "/root/shared", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
)

# 3.2 HEAVY IMAGE (H200) - Optimized for GPU inference
# ==========================================
# Includes: diffusers, transformers, torch, peft for Qwen image generation
heavy_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Install diffusers from GitHub (pinned version) for QwenImageEditPlusPipeline support
        "uv pip install --system --no-cache-dir git+https://github.com/huggingface/diffusers.git",
        # Install other dependencies with pinned versions
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "torch>=2.4.0 "
        "torchvision>=0.19.0 "
        "transformers==4.57.3 "
        "accelerate==1.12.0 "
        "safetensors>=0.4.3 "
        "fastsafetensors>=0.1.0 "  # Optional: faster safetensors loading (experimental)
        "peft>=0.10.0 "
        "pillow>=10.3.0 "
        "numpy>=1.26.4 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "python-dotenv "
        "huggingface-hub[cli] "
        "requests "
        "hf-transfer "
        "psutil "
        "boto3 "
        "awscli "
        "opencv-python-headless "  # For MAE image generation
        "scikit-image"  # For image processing utilities
    )
    .env(
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
)

# 3.3 SEGMENTATION IMAGE (T4) - Mask segmentation service
# ==========================================
# Includes: ultralytics, torch (CUDA), scikit-image for mask processing
segmentation_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Use CUDA torch for T4 GPU acceleration (segmentation models)
        "uv pip install --system --no-cache-dir "
        "torch>=2.4.0 "
        "torchvision>=0.19.0 "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "pillow>=10.3.0 "
        "numpy>=1.26.4 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "ultralytics==8.3.235 "  # FastSAM (pinned version)
        "transformers>=4.57.3 "  # BiRefNet (AutoModelForImageSegmentation)
        "einops "  # Required by BiRefNet
        "kornia "  # Required by BiRefNet
        "timm "  # Required by BiRefNet
        "opencv-python-headless "
        "scikit-image "  # For mask processing (binary_dilation, gaussian, etc.)
        "psutil"
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics", "PYTHONPATH": "/root"})
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
)

# 3.4 IMAGE UTILS IMAGE (CPU-only) - Minimal dependencies
# ==========================================
# Includes: OpenCV, scikit-image for image processing utilities
imageutils_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Minimal dependencies for image utils (resize, crop, encode, etc.)
        # Includes OpenCV for MAE and Canny generation
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "pillow>=10.3.0 "
        "numpy>=1.26.4 "
        "pydantic>=2.7.0 "
        "pydantic-settings>=2.2.0 "
        "python-multipart "
        "opencv-python-headless>=4.8.0 "
        "scikit-image>=0.25.2 "
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
)


# ==========================================
# 4. FASTAPI APP FACTORY
# ==========================================
# Factory function to create FastAPI apps with different router configurations
# depending on which service is being deployed.

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
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Get allowed origins from environment or default
    # Production: Set ALLOWED_ORIGINS env var with comma-separated list of allowed origins
    # Example: ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
    allowed_origins_str = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,https://localhost:3000"
    )
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

    # Lifespan context manager for startup/shutdown events
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events."""
        # Startup
        cleanup_task = None
        if job_manager_mode:
            # Start background cleanup scheduler for job manager
            import asyncio
            from app.services.job_cleanup import cleanup_expired_jobs
            
            def cleanup_wrapper():
                """Wrapper to get job_state_dictio and call cleanup."""
                try:
                    import sys
                    if 'modal_app' not in sys.modules:
                        import modal_app
                    else:
                        modal_app = sys.modules['modal_app']
                    job_state_dictio = getattr(modal_app, 'job_state_dictio', None)
                    if job_state_dictio is not None:
                        cleanup_expired_jobs(job_state_dictio)
                except Exception as e:
                    print(f"⚠️ [JobManager] Cleanup error: {e}")
            
            async def periodic_cleanup():
                """Periodic cleanup task."""
                try:
                    while True:
                        await asyncio.sleep(300)  # 5 minutes
                        cleanup_wrapper()
                except asyncio.CancelledError:
                    print("🛑 [JobManager] Background cleanup task cancelled")
                    raise
            
            cleanup_task = asyncio.create_task(periodic_cleanup())
            print("✅ [JobManager] Background cleanup scheduler started (runs every 5 minutes)")
        
        yield
        
        # Shutdown
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
        allow_methods=["POST", "GET"],  # Only allow necessary methods
        allow_headers=["Authorization", "Content-Type"],  # Only allow necessary headers
    )
    
    # Remove server header for security
    @fastapi_app.middleware("http")
    async def remove_server_header(request, call_next):
        response = await call_next(request)
        # Remove "server" header for security (MutableHeaders doesn't have pop, use del instead)
        if "server" in response.headers:
            del response.headers["server"]
        return response

    # Import common routers (no heavy dependencies)
    # system and debug routers should not require heavy dependencies
    # They use lazy imports for heavy modules
    from app.api.endpoints import system, debug
    fastapi_app.include_router(system.router)
    fastapi_app.include_router(debug.router)

    if job_manager_mode:
        # Job Manager: only async generation endpoints (no GPU, no heavy deps)
        from app.api.endpoints import generation_async
        fastapi_app.include_router(generation_async.router)
        # Background cleanup is handled in lifespan context manager above

    if include_segmentation:
        # Segmentation Service: smart mask segmentation (T4 GPU)
        try:
            from app.api.endpoints import smart_mask
            fastapi_app.include_router(smart_mask.router)
            print(f"✅ [FastAPI] Included smart_mask router with prefix: {smart_mask.router.prefix}")
            # Log all routes for debugging
            for route in fastapi_app.routes:
                route_path = getattr(route, 'path', None)
                route_methods = getattr(route, 'methods', None)
                if route_path and route_methods:
                    print(f"   📍 Route: {list(route_methods)} {route_path}")
        except Exception as e:
            print(f"❌ [FastAPI] Failed to include smart_mask router: {e}")
            import traceback
            traceback.print_exc()
            raise

    if include_image_utils:
        # Image Utils Service: image processing utilities (CPU-only)
        from app.api.endpoints import image_utils
        fastapi_app.include_router(image_utils.router)

    # Add ping endpoint for health checks
    @fastapi_app.get("/ping")
    async def ping():
        return {"status": "healthy"}

    return fastapi_app


# ==========================================
# 5. SERVICE CLASSES
# ==========================================
# Each service is a Modal class that defines a container with lifecycle hooks.
# Services can be deployed independently and scale based on demand.

# 5.1 JOB MANAGER SERVICE (CPU-only, Cold boot enabled, 20min idle timeout)
# ==========================================
# Coordinates async generation jobs. Receives job requests and dispatches them
# to H200 workers. Manages job state in job_state_dictio.
@app.cls(
    image=cpu_image,
    cpu=1,  # Minimal CPU (cheapest option - only coordination, no heavy processing)
    min_containers=0,  # Allow cold boot (scale to zero when idle)
    timeout=1800,  # 30 minutes - đủ thời gian cho cold boot và SSE streams khi bảo vệ đồ án
    scaledown_window=1200,  # Scale down after 20 minutes of inactivity
    volumes={VOL_MOUNT_PATH: volume},  # Mount volume for saving multi-step outputs
)
class JobManagerService:
    """
    CPU-based job manager for coordinating H200 inference jobs.
    Cold boot enabled - container starts on demand and scales down after 20 minutes of inactivity.
    Keeps container warm while frontend is active, saves costs when idle.
    Timeout 30 phút để đảm bảo không bị treo khi cold boot trong lúc bảo vệ đồ án.
    """
    container_start_time: float | None = None
    is_ready: bool = False

    @modal.enter()
    def prepare(self):
        """Container startup hook."""
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [JobManagerService] CPU container starting up...")
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [JobManagerService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """Container shutdown hook."""
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [JobManagerService] Container shutting down after {uptime:.2f}s of uptime")
        self.is_ready = False
        print("✅ [JobManagerService] Cleanup complete")

    @modal.asgi_app(label="job-manager")
    def serve(self):
        """
        Job manager endpoint for async generation coordination.
        Exposed at: https://<username>--artmancer-job-manager.modal.run
        """
        return create_fastapi_app(job_manager_mode=True)


# 5.2 API GATEWAY SERVICE (CPU-only, Cold boot enabled, 20min idle timeout)
# ==========================================
# Single entry point for all frontend requests. Routes requests to appropriate
# backend services (Generation, Segmentation, Image Utils, Job Manager).
@app.cls(
    image=cpu_image,
    cpu=1,  # Minimal CPU
    min_containers=0,  # Allow cold boot (scale to zero when idle)
    timeout=1800,  # 30 minutes - đủ thời gian cho cold boot khi bảo vệ đồ án
    scaledown_window=1200,  # Scale down after 20 minutes of inactivity
)
class APIGatewayService:
    """
    API Gateway - Single entry point for all requests.
    Routes requests to appropriate services (Generation, Segmentation, Image Utils, Job Manager).
    Cold boot enabled - container starts on demand and scales down after 20 minutes of inactivity.
    Keeps container warm while frontend is active, saves costs when idle.
    Timeout 30 phút để đảm bảo không bị treo khi cold boot trong lúc bảo vệ đồ án.
    """
    container_start_time: float | None = None
    is_ready: bool = False

    @modal.enter()
    def prepare(self):
        """Container startup hook."""
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [APIGatewayService] Container starting up...")
        
        # Set service URLs from environment or defaults
        segmentation_url = os.environ.setdefault(
            "SEGMENTATION_SERVICE_URL",
            "https://nxan2911--segmentation.modal.run"
        )
        image_utils_url = os.environ.setdefault(
            "IMAGE_UTILS_SERVICE_URL",
            "https://nxan2911--image-utils.modal.run"
        )
        job_manager_url = os.environ.setdefault(
            "JOB_MANAGER_SERVICE_URL",
            "https://nxan2911--job-manager.modal.run"
        )
        
        # Log service URLs for verification
        print("📋 [APIGatewayService] Service URLs configured:")
        print(f"   - Segmentation Service: {segmentation_url}")
        print(f"   - Image Utils Service: {image_utils_url}")
        print(f"   - Job Manager Service: {job_manager_url}")
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [APIGatewayService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """Container shutdown hook."""
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [APIGatewayService] Container shutting down after {uptime:.2f}s of uptime")
        self.is_ready = False
        print("✅ [APIGatewayService] Cleanup complete")

    @modal.asgi_app(label="api-gateway")
    def serve(self):
        """
        API Gateway endpoint - Single entry point for all requests.
        Exposed at: https://<username>--api-gateway.modal.run
        """
        from api_gateway.main import create_app
        return create_app()


# 5.4 SEGMENTATION SERVICE (T4) - Smart mask segmentation
# ==========================================
# Segmentation models (FastSAM, BiRefNet) for intelligent mask generation from bbox/points/strokes.
@app.cls(
    image=segmentation_image,
    gpu="T4",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=1800,  # 30 minutes - đủ thời gian cho cold boot và model loading khi bảo vệ đồ án
    min_containers=0,  # Cold start - only run when needed
    max_containers=5,  # Limit max containers to avoid GPU resource exhaustion
    scaledown_window=120,  # Scale down quickly (2 minutes)
)
@modal.concurrent(max_inputs=20)  # Allow multiple concurrent requests
class SegmentationService:
    """
    Segmentation service for smart mask generation.
    Uses T4 GPU for segmentation model inference (FastSAM, BiRefNet).
    Timeout 30 phút để đảm bảo không bị treo khi cold boot và model loading trong lúc bảo vệ đồ án.
    """
    # Class attributes (initialized in @modal.enter)
    container_start_time: float | None = None
    is_ready: bool = False
    segmentation_model: object | None = None

    @modal.enter()
    def warmup(self):
        """
        Container startup hook: pre-load segmentation model để giảm latency.
        """
        self.container_start_time = time.time()
        self.is_ready = False
        self.segmentation_model = None
        print("🚀 [SegmentationService] Container starting up...")
        
        import torch
        print(f"🔧 [SegmentationService] CUDA available: {torch.cuda.is_available()}")
        
        # Pre-load FastSAM model (default segmentation model)
        # Strategy: Load trực tiếp từ Volume (giống Qwen) để giảm cold-start
        try:
            from ultralytics import FastSAM  # type: ignore[attr-defined]
            
            segmentation_model_path = "FastSAM-s.pt"  # Default: download from GitHub (lighter version)
            
            # Load trực tiếp từ Volume (không copy để giảm cold-start, giống Qwen)
            vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
            if os.path.exists(vol_segmentation_file):
                segmentation_model_path = vol_segmentation_file
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
        """
        Segmentation service endpoint for smart mask generation.
        Exposed at: https://<username>--segmentation.modal.run
        """
        app = create_fastapi_app(include_segmentation=True)
        print(f"🚀 [SegmentationService] FastAPI app created with {len(app.routes)} routes")
        # Log all routes for debugging
        for route in app.routes:
            route_path = getattr(route, 'path', None)
            route_methods = getattr(route, 'methods', None)
            if route_path and route_methods:
                print(f"   📍 Route: {list(route_methods)} {route_path}")
        return app


# 5.5 IMAGE UTILS SERVICE (CPU-only) - Image utilities only
# ==========================================
# Lightweight image processing operations (extract object, resize, encode, etc.).
@app.cls(
    image=imageutils_image,
    cpu=1,  # Minimal CPU (cheapest)
    timeout=1800,  # 30 minutes - đủ thời gian cho cold boot khi bảo vệ đồ án
    min_containers=0,  # Cold start - instant startup
    scaledown_window=60,  # Scale down quickly (1 minute)
)
@modal.concurrent(max_inputs=50)  # High concurrency (CPU-only, cheap)
class ImageUtilsService:
    """
    Image utilities service for resize, crop, encode, etc.
    CPU-only operations - no GPU needed.
    Timeout 30 phút để đảm bảo không bị treo khi cold boot trong lúc bảo vệ đồ án.
    """
    container_start_time: float | None = None
    is_ready: bool = False

    @modal.enter()
    def prepare(self):
        """Container startup hook."""
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [ImageUtilsService] Container starting up...")
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [ImageUtilsService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """Container shutdown hook."""
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [ImageUtilsService] Container shutting down after {uptime:.2f}s of uptime")
        self.is_ready = False
        print("✅ [ImageUtilsService] Cleanup complete")

    @modal.asgi_app(label="image-utils")
    def serve(self):
        """
        Image utils service endpoint for image processing operations.
        Exposed at: https://<username>--artmancer-image-utils.modal.run
        """
        return create_fastapi_app(include_image_utils=True)


# ==========================================
# 6. WORKER FUNCTIONS
# ==========================================
# Background functions that run on-demand (scale-to-zero).

# 6.1 QWEN WORKER (Scale-to-Zero)
# ==========================================
# Executes image generation on H200 GPU. Called by Job Manager for async jobs.
# Updates job_state_dictio with progress and results.

@app.function(
    image=heavy_image,
    gpu="H200",
    timeout=1800,  # 30 minutes - đủ thời gian cho cold boot và model loading khi bảo vệ đồ án
    volumes={VOL_MOUNT_PATH: volume},
    max_containers=1,  # Single user - only need 1 container at a time
)
def qwen_worker(task_id: str, payload: dict):
    """
    H200 worker function for async image generation.
    
    This function is called by Job Manager to execute generation on H200 GPU.
    It updates job_state_dictio with progress and final results.
    
    Workflow:
    1. Updates job state to "initializing_h200"
    2. Checks if pipeline is already loaded (container reuse optimization)
    3. Loads Qwen pipeline from Volume only if needed (avoids reload on warm containers)
    4. Updates job state to "processing" with progress callbacks
    5. Runs generation with progress tracking
    6. Updates job state to "done" with result (base64 image)
    
    Args:
        task_id: Unique task identifier (used as key in job_state_dictio)
        payload: Generation request payload (same as GenerationRequest schema)
    
    Returns:
        dict with status and result (for logging/debugging)
    """
    try:
        # Update state: initializing_h200
        # Preserve created_at if exists, otherwise set to current time
        existing_state = job_state_dictio.get(task_id, {})
        job_state_dictio[task_id] = {
            "status": "initializing_h200",
            "progress": 0.1,
            "error": None,
            "result": None,
            "debug_info": None,
            "created_at": existing_state.get("created_at", time.time()),  # Preserve or set timestamp
        }
        
        print(f"🚀 [H200 Worker] Task {task_id}: Initializing H200 container...")
        
        # Setup environment: load model trực tiếp từ Volume (không copy)
        os.environ.setdefault("TASK_CHECKPOINTS_PATH", VOL_TASKS_PATH)
        
        if Path(VOL_MODEL_PATH).exists():
            os.environ["MODEL_PATH"] = VOL_MODEL_PATH
            print(f"🚀 MODEL_PATH set to {VOL_MODEL_PATH} (loaded directly from Volume)")
        else:
            print(
                f"⚠️ Base model snapshot not found at {VOL_MODEL_PATH}. "
                f"qwen_loader will load from Hugging Face ({BASE_MODEL_ID})."
            )
        
        # Import here to avoid loading on module import
        from app.models.schemas import GenerationRequest
        from app.services.generation_service import GenerationService
        
        # DOUBLE CHECK: Check if pipeline is already loaded and can be reused
        # This prevents reloading model on warm/hot containers (container reuse optimization)
        # Only clear cache if we need to reload (different task_type or flags)
        # This allows container reuse: if same task_type, keep model in memory
        pipeline_already_loaded = False
        try:
            from app.core.qwen_loader import (
                get_qwen_pipeline, 
                is_qwen_pipeline_loaded, 
                _flush_memory,
                _pipeline as global_pipeline  # Access global pipeline state
            )
            from app.models.schemas import GenerationRequest
            
            # Parse payload to get task_type
            request = GenerationRequest(**payload)
            task_type = request.task_type or "insertion"
            
            # DOUBLE CHECK 1: Check if base pipeline is loaded
            if global_pipeline is not None:
                # DOUBLE CHECK 2: Check if adapter for this task is loaded
                if is_qwen_pipeline_loaded(task_type):
                    existing_pipeline = get_qwen_pipeline(task_type)
                    if existing_pipeline is not None:
                        # Verify pipeline is actually usable (not corrupted)
                        try:
                            # Quick sanity check: verify pipeline has required attributes
                            if hasattr(existing_pipeline, 'unet') and existing_pipeline.unet is not None:
                                print(f"♻️ [H200 Worker] Task {task_id}: Pipeline already loaded for task_type={task_type}, REUSING (container warm/hot)")
                                pipeline_already_loaded = True
                                # Just flush GPU memory cache (not the model) to free unused memory
                                _flush_memory()
                            else:
                                print(f"⚠️ [H200 Worker] Task {task_id}: Pipeline exists but appears corrupted, will reload...")
                                pipeline_already_loaded = False
                        except Exception as e:
                            print(f"⚠️ [H200 Worker] Task {task_id}: Pipeline check failed: {e}, will reload...")
                            pipeline_already_loaded = False
                    else:
                        print(f"🔄 [H200 Worker] Task {task_id}: Pipeline state inconsistent, will reload...")
                        pipeline_already_loaded = False
                else:
                    print(f"🔄 [H200 Worker] Task {task_id}: Base pipeline loaded but adapter for {task_type} not loaded, will switch adapter...")
                    # Base model is loaded, just need to switch adapter (no reload needed)
                    pipeline_already_loaded = True  # Base model exists, adapter switch is fast
                    _flush_memory()
            else:
                print(f"🧹 [H200 Worker] Task {task_id}: No pipeline loaded (container cold), will load from scratch...")
                pipeline_already_loaded = False
                # Ensure clean state before loading
                from app.core.qwen_loader import clear_qwen_cache
                clear_qwen_cache()
                time.sleep(1)
                _flush_memory()
                print(f"✅ [H200 Worker] Task {task_id}: Cache cleared and memory flushed, ready for fresh load")
        except Exception as e:
            print(f"⚠️ [H200 Worker] Task {task_id}: Failed to check/reuse pipeline: {e}")
            # Fallback: clear cache to ensure clean state
            pipeline_already_loaded = False
            try:
                from app.core.qwen_loader import clear_qwen_cache, _flush_memory
                clear_qwen_cache()
                time.sleep(1)
                _flush_memory()
            except Exception:
                pass
        
        if pipeline_already_loaded:
            print(f"✅ [H200 Worker] Task {task_id}: Pipeline ready, skipping load (container reuse)")
        else:
            print(f"🔄 [H200 Worker] Task {task_id}: Pipeline will be loaded by GenerationService...")
        
        # Update state: loading_pipeline (pipeline is being loaded)
        job_state_dictio[task_id] = {
            "status": "loading_pipeline",
            "progress": 0.2,
            "error": None,
            "result": None,
            "debug_info": None,
            "created_at": job_state_dictio.get(task_id, {}).get("created_at", time.time()),  # Preserve timestamp
        }
        
        # Convert payload to GenerationRequest
        request = GenerationRequest(**payload)
        
        # Create callback to update loading progress
        def on_loading_progress(message: str, progress: float):
            """Callback to update loading progress during pipeline loading."""
            current_state = job_state_dictio.get(task_id, {})
            # Map progress (0.0-1.0) to overall progress (0.2-0.3)
            # Pipeline loading is 20-30% of total progress
            overall_progress = 0.2 + (progress * 0.1)
            job_state_dictio[task_id] = {
                **current_state,
                "status": "loading_pipeline",
                "progress": overall_progress,
                "loading_message": message,  # Store message for frontend
            }
            print(f"📊 [H200 Worker] Task {task_id}: Loading progress - {message} ({progress*100:.1f}%)")
        
        # Create callback to update status when pipeline is loaded
        def on_pipeline_loaded():
            """Callback to update status when pipeline finishes loading."""
            current_state = job_state_dictio.get(task_id, {})
            print(f"📢 [H200 Worker] Task {task_id}: on_pipeline_loaded callback called, updating status to processing...")
            job_state_dictio[task_id] = {
                **current_state,
                "status": "processing",
                "progress": 0.3,
                "current_step": None,  # Reset step tracking
                "total_steps": None,   # Will be set when inference starts
                "loading_message": None,  # Clear loading message
            }
            print(f"✅ [H200 Worker] Task {task_id}: Status updated to 'processing', pipeline ready for generation")
        
        # Create progress callback to update job state with cancellation check
        def progress_callback(step: int, timestep: int, total_steps: int):
            """Callback to update progress during generation with cancellation check."""
            try:
                # Check cancellation flag in job_state_dictio
                current_state = job_state_dictio.get(task_id, {})
                if current_state.get("cancelled", False):
                    print(f"⚠️ [H200 Worker] Task {task_id} cancelled at step {step}/{total_steps}")
                    # Update state to cancelled
                    job_state_dictio[task_id] = {
                        **current_state,
                        "status": "cancelled",
                        "progress": 0.3 + (step / total_steps) * 0.65,
                        "error": "Generation cancelled by user",
                    }
                    raise ValueError(f"Task {task_id} was cancelled")
                
                # Calculate progress: 0.3 (pipeline loaded) to 0.95 (before final processing)
                # Pipeline loading is 0-0.3, generation is 0.3-0.95, final processing is 0.95-1.0
                generation_progress = 0.3 + (step / total_steps) * 0.65
                job_state_dictio[task_id] = {
                    **current_state,
                    "status": "processing",
                    "progress": generation_progress,
                    "current_step": step,
                    "total_steps": total_steps,
                }
            except ValueError as e:
                # Re-raise cancellation error
                if "cancelled" in str(e).lower():
                    raise
                print(f"⚠️ [H200 Worker] Failed to update progress: {e}")
            except Exception as e:
                print(f"⚠️ [H200 Worker] Failed to update progress: {e}")
        
        # Run generation with progress callback and pipeline loaded callback
        print(f"🎨 [H200 Worker] Task {task_id}: Running generation...")
        generation_service = GenerationService()
        # Pass progress callback, pipeline loaded callback, and loading progress callback to generation service
        result = generation_service.generate(
            request, 
            progress_callback=progress_callback, 
            on_pipeline_loaded=on_pipeline_loaded,
            on_loading_progress=on_loading_progress
        )
        
        # Convert result image to base64
        if result.get("image"):
            # Image is already base64, just use it
            image_base64 = result["image"]
        else:
            raise ValueError("Generation result missing image")
        
        # Update state: done
        existing_state = job_state_dictio.get(task_id, {})
        job_state_dictio[task_id] = {
            "status": "done",
            "progress": 1.0,
            "error": None,
            "result": image_base64,
            "debug_info": result.get("debug_info"),
            "created_at": existing_state.get("created_at", time.time()),  # Preserve timestamp
        }
        
        print(f"✅ [H200 Worker] Task {task_id}: Generation completed")
        
        # Keep pipeline in memory for container reuse
        # Only flush GPU memory cache (not the model) to free unused memory
        # This allows subsequent inferences to reuse the loaded model without reloading
        try:
            from app.core.qwen_loader import _flush_memory, _pipeline as global_pipeline
            if global_pipeline is not None:
                print(f"🧹 [H200 Worker] Task {task_id}: Flushing GPU memory cache (keeping model in memory for container reuse)...")
                _flush_memory()
                print(f"✅ [H200 Worker] Task {task_id}: GPU memory flushed (model kept in memory for reuse)")
            else:
                print(f"⚠️ [H200 Worker] Task {task_id}: Pipeline not in memory, nothing to flush")
        except Exception as e:
            print(f"⚠️ [H200 Worker] Task {task_id}: Memory flush warning: {e}")
        
        return {
            "status": "done",
            "task_id": task_id,
            "result": image_base64,
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [H200 Worker] Task {task_id}: Error - {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Update state: error
        existing_state = job_state_dictio.get(task_id, {})
        job_state_dictio[task_id] = {
            "status": "error",
            "progress": 0.0,
            "error": error_msg,
            "result": None,
            "debug_info": None,
            "created_at": existing_state.get("created_at", time.time()),  # Preserve timestamp
        }
        
        # Cleanup on error too (but don't clear if container might be reused)
        try:
            from app.core.qwen_loader import clear_qwen_cache, _pipeline as global_pipeline
            # Only clear cache if error is critical (OOM, corruption, etc.)
            # Don't clear on user cancellation or validation errors
            if "out of memory" in error_msg.lower() or "corrupt" in error_msg.lower():
                print(f"🧹 [H200 Worker] Task {task_id}: Cleaning up pipeline cache after critical error...")
                clear_qwen_cache()
            else:
                print(f"⚠️ [H200 Worker] Task {task_id}: Non-critical error, keeping pipeline in memory for reuse")
        except Exception:
            pass  # Ignore cleanup errors
        
        return {
            "status": "error",
            "task_id": task_id,
            "error": error_msg,
        }


# ==========================================
# 7. HELPER FUNCTIONS
# ==========================================
# Utility functions for Volume management and setup.

# 7.1 Remove specific checkpoint file from Volume
# ==========================================
@app.function(
    volumes={VOL_MOUNT_PATH: volume},
    timeout=60,
)
def remove_checkpoint_file(filename: str):
    """
    Remove a specific checkpoint file from Modal Volume.
    
    Args:
        filename: Name of the checkpoint file to remove (e.g., "insertion_cp.safetensors")
    
    Returns:
        True if file was removed, False otherwise
    """
    file_path = f"{VOL_MOUNT_PATH}/{filename}"
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            volume.commit()
            print(f"  🗑️ Removed: {filename}")
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to remove {file_path}: {e}")
            return False
    else:
        print(f"  ℹ️ File not found: {filename}")
        return False


# 7.2 Clear all checkpoints from Volume
# ==========================================
@app.function(
    volumes={VOL_MOUNT_PATH: volume},
    timeout=60,
)
def clear_checkpoints():
    """
    Remove all .safetensors files from Modal Volume.
    Uses remove_checkpoint_file internally to avoid code duplication.
    
    Usage:
        modal run modal_app.py::clear_checkpoints
    """
    import glob
    
    print(f"🗑️ Clearing all checkpoints from Volume '{VOLUME_NAME}' ...")
    
    # Find all safetensors files
    pattern = f"{VOL_MOUNT_PATH}/*.safetensors"
    files = glob.glob(pattern)
    
    if not files:
        print("  No checkpoint files found to remove.")
        return []
    
    removed = []
    for file_path in files:
        filename = os.path.basename(file_path)
        # Use remove_checkpoint_file logic but don't commit each time (commit once at end)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  🗑️ Removed: {filename}")
                removed.append(filename)
            except Exception as e:
                print(f"  ⚠️ Failed to remove {file_path}: {e}")
    
    # Commit changes to volume once at the end (more efficient than committing per file)
    if removed:
        volume.commit()
    
    print(f"✅ Removed {len(removed)} checkpoint file(s).")
    return removed


# 7.3 Upload LoRA checkpoints from local to Volume
# ==========================================
# Local entrypoint: run from your local machine to upload checkpoints.
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
        modal run modal_app.py::upload_checkpoints --force  # Overwrite existing files
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
    
    # Check which files are present
    found_files = {f.name for f in checkpoint_files}
    missing_files = set(expected_files.keys()) - found_files
    extra_files = found_files - set(expected_files.keys())
    
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
    
    if force:
        print("🔄 Force mode: clearing existing checkpoints first...")
        # Call remote function to clear existing checkpoints
        clear_checkpoints.remote()
        print("✅ Existing checkpoints cleared.")
    else:
        # Check and remove individual files that will be uploaded (to avoid FileExistsError)
        print("🔍 Checking for existing files...")
        for checkpoint_file in checkpoint_files:
            filename = checkpoint_file.name
            print(f"  Checking: {filename}")
            removed = remove_checkpoint_file.remote(filename)
            if removed:
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
                # Try to remove and upload again
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


# 7.4 Setup all models in Volume
# ==========================================
# Downloads Qwen base model, FastSAM model, and BiRefNet model to Volume for faster cold starts.
# Optional but recommended: reduces cold-start from 20-60s to 1-3s.
@app.function(
    image=heavy_image,
    volumes={VOL_MOUNT_PATH: volume},
    timeout=3600,  # 1 hour for large model downloads
)
def setup_volume():
    """
    Download Qwen base model, FastSAM model, and BiRefNet model into Modal Volume for faster cold starts.

    This is optional but recommended. If not run, the models will be downloaded
    on first use (slower cold-start).

    Usage:
        modal run modal_app.py::setup_volume
    """
    from huggingface_hub import snapshot_download
    import urllib.request

    print("🚀 Starting setup_volume: downloading models to Volume...")
    
    # 1. Download Qwen base model
    print("\n📦 [1/3] Downloading Qwen base model...")
    os.makedirs(VOL_MODEL_PATH, exist_ok=True)

    try:
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            local_dir=VOL_MODEL_PATH,
            ignore_patterns=["*.msgpack", "*.bin", ".git*"],
        )
        print(f"✅ Base model downloaded to {VOL_MODEL_PATH}")
    except Exception as exc:
        print(f"❌ Failed to download base model: {exc}")
        return

    print("📂 Current contents of base model directory:")
    try:
        print(os.listdir(VOL_MODEL_PATH))
    except Exception:
        pass

    # 2. Download FastSAM model
    print("\n📦 [2/3] Downloading FastSAM-s.pt...")
    os.makedirs(VOL_SEGMENTATION_PATH, exist_ok=True)
    
    fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"
    vol_segmentation_file = f"{VOL_SEGMENTATION_PATH}/FastSAM-s.pt"
    
    try:
        print(f"📥 Downloading FastSAM-s.pt from {fastsam_url}...")
        urllib.request.urlretrieve(fastsam_url, vol_segmentation_file)
        
        # Get file size
        file_size = os.path.getsize(vol_segmentation_file)
        print(f"✅ FastSAM-s.pt downloaded to {vol_segmentation_file} ({file_size / 1024 / 1024:.1f} MB)")
        
        print("✅ FastSAM-s.pt cached in Volume. SegmentationService will use this for faster cold starts.")
    except Exception as exc:
        print(f"❌ Failed to download FastSAM-s.pt: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Download BiRefNet model
    print("\n📦 [3/3] Downloading BiRefNet model...")
    os.makedirs(VOL_BIREFNET_PATH, exist_ok=True)
    
    birefnet_repo_id = "zhengpeng7/BiRefNet"
    vol_birefnet_dir = VOL_BIREFNET_PATH
    
    try:
        print(f"📥 Downloading BiRefNet from {birefnet_repo_id}...")
        snapshot_download(
            repo_id=birefnet_repo_id,
            local_dir=vol_birefnet_dir,
            ignore_patterns=["*.msgpack", ".git*"],
        )
        
        # Get directory size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(vol_birefnet_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        print(f"✅ BiRefNet model downloaded to {vol_birefnet_dir} ({total_size / 1024 / 1024:.1f} MB)")
        
        print("✅ BiRefNet cached in Volume. Will be loaded directly from Volume for faster cold starts.")
    except Exception as exc:
        print(f"❌ Failed to download BiRefNet model: {exc}")
        import traceback
        traceback.print_exc()
        return
    
    # Commit all changes to Volume
    print("\n💾 Committing changes to Volume...")
    volume.commit()
    print("✅ All models cached in Volume successfully!")


# ==========================================
# 8. DEPLOYMENT & USAGE
# ==========================================

# Deploy all services:
#   modal deploy modal_app.py
#
# This deploys 4 public endpoints:
# - API Gateway: https://<username>--api-gateway.modal.run
#   - Single entry point, routes all requests to backend services
# - Job Manager: https://<username>--job-manager.modal.run
#   - Handles async generation job coordination, dispatches to H200 workers
# - Segmentation Service: https://<username>--segmentation.modal.run
#   - Smart mask segmentation (T4 GPU)
# - Image Utils Service: https://<username>--image-utils.modal.run
#   - Image processing utilities (CPU)
#
# Note: H200 worker (qwen_worker) is not a public endpoint - it's spawned internally by Job Manager.
#
# All services support cold boot (scale to zero when idle) for cost optimization.
#
# For local testing with hot-reload:
#   modal serve modal_app.py
#
# Setup Volume (recommended before first deployment):
#   modal run modal_app.py::setup_volume          # Download Qwen base model, FastSAM model, and BiRefNet model
#   modal run modal_app.py::upload_checkpoints    # Upload LoRA checkpoints



