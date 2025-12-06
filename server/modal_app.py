"""
Modal app for ArtMancer backend.

This file defines all Modal services, images, and helper functions for deployment.

Architecture:
- Heavy service (A100-80GB): Qwen generation, optimized for large model + LoRA.
- Light services (T4/CPU): smart-mask, image-utils, debug, system.
- API Gateway (CPU): Routes requests to appropriate services.
- Job Manager (CPU): Coordinates async generation jobs.

Features:
- Modal Volume for checkpoints + base model snapshot.
- Load model trực tiếp từ Volume (không copy) để giảm cold-start từ 20-60s xuống 1-3s.
- Cold boot enabled for cost optimization (scales to zero when idle).

Table of Contents:
1. CONFIGURATION - Constants and paths
2. INFRASTRUCTURE - Modal app, volume, dictio setup
3. IMAGE DEFINITIONS - Docker images for different services
   - cpu_image: Job Manager & API Gateway
   - heavy_image: Qwen generation (A100)
   - fastsam_image: FastSAM segmentation (T4)
   - imageutils_image: Image utilities (CPU)
4. create_fastapi_app() - Factory function for FastAPI apps
5. SERVICE CLASSES - Modal service definitions
   - JobManagerService: Async job coordination
   - APIGatewayService: Request routing
   - QwenService: Image generation
   - FastSAMService: Smart mask segmentation
   - ImageUtilsService: Image processing utilities
6. WORKER FUNCTIONS - Background tasks
   - run_generation(): A100 worker for async generation
7. HELPER FUNCTIONS - Volume management utilities
   - remove_checkpoint_file(): Remove single checkpoint
   - clear_checkpoints(): Remove all checkpoints
   - upload_checkpoints(): Upload checkpoints from local
   - setup_volume(): Download Qwen base model
   - setup_fastsam_volume(): Download FastSAM model
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
VOL_FASTSAM_PATH = f"{VOL_MOUNT_PATH}/fastsam"  # FastSAM-s.pt cache (load trực tiếp từ Volume, không copy)

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

# 3.2 HEAVY IMAGE (A100) - Optimized for GPU inference
# ==========================================
# Includes: diffusers, transformers, torch, peft for Qwen image generation
heavy_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Install diffusers from GitHub (pinned version) for QwenImageEditPlusPipeline support
        "uv pip install --system --no-cache-dir git+https://github.com/huggingface/diffusers.git@6290fdfda40610ce7b99920146853614ba529c6e",
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

# 3.3 FASTSAM IMAGE (T4) - FastSAM segmentation only
# ==========================================
# Includes: ultralytics, torch (CUDA), scikit-image for mask processing
fastsam_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        # Use CUDA torch for T4 GPU acceleration (FastSAM)
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
    include_generation: bool = False,
    job_manager_mode: bool = False,
    include_fastsam: bool = False,
    include_image_utils: bool = False,
) -> FastAPI:
    """
    Create FastAPI app instance with appropriate routers.
    
    This function is used by all Modal services to create their FastAPI apps.
    Each service includes only the routers it needs.
    
    Args:
        include_generation: If True, includes generation_sync router (Qwen Service).
        job_manager_mode: If True, includes generation_async router (Job Manager).
        include_fastsam: If True, includes smart_mask router (FastSAM Service).
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
    elif include_generation:
        # Qwen Service: only sync generation (requires diffusers, transformers, peft)
        from app.api.endpoints import generation_sync
        fastapi_app.include_router(generation_sync.router)

    if include_fastsam:
        # FastSAM Service: smart mask segmentation (T4 GPU)
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
# to A100 workers. Manages job state in job_state_dictio.
@app.cls(
    image=cpu_image,
    cpu=1,  # Minimal CPU (cheapest option - only coordination, no heavy processing)
    min_containers=0,  # Allow cold boot (scale to zero when idle)
    timeout=60,  # Short timeout (just coordination, no heavy processing)
    scaledown_window=1200,  # Scale down after 20 minutes of inactivity
)
class JobManagerService:
    """
    CPU-based job manager for coordinating A100 inference jobs.
    Cold boot enabled - container starts on demand and scales down after 20 minutes of inactivity.
    Keeps container warm while frontend is active, saves costs when idle.
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
    timeout=1800,  # 30 minutes - needs to be longer than generation service timeout (20 min)
    scaledown_window=1200,  # Scale down after 20 minutes of inactivity
)
class APIGatewayService:
    """
    API Gateway - Single entry point for all requests.
    Routes requests to appropriate services (Generation, Segmentation, Image Utils, Job Manager).
    Cold boot enabled - container starts on demand and scales down after 20 minutes of inactivity.
    Keeps container warm while frontend is active, saves costs when idle.
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
        generation_url = os.environ.setdefault(
            "GENERATION_SERVICE_URL",
            "https://nxan2911--qwen.modal.run"
        )
        segmentation_url = os.environ.setdefault(
            "SEGMENTATION_SERVICE_URL",
            "https://nxan2911--fastsam.modal.run"
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
        print(f"   - Generation Service: {generation_url}")
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


# 5.3 QWEN SERVICE (A100) - Image generation only
# ==========================================
# Synchronous image generation service. Loads Qwen model and LoRA checkpoints.
# Note: For async generation, use Job Manager + A100 worker instead.
@app.cls(
    image=heavy_image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=1800,  # 30 phút phòng model chạy lâu
    scaledown_window=120,  # Giữ container 2 phút sau request cuối (giảm cold start)
)
@modal.concurrent(max_inputs=3)  # Input concurrency per container
class QwenService:
    # Class attributes (initialized in @modal.enter)
    container_start_time: float | None = None
    is_ready: bool = False

    @modal.enter()
    def prepare_env(self):
        """
        Container startup hook: chạy khi container khởi động.
        - Load model trực tiếp từ Volume (không copy để giảm cold-start)
        - Thiết lập MODEL_PATH cho qwen_loader
        - Track container lifecycle
        """
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [QwenService] Container starting up...")
        
        # Luôn set TASK_CHECKPOINTS_PATH để code khác dùng nếu cần
        os.environ.setdefault("TASK_CHECKPOINTS_PATH", VOL_TASKS_PATH)

        # Load model trực tiếp từ Volume (không copy để giảm cold-start từ 20-60s xuống 1-3s)
        if Path(VOL_MODEL_PATH).exists():
            os.environ["MODEL_PATH"] = VOL_MODEL_PATH
            print(f"🚀 MODEL_PATH set to {VOL_MODEL_PATH} (loaded directly from Volume)")
        else:
            # Chưa setup Volume: qwen_loader sẽ dùng HF ID
            print(
                f"⚠️ Base model snapshot not found at {VOL_MODEL_PATH}. "
                f"qwen_loader will load from Hugging Face ({BASE_MODEL_ID})."
            )
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [QwenService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """
        Container shutdown hook: chạy khi container tắt (sau idle timeout hoặc preemption).
        - Log container lifecycle
        - Cleanup resources nếu cần
        """
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [QwenService] Container shutting down after {uptime:.2f}s of uptime")
        else:
            print("🛑 [QwenService] Container shutting down")
        
        self.is_ready = False
        print("✅ [QwenService] Cleanup complete")

    @modal.asgi_app(label="qwen")
    def serve(self):
        """
        Qwen service endpoint for image generation.
        Exposed at: https://<username>--artmancer-qwen.modal.run
        Note: This is kept for backward compatibility. New async flow uses CPU job manager.
        """
        return create_fastapi_app(include_generation=True)



# 5.4 FASTSAM SERVICE (T4) - Smart mask segmentation only
# ==========================================
# FastSAM model for intelligent mask generation from bbox/points/strokes.
@app.cls(
    image=fastsam_image,
    gpu="T4",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=300,  # 5 phút để đủ thời gian download models (FastSAM)
    min_containers=0,  # Cold start - only run when needed
    scaledown_window=120,  # Scale down quickly (2 minutes)
)
@modal.concurrent(max_inputs=20)  # Allow multiple concurrent requests
class FastSAMService:
    """
    FastSAM service for smart mask segmentation.
    Uses T4 GPU for FastSAM model inference.
    """
    # Class attributes (initialized in @modal.enter)
    container_start_time: float | None = None
    is_ready: bool = False
    fastsam_model: object | None = None

    @modal.enter()
    def warmup(self):
        """
        Container startup hook: pre-load FastSAM model để giảm latency.
        """
        self.container_start_time = time.time()
        self.is_ready = False
        self.fastsam_model = None
        print("🚀 [FastSAMService] Container starting up...")
        
        import torch
        print(f"🔧 [FastSAMService] CUDA available: {torch.cuda.is_available()}")
        
        # Pre-load FastSAM model
        # Strategy: Load trực tiếp từ Volume (giống Qwen) để giảm cold-start
        try:
            from ultralytics import FastSAM  # type: ignore[attr-defined]
            
            fastsam_model_path = "FastSAM-s.pt"  # Default: download from GitHub (lighter version)
            
            # Load trực tiếp từ Volume (không copy để giảm cold-start, giống Qwen)
            vol_fastsam_file = f"{VOL_FASTSAM_PATH}/FastSAM-s.pt"
            if os.path.exists(vol_fastsam_file):
                fastsam_model_path = vol_fastsam_file
                print(f"🚀 FastSAM-s.pt loaded directly from Volume: {vol_fastsam_file}")
            else:
                print("⚠️ FastSAM-s.pt not in Volume, will download from GitHub (run setup_fastsam_volume to cache)")
            
            print(f"🔄 Loading FastSAM-s from: {fastsam_model_path}")
            self.fastsam_model = FastSAM(fastsam_model_path)
            print("✅ FastSAM-s model loaded")
        except Exception as e:
            print(f"⚠️ FastSAM warmup failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [FastSAMService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """Container shutdown hook."""
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [FastSAMService] Container shutting down after {uptime:.2f}s of uptime")
        else:
            print("🛑 [FastSAMService] Container shutting down")
        
        self.fastsam_model = None
        self.is_ready = False
        print("✅ [FastSAMService] Cleanup complete")

    @modal.asgi_app(label="fastsam")
    def serve(self):
        """
        FastSAM service endpoint for smart mask segmentation.
        Exposed at: https://<username>--fastsam.modal.run
        """
        app = create_fastapi_app(include_fastsam=True)
        print(f"🚀 [FastSAMService] FastAPI app created with {len(app.routes)} routes")
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
    timeout=60,  # Short timeout (lightweight operations)
    min_containers=0,  # Cold start - instant startup
    scaledown_window=60,  # Scale down quickly (1 minute)
)
@modal.concurrent(max_inputs=50)  # High concurrency (CPU-only, cheap)
class ImageUtilsService:
    """
    Image utilities service for resize, crop, encode, etc.
    CPU-only operations - no GPU needed.
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

# 6.1 A100 WORKER FUNCTION (Scale-to-Zero)
# ==========================================
# Executes image generation on A100 GPU. Called by Job Manager for async jobs.
# Updates job_state_dictio with progress and results.

@app.function(
    image=heavy_image,
    gpu="A100-80GB",
    timeout=1800,  # 30 minutes
    volumes={VOL_MOUNT_PATH: volume},
    max_containers=1,  # Don't spawn multiple A100s
)
def run_generation(task_id: str, payload: dict):
    """
    A100 worker function for async image generation.
    
    This function is called by Job Manager to execute generation on A100 GPU.
    It updates job_state_dictio with progress and final results.
    
    Workflow:
    1. Updates job state to "initializing_a100"
    2. Loads Qwen pipeline from Volume
    3. Updates job state to "processing" with progress callbacks
    4. Runs generation with progress tracking
    5. Updates job state to "done" with result (base64 image)
    
    Args:
        task_id: Unique task identifier (used as key in job_state_dictio)
        payload: Generation request payload (same as GenerationRequest schema)
    
    Returns:
        dict with status and result (for logging/debugging)
    """
    try:
        # Update state: initializing_a100
        # Preserve created_at if exists, otherwise set to current time
        existing_state = job_state_dictio.get(task_id, {})
        job_state_dictio[task_id] = {
            "status": "initializing_a100",
            "progress": 0.1,
            "error": None,
            "result": None,
            "debug_info": None,
            "created_at": existing_state.get("created_at", time.time()),  # Preserve or set timestamp
        }
        
        print(f"🚀 [A100 Worker] Task {task_id}: Initializing A100 container...")
        
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
        
        print(f"🔄 [A100 Worker] Task {task_id}: Pipeline will be loaded by GenerationService...")
        
        # Update state: processing
        job_state_dictio[task_id] = {
            "status": "processing",
            "progress": 0.3,
            "error": None,
            "result": None,
            "debug_info": None,
            "created_at": job_state_dictio.get(task_id, {}).get("created_at", time.time()),  # Preserve timestamp
        }
        
        # Convert payload to GenerationRequest
        request = GenerationRequest(**payload)
        
        # Create progress callback to update job state with cancellation check
        def progress_callback(step: int, timestep: int, total_steps: int):
            """Callback to update progress during generation with cancellation check."""
            try:
                # Check cancellation flag in job_state_dictio
                current_state = job_state_dictio.get(task_id, {})
                if current_state.get("cancelled", False):
                    print(f"⚠️ [A100 Worker] Task {task_id} cancelled at step {step}/{total_steps}")
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
                print(f"⚠️ [A100 Worker] Failed to update progress: {e}")
            except Exception as e:
                print(f"⚠️ [A100 Worker] Failed to update progress: {e}")
        
        # Run generation with progress callback
        print(f"🎨 [A100 Worker] Task {task_id}: Running generation...")
        generation_service = GenerationService()
        # Pass progress callback to generation service
        result = generation_service.generate(request, progress_callback=progress_callback)
        
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
        
        print(f"✅ [A100 Worker] Task {task_id}: Generation completed")
        
        return {
            "status": "done",
            "task_id": task_id,
            "result": image_base64,
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ [A100 Worker] Task {task_id}: Error - {error_msg}")
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


# 7.4 Setup Qwen base model snapshot in Volume
# ==========================================
# Downloads Qwen base model from Hugging Face to Volume for faster cold starts.
# Optional but recommended: reduces cold-start from 20-60s to 1-3s.
@app.function(
    image=heavy_image,
    volumes={VOL_MOUNT_PATH: volume},
    timeout=3600,  # 1 hour for large model download
)
def setup_volume():
    """
    Download Qwen base model into Modal Volume for faster cold starts.

    This is optional but recommended. If not run, the model will be downloaded
    on first use (slower cold-start).

    Usage:
        modal run modal_app.py::setup_volume
    """
    from huggingface_hub import snapshot_download

    print("🚀 Starting setup_volume: downloading base model to Volume...")
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


# 7.5 Setup FastSAM model in Volume
# ==========================================
# Downloads FastSAM-s.pt model to Volume for faster FastSAM service cold starts.
@app.function(
    image=fastsam_image,
    volumes={VOL_MOUNT_PATH: volume},
    timeout=600,  # 10 minutes for model download
)
def setup_fastsam_volume():
    """
    Download FastSAM-s.pt model into Modal Volume for faster FastSAM service cold starts.
    
    Optional but recommended for faster FastSAM service startup.

    Usage:
        modal run modal_app.py::setup_fastsam_volume
    """
    import urllib.request
    
    print("🚀 Starting setup_fastsam_volume: downloading FastSAM-s.pt to Volume...")
    os.makedirs(VOL_FASTSAM_PATH, exist_ok=True)
    
    fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"
    vol_fastsam_file = f"{VOL_FASTSAM_PATH}/FastSAM-s.pt"
    
    try:
        print(f"📥 Downloading FastSAM-s.pt from {fastsam_url}...")
        urllib.request.urlretrieve(fastsam_url, vol_fastsam_file)
        
        # Get file size
        file_size = os.path.getsize(vol_fastsam_file)
        print(f"✅ FastSAM-s.pt downloaded to {vol_fastsam_file} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Commit changes to Volume
        volume.commit()
        
        print("✅ FastSAM-s.pt cached in Volume. FastSAMService will use this for faster cold starts.")
    except Exception as exc:
        print(f"❌ Failed to download FastSAM-s.pt: {exc}")
        import traceback
        traceback.print_exc()
        return


# ==========================================
# 8. DEPLOYMENT & USAGE
# ==========================================

# Deploy all services:
#   modal deploy modal_app.py
#
# This deploys all services defined above:
# - API Gateway: https://<username>--api-gateway.modal.run
#   - Routes all requests to appropriate services
# - Job Manager: https://<username>--job-manager.modal.run
#   - Handles async generation job coordination
# - Qwen Service: https://<username>--qwen.modal.run
#   - Synchronous image generation (A100 GPU)
# - FastSAM Service: https://<username>--fastsam.modal.run
#   - Smart mask segmentation (T4 GPU)
# - Image Utils Service: https://<username>--image-utils.modal.run
#   - Image processing utilities (CPU)
#
# All services support cold boot (scale to zero when idle) for cost optimization.
#
# For local testing with hot-reload:
#   modal serve modal_app.py
#
# Setup Volume (recommended before first deployment):
#   modal run modal_app.py::setup_volume          # Download Qwen base model
#   modal run modal_app.py::setup_fastsam_volume  # Download FastSAM model
#   modal run modal_app.py::upload_checkpoints    # Upload LoRA checkpoints



