"""
Modal app for ArtMancer backend.

- Heavy service (A100-80GB): Qwen generation, optimized for large model + LoRA.
- Light service (T4/CPU): smart-mask, image-utils, debug, system.

Supports:
- Modal Volume for checkpoints + optional base model snapshot.
- Local SSD cache for base model via MODEL_PATH env (used by qwen_loader).
"""

from __future__ import annotations

import os
import shutil
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
VOL_FASTSAM_PATH = f"{VOL_MOUNT_PATH}/fastsam"  # FastSAM-x.pt cache
# Local SSD cache (inside container)
LOCAL_CACHE_PATH = "/tmp/model_cache"
LOCAL_FASTSAM_PATH = "/tmp/fastsam_cache"  # Local cache for FastSAM-x.pt

# ==========================================
# INFRASTRUCTURE
# ==========================================
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ==========================================
# 1. HEAVY IMAGE (A100) - Tối ưu cho GPU
# ==========================================
heavy_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # Install diffusers from GitHub (latest) for QwenImageEditPlusPipeline support
        "uv pip install --system --no-cache-dir git+https://github.com/huggingface/diffusers.git",
        # Install other dependencies
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "torch>=2.4.0 "
        "torchvision>=0.19.0 "
        "transformers>=4.40.0 "
        "accelerate>=0.29.0 "
        "safetensors>=0.4.3 "
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

# ==========================================
# 2. LIGHT IMAGE (T4) - Smart-mask, image-utils (YOLO, rembg)
# ==========================================
light_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # Use CUDA torch for T4 GPU acceleration (YOLO, rembg)
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
        "rembg[gpu] "
        "onnxruntime-gpu "
        "scikit-image "
        "ultralytics "
        "opencv-python-headless "
        "psutil"
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics", "PYTHONPATH": "/root"})
    .add_local_dir("app", "/root/app", ignore=["**/__pycache__", "**/*.pyc", "**/.env", "**/.git"])
)


def create_fastapi_app(include_generation: bool = False) -> FastAPI:
    """
    Create FastAPI app instance with appropriate routers.
    
    Args:
        include_generation: If True, creates Heavy Service app (generation only).
                           If False, creates Light Service app (smart_mask, image_utils).
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    # Get allowed origins from environment or default
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,https://localhost:3000"
    ).split(",")

    fastapi_app = FastAPI(
        title="ArtMancer API",
        description="AI-powered image editing with Qwen models",
        version="2.1.0",
    )

    # Configure CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins],
        allow_origin_regex=".*",  # dev: allow any origin to avoid CORS during local use
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import common routers (no heavy dependencies)
    from app.api.endpoints import system, debug
    fastapi_app.include_router(system.router)
    fastapi_app.include_router(debug.router)

    if include_generation:
        # Heavy Service: only generation (requires diffusers, transformers, peft)
        from app.api.endpoints import generation
        fastapi_app.include_router(generation.router)
    else:
        # Light Service: smart_mask, image_utils (requires ultralytics, rembg, opencv)
        from app.api.endpoints import smart_mask, image_utils
        fastapi_app.include_router(smart_mask.router)
        fastapi_app.include_router(image_utils.router)

    # Add ping endpoint for health checks
    @fastapi_app.get("/ping")
    async def ping():
        return {"status": "healthy"}

    return fastapi_app


# ==========================================
# HEAVY SERVICE (QWEN - A100, cls-based)
# ==========================================
@app.cls(
    image=heavy_image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=1800,  # 30 phút phòng model chạy lâu
    scaledown_window=300,  # Giữ container 5 phút sau request cuối (giảm cold start)
)
@modal.concurrent(max_inputs=3)  # Input concurrency per container
class HeavyService:
    # Class attributes (initialized in @modal.enter)
    container_start_time: float | None = None
    is_ready: bool = False

    @modal.enter()
    def prepare_env(self):
        """
        Container startup hook: chạy khi container khởi động.
        - Copy base model snapshot từ Volume sang local SSD nếu có
        - Thiết lập MODEL_PATH cho qwen_loader
        - Track container lifecycle
        """
        self.container_start_time = time.time()
        self.is_ready = False
        print("🚀 [HeavyService] Container starting up...")
        
        src_model = Path(VOL_MODEL_PATH)
        dst_model = Path(f"{LOCAL_CACHE_PATH}/base_model")

        # Luôn set TASK_CHECKPOINTS_PATH để code khác dùng nếu cần
        os.environ.setdefault("TASK_CHECKPOINTS_PATH", VOL_TASKS_PATH)

        if src_model.exists():
            dst_model.parent.mkdir(parents=True, exist_ok=True)
            if not dst_model.exists():
                t0 = time.time()
                print(f"📦 [Cold Start] Copying base model from {src_model} to {dst_model} ...")
                shutil.copytree(src_model, dst_model, dirs_exist_ok=True)
                print(f"✅ Copied base model in {time.time() - t0:.2f}s")
            else:
                print(f"✅ Base model already cached at {dst_model}")

            os.environ["MODEL_PATH"] = str(dst_model)
            print(f"🚀 MODEL_PATH set to {dst_model}")
        else:
            # Chưa setup Volume: qwen_loader sẽ dùng HF ID
            print(
                f"⚠️ Base model snapshot not found at {src_model}. "
                f"qwen_loader will load from Hugging Face ({BASE_MODEL_ID})."
            )
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [HeavyService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """
        Container shutdown hook: chạy khi container tắt (sau idle timeout hoặc preemption).
        - Log container lifecycle
        - Cleanup resources nếu cần
        """
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [HeavyService] Container shutting down after {uptime:.2f}s of uptime")
        else:
            print("🛑 [HeavyService] Container shutting down")
        
        self.is_ready = False
        print("✅ [HeavyService] Cleanup complete")

    @modal.asgi_app()
    def serve(self):
        return create_fastapi_app(include_generation=True)



# ==========================================
# LIGHT SERVICE (T4/CPU) - smart-mask, image-utils, debug, system
# ==========================================
@app.cls(
    image=light_image,
    gpu="T4",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=300,  # 5 phút để đủ thời gian download models (FastSAM, rembg)
    scaledown_window=300,  # Giữ ấm 5 phút
)
@modal.concurrent(max_inputs=20)
class LightService:
    """
    Light endpoint for lightweight tasks.
    Uses T4 GPU (or CPU) for tasks like smart-mask, image-utils, debug, system.
    """
    # Class attributes (initialized in @modal.enter)
    container_start_time: float | None = None
    is_ready: bool = False
    fastsam_model: object | None = None
    rembg_session: object | None = None

    @modal.enter()
    def warmup(self):
        """
        Container startup hook: pre-load models để giảm latency request đầu.
        - Load FastSAM model cho smart-mask
        - Load rembg session cho remove-background
        - Track container lifecycle
        """
        self.container_start_time = time.time()
        self.is_ready = False
        self.fastsam_model = None
        self.rembg_session = None
        print("🚀 [LightService] Container starting up...")
        
        import torch
        print(f"🔧 [LightService] CUDA available: {torch.cuda.is_available()}")
        
        # Pre-load FastSAM model (dùng cho smart-mask)
        # Strategy: Volume → Local SSD → Load (faster cold start)
        try:
            from ultralytics import FastSAM  # type: ignore[attr-defined]
            
            fastsam_model_path = "FastSAM-x.pt"  # Default: download from GitHub
            
            # Check if FastSAM exists in Volume, copy to local SSD for faster load
            vol_fastsam_file = f"{VOL_FASTSAM_PATH}/FastSAM-x.pt"
            if os.path.exists(vol_fastsam_file):
                print(f"📦 Found FastSAM-x.pt in Volume: {vol_fastsam_file}")
                os.makedirs(LOCAL_FASTSAM_PATH, exist_ok=True)
                local_fastsam_file = f"{LOCAL_FASTSAM_PATH}/FastSAM-x.pt"
                
                # Copy from Volume to local SSD (much faster I/O)
                if not os.path.exists(local_fastsam_file):
                    print(f"📋 Copying FastSAM-x.pt from Volume to local SSD: {local_fastsam_file}")
                    shutil.copy2(vol_fastsam_file, local_fastsam_file)
                    print("✅ FastSAM-x.pt copied to local SSD")
                else:
                    print("✅ FastSAM-x.pt already in local SSD cache")
                
                fastsam_model_path = local_fastsam_file
            else:
                print("⚠️ FastSAM-x.pt not in Volume, will download from GitHub (run setup_fastsam_volume to cache)")
            
            print(f"🔄 Loading FastSAM-x from: {fastsam_model_path}")
            self.fastsam_model = FastSAM(fastsam_model_path)
            print("✅ FastSAM-x model loaded")
        except Exception as e:
            print(f"⚠️ FastSAM warmup failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Pre-load rembg session (dùng cho remove-background)
        try:
            print("🔄 Loading rembg session (u2net)...")
            from rembg import new_session
            self.rembg_session = new_session("u2net")
            print("✅ rembg session loaded")
        except Exception as e:
            print(f"⚠️ rembg warmup failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.is_ready = True
        uptime = time.time() - self.container_start_time
        print(f"✅ [LightService] Container ready! Startup took {uptime:.2f}s")

    @modal.exit()
    def cleanup(self):
        """
        Container shutdown hook: chạy khi container tắt (sau idle timeout hoặc preemption).
        - Log container lifecycle
        - Cleanup models nếu cần
        """
        if self.container_start_time:
            uptime = time.time() - self.container_start_time
            print(f"🛑 [LightService] Container shutting down after {uptime:.2f}s of uptime")
        else:
            print("🛑 [LightService] Container shutting down")
        
        # Cleanup models (optional - Modal sẽ tự cleanup khi container tắt)
        self.fastsam_model = None
        self.rembg_session = None
        self.is_ready = False
        print("✅ [LightService] Cleanup complete")

    @modal.asgi_app()
    def serve(self):
        return create_fastapi_app(include_generation=False)


# Helper: clear checkpoints from Volume
@app.function(
    volumes={VOL_MOUNT_PATH: volume},
    timeout=60,
)
def clear_checkpoints():
    """
    Remove all .safetensors files from Modal Volume.
    
    Usage:
        modal run modal_app.py::clear_checkpoints
    """
    import glob
    
    print(f"🗑️ Clearing checkpoints from Volume '{VOLUME_NAME}' ...")
    
    # Find all safetensors files
    pattern = f"{VOL_MOUNT_PATH}/*.safetensors"
    files = glob.glob(pattern)
    
    if not files:
        print("  No checkpoint files found to remove.")
        return []
    
    removed = []
    for f in files:
        try:
            os.remove(f)
            filename = os.path.basename(f)
            print(f"  🗑️ Removed: {filename}")
            removed.append(filename)
        except Exception as e:
            print(f"  ⚠️ Failed to remove {f}: {e}")
    
    # Commit changes to volume
    volume.commit()
    print(f"✅ Removed {len(removed)} checkpoint files.")
    return removed


# Helper: upload LoRA checkpoints (.safetensors) from local to Volume
@app.local_entrypoint()
def upload_checkpoints(local_checkpoints_dir: str = "./checkpoints", force: bool = False):
    """
    Upload LoRA checkpoint files (.safetensors) to Modal Volume root.

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
        print(f"No .safetensors files found in {local_path}")
        return

    print(f"Uploading checkpoints from {local_path} to Volume '{VOLUME_NAME}' ...")
    
    if force:
        print("🔄 Force mode: clearing existing checkpoints first...")
        # Call remote function to clear existing checkpoints
        clear_checkpoints.remote()
        print("✅ Existing checkpoints cleared.")

    with volume.batch_upload() as upload:
        for checkpoint_file in checkpoint_files:
            remote_path = f"/{checkpoint_file.name}"
            print(f"  📤 Uploading {checkpoint_file.name} -> {remote_path}")
            upload.put_file(checkpoint_file, remote_path)

    print("✅ Checkpoints uploaded. Files are now available at /checkpoints/ in Modal containers.")


# Helper: setup base model snapshot inside Volume (optional, for large models)
@app.function(
    image=heavy_image,
    volumes={VOL_MOUNT_PATH: volume},
    timeout=3600,  # Cho phép chạy 1 tiếng để download
)
def setup_volume():
    """
    Download Qwen base model into Modal Volume for faster cold starts.

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


# Helper: setup FastSAM-x.pt model inside Volume for faster LightService cold starts
@app.function(
    image=light_image,
    volumes={VOL_MOUNT_PATH: volume},
    timeout=600,  # 10 phút để download
)
def setup_fastsam_volume():
    """
    Download FastSAM-x.pt model into Modal Volume for faster LightService cold starts.
    
    Usage:
        modal run modal_app.py::setup_fastsam_volume
    """
    import urllib.request
    
    print("🚀 Starting setup_fastsam_volume: downloading FastSAM-x.pt to Volume...")
    os.makedirs(VOL_FASTSAM_PATH, exist_ok=True)
    
    fastsam_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-x.pt"
    vol_fastsam_file = f"{VOL_FASTSAM_PATH}/FastSAM-x.pt"
    
    try:
        print(f"📥 Downloading FastSAM-x.pt from {fastsam_url}...")
        urllib.request.urlretrieve(fastsam_url, vol_fastsam_file)
        
        # Get file size
        file_size = os.path.getsize(vol_fastsam_file)
        print(f"✅ FastSAM-x.pt downloaded to {vol_fastsam_file} ({file_size / 1024 / 1024:.1f} MB)")
        
        # Commit changes to Volume
        volume.commit()
        
        print("✅ FastSAM-x.pt cached in Volume. LightService will use this for faster cold starts.")
    except Exception as exc:
        print(f"❌ Failed to download FastSAM-x.pt: {exc}")
        import traceback
        traceback.print_exc()
        return


# For local testing
if __name__ == "__main__":
    # This allows running with: modal run modal_app.py
    # or: modal serve modal_app.py
    pass

