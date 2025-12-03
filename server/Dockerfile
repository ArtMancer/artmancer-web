# Use CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install --no-cache-dir uv && \
    echo "✅ uv installed"

# Copy dependency files first (for better caching)
# NOTE: If building from GitHub, ensure build context is set to 'server/' directory
# In RunPod: Set "Docker Build Context" to "server" when deploying from GitHub
COPY pyproject.toml uv.lock* ./

# Install all dependencies
# This step can take 5-15 minutes depending on network speed
RUN echo "📦 Installing Python dependencies..." && \
    echo "   This may take 5-15 minutes. Please wait..." && \
    uv pip install --system \
    fastapi[standard]>=0.115.0 \
    uvicorn[standard]>=0.23.0 \
    torch>=2.8.0 \
    torchvision>=0.23.0 \
    "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    transformers>=4.57.3 \
    accelerate>=1.10.1 \
    bitsandbytes>=0.43.0 \
    safetensors>=0.4.0 \
    peft>=0.10.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    python-multipart>=0.0.9 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.6.0 \
    python-dotenv>=1.0.0 \
    "huggingface-hub[cli]>=0.35.1" \
    requests>=2.31.0 \
    hf-xet>=1.1.10 \
    rembg>=2.0.68 \
    onnxruntime>=1.16.0 \
    torchmetrics>=1.0.0 \
    scikit-image>=0.21.0 \
    ultralytics>=8.0.0 \
    opencv-python-headless>=4.11.0.86 \
    pandas>=2.3.3 && \
    echo "✅ All dependencies installed"

# Verify OpenCV installation
RUN echo "🔍 Verifying OpenCV..." && \
    python3 -c "import cv2; print(f'✅ OpenCV verified: cv2 version {cv2.__version__}')" || \
    (echo "❌ ERROR: OpenCV (cv2) is not installed!" && exit 1)

# Download Qwen base model during build (OPTIONAL - can be slow)
# NOTE: This step can take 10-30 minutes. If build times out, comment out this section.
# The model will be automatically downloaded at runtime on first use if not cached.
# Uncomment the section below if you want to pre-download the model in the image:
#
# RUN echo "📥 Pre-downloading Qwen model (this may take 10-30 minutes)..." && \
#     python3 -c "\
# from huggingface_hub import snapshot_download; \
# from pathlib import Path; \
# base_model_id = 'Qwen/Qwen-Image-Edit-2509'; \
# cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'; \
# model_cache_path = cache_dir / f'models--{base_model_id.replace(\"/\", \"--\")}'; \
# if not (model_cache_path.exists() and any(model_cache_path.iterdir())): \
#     print(f'📥 Downloading {base_model_id}...'); \
#     snapshot_download(repo_id=base_model_id, local_dir_use_symlinks=False, resume_download=True); \
#     print(f'✅ Downloaded {base_model_id}'); \
# else: \
#     print(f'✅ Model already cached'); \
# "
#
# For now, skip model download in build - it will be downloaded at runtime
RUN echo "ℹ️  Qwen model will be downloaded at runtime on first use"

# Copy source code (must be last to avoid rebuild on every code change)
COPY app/ ./app/
COPY rp_handler.py ./

# Set environment variables
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV PORT=80
ENV PORT_HEALTH=80

# Expose port 80 (default for RunPod)
EXPOSE 80

# Run the handler
CMD ["python3", "rp_handler.py"]

