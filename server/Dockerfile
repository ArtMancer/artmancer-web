# 1. Base Image: Dùng bản Runtime là chuẩn (nhẹ hơn bản devel)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2. Thiết lập biến môi trường quan trọng
# PYTHONUNBUFFERED=1: Giúp log hiện ngay lập tức trên RunPod (rất quan trọng để debug)
# PYTHONDONTWRITEBYTECODE=1: Không sinh ra file .pyc (tiết kiệm chút ít dung lượng)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PORT=80

WORKDIR /app

# 3. Cài đặt system dependencies
# Gom nhóm lại và dọn dẹp ngay trong cùng 1 layer để giảm dung lượng
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3-pip \
    # Thêm curl hoặc wget nếu cần healthcheck sau này
    && rm -rf /var/lib/apt/lists/*

# 4. Cài đặt uv (Package manager siêu tốc)
RUN pip install --no-cache-dir uv

# 5. Copy file dependency trước
COPY pyproject.toml uv.lock* ./

# 6. Cài đặt Dependencies
# TỐI ƯU HÓA:
# - Dùng --no-cache để uv không lưu file cài đặt tạm (giảm dung lượng image đáng kể).
# - Chỉ định index-url cho pytorch để đảm bảo cài đúng bản cu121 (tránh cài nhầm bản cpu hoặc cuda khác gây nặng).
RUN echo "📦 Installing dependencies..." && \
    uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    --find-links https://download.pytorch.org/whl/cu121 \
    fastapi[standard]>=0.115.0 \
    uvicorn[standard]>=0.23.0 \
    torch>=2.1.0 \
    torchvision>=0.16.0 \
    "diffusers @ git+https://github.com/huggingface/diffusers.git" \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    safetensors>=0.4.1 \
    peft>=0.7.0 \
    pillow>=10.2.0 \
    numpy>=1.26.0 \
    python-multipart \
    pydantic>=2.5.0 \
    pydantic-settings \
    python-dotenv \
    "huggingface-hub[cli]" \
    requests \
    hf-xet \
    rembg \
    onnxruntime-gpu \
    torchmetrics \
    scikit-image \
    ultralytics \
    opencv-python-headless \
    pandas

# Lưu ý: Mình đã sửa onnxruntime -> onnxruntime-gpu để tận dụng GPU

# 7. Copy Code (Luôn để cuối cùng để tận dụng Docker Layer Caching)
COPY app/ ./app/
COPY rp_handler.py ./

# 8. Run
EXPOSE 80
CMD ["python3", "rp_handler.py"]