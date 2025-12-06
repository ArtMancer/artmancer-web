#!/bin/bash
# Script để upload tất cả checkpoints vào Modal Volume

set -e

echo "🚀 Bắt đầu upload checkpoints vào Modal Volume..."
echo ""

# 1. Upload FastSAM checkpoint
echo "📦 [1/3] Uploading FastSAM-s.pt checkpoint..."
modal run modal_app.py::setup_fastsam_volume
echo "✅ FastSAM checkpoint uploaded"
echo ""

# 2. Upload Qwen base model (tùy chọn - mất nhiều thời gian)
echo "📦 [2/3] Uploading Qwen base model..."
echo "⚠️  Lưu ý: Qwen base model rất lớn, có thể mất 30-60 phút"
read -p "Bạn có muốn upload Qwen base model không? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    modal run modal_app.py::setup_volume
    echo "✅ Qwen base model uploaded"
else
    echo "⏭️  Bỏ qua Qwen base model (sẽ tự động download khi cần)"
fi
echo ""

# 3. Upload LoRA checkpoints
echo "📦 [3/3] Uploading LoRA checkpoints (.safetensors)..."
if [ -d "./checkpoints" ] && [ "$(ls -A ./checkpoints/*.safetensors 2>/dev/null)" ]; then
    modal run modal_app.py::upload_checkpoints --local-checkpoints-dir ./checkpoints
    echo "✅ LoRA checkpoints uploaded"
else
    echo "⚠️  Không tìm thấy file .safetensors trong ./checkpoints"
    echo "   Vui lòng đảm bảo có các file: insertion_cp.safetensors, removal_cp.safetensors, wb_cp.safetensors"
fi
echo ""

echo "✅ Hoàn tất! Tất cả checkpoints đã được upload vào Modal Volume."
echo "📂 Volume path: /checkpoints/"
echo "   - FastSAM: /checkpoints/fastsam/FastSAM-s.pt"
echo "   - Qwen base: /checkpoints/base_model/"
echo "   - LoRA: /checkpoints/*.safetensors"

