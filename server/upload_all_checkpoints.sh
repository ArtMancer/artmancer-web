#!/bin/bash
# Script để upload tất cả checkpoints vào Modal Volume

set -e

echo "🚀 Bắt đầu upload checkpoints vào Modal Volume..."
echo ""

# 1. Upload Qwen base model and FastSAM model (tùy chọn - mất nhiều thời gian)
echo "📦 [1/2] Uploading Qwen base model and FastSAM model..."
echo "⚠️  Lưu ý: Qwen base model rất lớn, có thể mất 30-60 phút"
read -p "Bạn có muốn upload models không? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    modal run modal_app.py::setup_volume
    echo "✅ Models uploaded (Qwen base model + FastSAM)"
else
    echo "⏭️  Bỏ qua models (sẽ tự động download khi cần)"
fi
echo ""

# 2. Upload LoRA checkpoints
echo "📦 [2/2] Uploading LoRA checkpoints (.safetensors)..."
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

