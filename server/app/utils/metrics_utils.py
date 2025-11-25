"""
Metrics utilities for image evaluation.

Các metric và ý nghĩa (đầu vào giả định là ảnh đã chuẩn hóa về [0, 1], dạng tensor (B, C, H, W)):

- PSNR (Peak Signal-to-Noise Ratio) [dB] – Cao hơn là tốt:
  + ~20–30 dB: chấp nhận được
  + >30 dB: tốt
  + >40 dB: rất tốt/xuất sắc
  Lưu ý: tăng ~1 dB là cải thiện nhỏ; ~3 dB là đáng kể.

- SSIM (Structural Similarity) ∈ [0, 1] – Cao hơn là tốt:
  + >0.90: tốt
  + >0.95: rất tốt
  + 1.0: trùng khớp hoàn hảo

- LPIPS (Learned Perceptual Image Patch Similarity) ∈ [0, 1] – Thấp hơn là tốt:
  + <0.20: khá
  + <0.10: tốt
  + <0.05: rất tốt
  Lưu ý: Với normalize=True, metric sẽ tự chuẩn hóa từ [0,1] sang [-1,1] nội bộ.

- ΔE00 (Delta E 2000) ≥ 0 – Thấp hơn là tốt (đo sai khác màu trong không gian cảm nhận):
  + <1.0: hầu như không thể nhận ra
  + 1.0–2.0: khó nhận ra (quan sát kỹ)
  + 2.0–5.0: nhận ra nhẹ
  + 5.0–10.0: khác biệt rõ
  + >10: khác biệt lớn

Gợi ý sử dụng:
- Duy trì chuẩn hóa ảnh về [0, 1] trước khi tính PSNR/SSIM/LPIPS.
- Batch size và device của tensor nên khớp với metric instance (đã chuyển .to(device)).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

logger = logging.getLogger(__name__)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize metric instances (assume images in [0, 1])
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
# LPIPS with AlexNet backbone as required
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# Initialize CLIP model for CLIP-S (lazy loading)
_clip_model = None
_clip_preprocess = None

def _get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import clip
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
            _clip_model.eval()
            logger.info("✅ CLIP model loaded for CLIP-S metric")
        except ImportError:
            logger.warning("⚠️ CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return None, None
    return _clip_model, _clip_preprocess

logger.info(f"📊 Metrics initialized on device: {device}")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to torch.Tensor in format (B, C, H, W) with values in [0, 1].
    
    Args:
        img: PIL Image (RGB)
    
    Returns:
        Tensor of shape (1, 3, H, W) with values in [0, 1]
    """
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor and add batch dimension: (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, metric_instance: PeakSignalNoiseRatio) -> torch.Tensor:
    """
    Tính PSNR (dB). Cao hơn là tốt.
    
    Đầu vào: img1, img2 dạng (B, C, H, W), giá trị ∈ [0, 1].
    Gợi ý mức tốt: >30 dB (tốt), >40 dB (rất tốt).
    """
    img1 = img1.to(metric_instance.device)
    img2 = img2.to(metric_instance.device)
    return metric_instance(img1, img2)


def compute_lpips(img1: torch.Tensor, img2: torch.Tensor, metric_instance: LearnedPerceptualImagePatchSimilarity) -> torch.Tensor:
    """
    Tính LPIPS. Thấp hơn là tốt.
    
    Đầu vào: img1, img2 dạng (B, C, H, W), ∈ [0, 1]. Với normalize=True sẽ tự đổi về [-1, 1].
    Gợi ý mức tốt: <0.10 (tốt), <0.05 (rất tốt).
    """
    img1 = img1.to(metric_instance.device)
    img2 = img2.to(metric_instance.device)
    return metric_instance(img1, img2)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, metric_instance: StructuralSimilarityIndexMeasure) -> torch.Tensor:
    """
    Tính SSIM ∈ [0, 1]. Cao hơn là tốt.
    
    Đầu vào: img1, img2 dạng (B, C, H, W), ∈ [0, 1].
    """
    img1 = img1.to(metric_instance.device)
    img2 = img2.to(metric_instance.device)
    return metric_instance(img1, img2)


def compute_de00(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Tính Delta E 2000 (độ sai khác màu). Thấp hơn là tốt.
    
    - Đầu vào: img1, img2 dạng (B, C, H, W), ∈ [0, 1].
    - Trả về: trung bình ΔE00 của batch (float, CPU).
    - <1.0: không thể nhận ra
    - 1.0–2.0: khó nhận ra
    - 2.0–5.0: nhận ra nhẹ
    - 5.0–10.0: khác biệt rõ
    - >10: khác biệt lớn
    
    Lưu ý: Hàm chạy trên CPU và chuyển sang numpy; có thể là bottleneck cho batch lớn.
    """
    # Chuyển về (B, H, W, C) cho skimage, giữ giá trị float [0, 1]
    img1_np = img1.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    img2_np = img2.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    
    # Clip to [0, 1] to ensure valid range
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    batch_size = img1_np.shape[0]
    de00_scores = []
    
    for i in range(batch_size):
        lab1 = rgb2lab(img1_np[i])
        lab2 = rgb2lab(img2_np[i])
        de00 = deltaE_ciede2000(lab1, lab2)
        de00_scores.append(np.mean(de00))
    
    return float(np.mean(de00_scores))


def compute_clip_score(img1: Image.Image, img2: Image.Image) -> float | None:
    """
    Compute CLIP Similarity Score (CLIP-S).
    
    Args:
        img1: First image (PIL Image)
        img2: Second image (PIL Image)
    
    Returns:
        CLIP-S score (higher is better, typically in [0, 1]) or None if CLIP unavailable
    """
    clip_model, clip_preprocess = _get_clip_model()
    if clip_model is None or clip_preprocess is None:
        return None
    
    try:
        # Preprocess images
        img1_tensor = clip_preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = clip_preprocess(img2).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            img1_features = clip_model.encode_image(img1_tensor)
            img2_features = clip_model.encode_image(img2_tensor)
            
            # Normalize features
            img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
            img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            clip_score = (img1_features @ img2_features.T).item()
        
        return float(clip_score)
    except Exception as e:
        logger.warning(f"⚠️ Error computing CLIP-S: {e}")
        return None


def compute_all_metrics(img1: Image.Image, img2: Image.Image) -> dict[str, float]:
    """
    Compute all metrics for a pair of PIL Images.
    
    Args:
        img1: Original image (PIL Image)
        img2: Target/reference image (PIL Image)
    
    Returns:
        Dictionary with all metric scores (PSNR, SSIM, LPIPS, CLIP-S, ΔE00)
    """
    # Ensure same size
    if img1.size != img2.size:
        logger.info(f"🔄 Resizing images to match: {img1.size} -> {img2.size}")
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to tensors
    tensor1 = pil_to_tensor(img1)
    tensor2 = pil_to_tensor(img2)
    
    # Compute metrics
    with torch.no_grad():
        psnr_score = compute_psnr(tensor1, tensor2, psnr_metric).item()
        lpips_score = compute_lpips(tensor1, tensor2, lpips_metric).item()
        ssim_score = compute_ssim(tensor1, tensor2, ssim_metric).item()
        de00_score = compute_de00(tensor1, tensor2)
    
    # Compute CLIP-S (separate, as it uses different preprocessing)
    clip_score = compute_clip_score(img1, img2)
    
    result = {
        "psnr": float(psnr_score),
        "lpips": float(lpips_score),
        "ssim": float(ssim_score),
        "de00": float(de00_score),
    }
    
    if clip_score is not None:
        result["clip_score"] = clip_score
    
    return result

