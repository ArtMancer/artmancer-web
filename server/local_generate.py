"""
Script chạy trực tiếp Qwen Image Edit trên máy Local (Không dùng API/Server).

Qwen Image Edit cần 4 conditional images:
1. mask_cond: Mask image (RGB, vùng trắng = vùng cần edit)
2. background_rgb: Background với mask applied (vùng mask = đen)
3. obj_rgb: Object isolated (vùng ngoài mask = đen)
4. mae_image: MAE inpainted preview

Yêu cầu: Đã cài torch, diffusers (from git), transformers, accelerate.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
MODEL_PATH = "./checkpoints/base_model"
TASKS_PATH = "./checkpoints/tasks"

# Dataset paths - các conditional images đã được chuẩn bị sẵn
DATASET_DIR = "./dataset"
DEFAULT_COND_1 = f"{DATASET_DIR}/condition-1.png"  # Object/Mask
DEFAULT_COND_2 = f"{DATASET_DIR}/condition-2.png"  # Background masked
DEFAULT_COND_3 = f"{DATASET_DIR}/condition-3.png"  # Masked BG / MAE
DEFAULT_COND_4 = f"{DATASET_DIR}/condition-3.png"  # MAE (dùng lại cond-3 nếu không có)

OUTPUT_DIR = "./output_local"


def setup_pipeline(model_path: str):
    """Load Qwen pipeline vào GPU."""
    print(f"⏳ Đang load model từ: {model_path}")
    print("   (Việc này có thể mất 1-2 phút)...")
    
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError:
        print("❌ Không tìm thấy QwenImageEditPlusPipeline!")
        print("   Cài diffusers từ GitHub: pip install git+https://github.com/huggingface/diffusers.git")
        raise

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        use_safetensors=True,
    )
    
    print("✅ Model loaded!")
    return pipe


def load_task_checkpoint(pipe, tasks_path: str, task_name: str):
    """Load LoRA weights cho task cụ thể."""
    file_map = {
        "insertion": "insertion_cp.safetensors",
        "removal": "removal_cp.safetensors",
        "white-balance": "wb_cp.safetensors",
    }
    
    filename = file_map.get(task_name)
    if not filename:
        print(f"⚠️ Unknown task '{task_name}', using base model.")
        return pipe

    ckpt_path = Path(tasks_path) / filename
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint not found: {ckpt_path}. Using base model.")
        return pipe

    print(f"🔄 Loading task checkpoint: {filename}...")
    
    try:
        pipe.load_lora_weights(str(tasks_path), weight_name=filename, adapter_name=task_name)
        pipe.set_adapters([task_name])
        print(f"✅ Loaded LoRA weights for task: {task_name}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("   Continuing with base model.")

    return pipe


def load_conditional_images(
    cond1_path: str,
    cond2_path: str,
    cond3_path: str,
    cond4_path: str,
) -> list[Image.Image]:
    """
    Load 4 conditional images for Qwen pipeline.
    
    Qwen expects these in order:
    1. mask_cond: Mask (white = area to edit)
    2. background_rgb: Background with mask applied
    3. obj_rgb: Object isolated
    4. mae_image: MAE inpainted preview
    
    Returns:
        List of 4 PIL Images in RGB mode
    """
    paths = [cond1_path, cond2_path, cond3_path, cond4_path]
    images = []
    
    for i, path in enumerate(paths, 1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Conditional image {i} not found: {path}")
        
        img = Image.open(path).convert("RGB")
        images.append(img)
        print(f"   ✅ Loaded cond-{i}: {path} ({img.size})")
    
    # Ensure all images have the same size
    base_size = images[0].size
    for i, img in enumerate(images[1:], 2):
        if img.size != base_size:
            print(f"   🔄 Resizing cond-{i} from {img.size} to {base_size}")
            images[i-1] = img.resize(base_size, Image.Resampling.LANCZOS)
    
    return images


def generate(
    pipe,
    prompt: str,
    conditional_images: list[Image.Image],
    num_steps: int = 28,
    guidance_scale: float = 4.0,
    true_cfg_scale: float = 3.3,
    seed: int = 42,
) -> Image.Image:
    """
    Generate image using Qwen pipeline.
    
    Args:
        pipe: QwenImageEditPlusPipeline
        prompt: Text prompt
        conditional_images: List of 4 conditional images
        num_steps: Number of inference steps
        guidance_scale: Guidance scale (not used by Qwen, but kept for compatibility)
        true_cfg_scale: True CFG scale for Qwen
        seed: Random seed
    
    Returns:
        Generated PIL Image
    """
    # Get output size from first conditional image
    width, height = conditional_images[0].size
    
    # Setup generator for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    print(f"\n🚀 Generating...")
    print(f"   📝 Prompt: {prompt}")
    print(f"   🔢 Steps: {num_steps}")
    print(f"   📐 Size: {width}x{height}")
    print(f"   🎲 Seed: {seed}")
    
    t0 = time.time()
    
    with torch.inference_mode():
        # Qwen pipeline expects:
        # - image: list of conditional images (NOT the original input image!)
        # - prompt: text description
        # - true_cfg_scale: guidance
        result = pipe(
            image=conditional_images,  # List of 4 conditional images
            prompt=prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            generator=generator,
        )
    
    duration = time.time() - t0
    print(f"✅ Done in {duration:.2f}s")
    
    # Get output image
    if hasattr(result, "images"):
        output = result.images[0] if isinstance(result.images, list) else result.images
    else:
        output = result[0] if isinstance(result, (list, tuple)) else result
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen Image Edit locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python local_generate.py -p "a yellow rubber duck on the table"
  python local_generate.py -p "remove the object" -t removal
  python local_generate.py -p "a cute cat" --cond1 mask.png --cond2 bg.png --cond3 obj.png --cond4 mae.png
"""
    )
    
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    parser.add_argument("--task", "-t", type=str, default="insertion", 
                        choices=["insertion", "removal", "white-balance"])
    parser.add_argument("--cond1", type=str, default=DEFAULT_COND_1, 
                        help="Conditional image 1 (mask/object)")
    parser.add_argument("--cond2", type=str, default=DEFAULT_COND_2, 
                        help="Conditional image 2 (background)")
    parser.add_argument("--cond3", type=str, default=DEFAULT_COND_3, 
                        help="Conditional image 3 (masked bg)")
    parser.add_argument("--cond4", type=str, default=DEFAULT_COND_4, 
                        help="Conditional image 4 (MAE)")
    parser.add_argument("--steps", "-s", type=int, default=28, help="Inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg", type=float, default=3.3, help="True CFG scale")
    
    args = parser.parse_args()

    # 1. Load conditional images
    print("\n🖼️  Loading conditional images...")
    try:
        conditional_images = load_conditional_images(
            args.cond1, args.cond2, args.cond3, args.cond4
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 2. Load model
    pipe = setup_pipeline(MODEL_PATH)
    
    # 3. Load task checkpoint
    pipe = load_task_checkpoint(pipe, TASKS_PATH, args.task)

    # 4. Generate
    output = generate(
        pipe=pipe,
        prompt=args.prompt,
        conditional_images=conditional_images,
        num_steps=args.steps,
        true_cfg_scale=args.cfg,
        seed=args.seed,
    )

    # 5. Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = f"{OUTPUT_DIR}/result_{args.task}_{timestamp}.png"
    output.save(save_path)
    print(f"\n💾 Saved: {save_path}")
    
    # Try to open image
    try:
        if os.name == 'nt':
            os.startfile(save_path)
        elif os.uname().sysname == 'Darwin':
            os.system(f"open '{save_path}'")
        else:
            os.system(f"xdg-open '{save_path}' 2>/dev/null || echo 'View at: {save_path}'")
    except Exception:
        pass


if __name__ == "__main__":
    main()
