"""
Test script for ArtMancer generation on Modal.

Qwen Image Edit conditional images:
- Insertion (3 images): object, mask, masked_bg
- Removal (3 images): mask, masked_bg, mae

Usage:
    python test_generate.py                           # Basic test
    python test_generate.py --prompt "a cute cat"     # Custom prompt
    python test_generate.py --task removal            # Object removal
    python test_generate.py --steps 10                # Faster (fewer steps)
    python test_generate.py --check-only              # Health check only
"""
from __future__ import annotations

import base64
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ==========================================
# CONFIGURATION
# ==========================================
# Modal endpoints
MODAL_HEAVY_ENDPOINT = "https://nxan2911--artmancer-heavyservice-serve.modal.run"
MODAL_LIGHT_ENDPOINT = "https://nxan2911--artmancer-lightservice-serve.modal.run"

# Output directory for generated images
OUTPUT_DIR = Path(__file__).parent / "generated_images"

# Default test images - conditional images đã được chuẩn bị sẵn
DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_INPUT_IMAGE = DATASET_DIR / "image.png"  # Original (dùng để backend tạo conditionals nếu cần)

# Conditional images theo task:
# - Insertion: object, mask, masked_bg (3 images)
# - Removal: mask, masked_bg, mae (3 images) - NO object
DEFAULT_OBJECT = DATASET_DIR / "condition-1.png"      # Object isolated
DEFAULT_MASK = DATASET_DIR / "condition-2.png"        # Mask (white = edit area)
DEFAULT_MASKED_BG = DATASET_DIR / "condition-3.png"   # Background with mask applied
DEFAULT_MAE = DATASET_DIR / "condition-3.png"         # MAE image (for removal only)


def _load_image_base64(path: str | Path) -> str:
    """Load image file and convert to base64 string."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _save_base64_image(b64_data: str, output_path: Path) -> Path:
    """Save base64 image data to file."""
    # Remove data URL prefix if present
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    
    image_data = base64.b64decode(b64_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_data)
    return output_path


def open_image(image_path: Path) -> None:
    """Open image with default system viewer."""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["open", str(image_path)], check=True)
        elif system == "Windows":
            os.startfile(str(image_path))  # type: ignore
        else:  # Linux/WSL
            # Try xdg-open first
            try:
                subprocess.run(["xdg-open", str(image_path)], check=True, stderr=subprocess.DEVNULL)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # WSL: try Windows explorer
                try:
                    wsl_path = subprocess.run(
                        ["wslpath", "-w", str(image_path)],
                        capture_output=True, text=True, check=True
                    ).stdout.strip()
                    subprocess.run(["explorer.exe", wsl_path], check=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print(f"   ℹ️  View image at: {image_path}")
                    return
        print(f"   🖼️  Opened: {image_path.name}")
    except Exception as e:
        print(f"   ⚠️  Could not open image: {e}")
        print(f"   ℹ️  View at: {image_path}")


def health_check(endpoint_url: str) -> bool:
    """
    Check if Modal endpoint is healthy (waits for cold start, no timeout).
    
    Args:
        endpoint_url: Modal endpoint URL
    
    Returns:
        True if healthy, False on error
    """
    ping_url = f"{endpoint_url}/ping"
    print(f"🏥 Health check: {ping_url}")
    print("   ⏳ Waiting for response (Modal cold start may take 1-2 min)...")
    
    try:
        # No timeout - wait until server responds
        response = requests.get(ping_url)
        
        if response.status_code == 200:
            print("   ✅ Ready!")
            return True
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        return False


def load_conditional_images_base64(
    task_type: str,
    object_path: Optional[Path] = None,
    mask_path: Optional[Path] = None,
    masked_bg_path: Optional[Path] = None,
    mae_path: Optional[Path] = None,
) -> list[str]:
    """
    Load conditional images based on task type.
    
    Insertion (3 images): object, mask, masked_bg
    Removal (3 images): mask, masked_bg, mae
    
    Args:
        task_type: "insertion" or "removal"
        object_path: Path to object image (for insertion)
        mask_path: Path to mask image
        masked_bg_path: Path to masked background image
        mae_path: Path to MAE image (for removal)
    
    Returns:
        List of base64 encoded image strings
    """
    if task_type == "insertion":
        # Insertion: object, mask, masked_bg (3 images)
        paths = [
            ("object", object_path or DEFAULT_OBJECT),
            ("mask", mask_path or DEFAULT_MASK),
            ("masked_bg", masked_bg_path or DEFAULT_MASKED_BG),
        ]
    elif task_type == "removal":
        # Removal: mask, masked_bg, mae (3 images) - NO object
        paths = [
            ("mask", mask_path or DEFAULT_MASK),
            ("masked_bg", masked_bg_path or DEFAULT_MASKED_BG),
            ("mae", mae_path or DEFAULT_MAE),
        ]
    else:
        # White-balance or other: same as insertion for now
        paths = [
            ("object", object_path or DEFAULT_OBJECT),
            ("mask", mask_path or DEFAULT_MASK),
            ("masked_bg", masked_bg_path or DEFAULT_MASKED_BG),
        ]
    
    images_b64 = []
    for name, path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Conditional image '{name}' not found: {path}")
        
        b64 = _load_image_base64(path)
        images_b64.append(b64)
        print(f"   ✅ Loaded {name}: {path.name}")
    
    print(f"   📊 Total conditionals: {len(images_b64)} images for task '{task_type}'")
    return images_b64


def build_request(
    prompt: str,
    task_type: str = "insertion",
    input_image_path: Optional[Path] = None,
    object_path: Optional[Path] = None,
    mask_path: Optional[Path] = None,
    masked_bg_path: Optional[Path] = None,
    mae_path: Optional[Path] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.0,
    true_cfg_scale: float = 3.3,
    seed: int = 42,
    input_quality: str = "resized",
) -> dict[str, Any]:
    """
    Build generation request payload with conditional images based on task.
    
    Insertion: 3 images (object, mask, masked_bg)
    Removal: 4 images (mask, masked_bg, object, mae)
    
    Args:
        prompt: Generation prompt
        task_type: "insertion", "removal", or "white-balance"
        input_image_path: Path to input image
        object_path: Path to object image
        mask_path: Path to mask image
        masked_bg_path: Path to masked background image
        mae_path: Path to MAE image (for removal only)
        num_inference_steps: Number of steps
        guidance_scale: Guidance scale
        true_cfg_scale: True CFG scale
        seed: Random seed
        input_quality: "resized" or "original"
    
    Returns:
        Request payload dict
    """
    # Use default paths if not provided
    input_path = input_image_path or DEFAULT_INPUT_IMAGE
    obj = object_path or DEFAULT_OBJECT
    mask = mask_path or DEFAULT_MASK
    masked_bg = masked_bg_path or DEFAULT_MASKED_BG
    mae = mae_path or DEFAULT_MAE
    
    # Load input image (backend may use for reference)
    try:
        input_image_b64 = _load_image_base64(input_path)
        print(f"   ✅ Input: {input_path.name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Load conditional images based on task type
    conditional_images_b64 = load_conditional_images_base64(
        task_type=task_type,
        object_path=obj,
        mask_path=mask,
        masked_bg_path=masked_bg,
        mae_path=mae if task_type == "removal" else None,
    )

    return {
        "prompt": prompt,
        "input_image": input_image_b64,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "task_type": task_type,
        "seed": seed,
        "input_quality": input_quality,
        "conditional_images": conditional_images_b64,
    }


def generate(
    prompt: str = "yellow rubber duck, orange beak, black eyes, frontal perspective, white textured surface, photorealistic, ultra-detailed, correct white balance, soft neutral lighting",
    task_type: str = "insertion",
    input_image: Optional[str] = None,
    object_img: Optional[str] = None,
    mask_img: Optional[str] = None,
    masked_bg_img: Optional[str] = None,
    mae_img: Optional[str] = None,
    num_steps: int = 28,
    seed: int = 42,
    endpoint_url: Optional[str] = None,
    save_output: bool = True,
    open_output: bool = True,
) -> Optional[Path]:
    """
    Generate image using Modal endpoint.
    
    Conditional images per task:
    - Insertion: object, mask, masked_bg (3 images)
    - Removal: mask, masked_bg, mae (3 images)
    
    Args:
        prompt: Generation prompt
        task_type: "insertion", "removal", or "white-balance"
        input_image: Path to input image
        object_img: Path to object image
        mask_img: Path to mask image
        masked_bg_img: Path to masked background image
        mae_img: Path to MAE image (for removal only)
        num_steps: Number of inference steps
        seed: Random seed
        endpoint_url: Modal endpoint URL
        save_output: Save output image to file
        open_output: Open output image after generation
    
    Returns:
        Path to saved image if successful, None otherwise
    """
    endpoint = endpoint_url or MODAL_HEAVY_ENDPOINT
    endpoint = endpoint.rstrip("/")
    
    print("=" * 60)
    print("🎨 ArtMancer Generation Test")
    print("=" * 60)
    print(f"🌐 Endpoint: {endpoint}")
    print()
    
    # Health check
    if not health_check(endpoint):
        print("❌ Endpoint not ready")
        return None
    
    # Build request
    print("\n📦 Preparing request...")
    try:
        payload = build_request(
            prompt=prompt,
            task_type=task_type,
            input_image_path=Path(input_image) if input_image else None,
            object_path=Path(object_img) if object_img else None,
            mask_path=Path(mask_img) if mask_img else None,
            masked_bg_path=Path(masked_bg_img) if masked_bg_img else None,
            mae_path=Path(mae_img) if mae_img else None,
            num_inference_steps=num_steps,
            seed=seed,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Send async request
    api_url = f"{endpoint}/api/generate/async"
    num_conds = 3  # Both insertion and removal use 3 conditional images
    
    print("\n🚀 Generating (async mode)...")
    print(f"   📝 Prompt: {prompt[:80]}...")
    print(f"   🎯 Task: {task_type}")
    print(f"   🔢 Steps: {num_steps}")
    print(f"   🎲 Seed: {seed}")
    print(f"   📊 Conditionals: {num_conds} images")
    print()
    
    start_time = time.time()
    
    try:
        # Submit async job
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,  # Short timeout for job submission
        )
        
        elapsed = time.time() - start_time
        print(f"📡 Job submitted in {elapsed:.1f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            if not task_id:
                print("❌ No task_id in response")
                return None
            
            print(f"✅ Task ID: {task_id}")
            print("⏳ Waiting for generation to complete...")
            
            # Poll for result
            status_url = f"{endpoint}/api/generate/status/{task_id}"
            result_url = f"{endpoint}/api/generate/result/{task_id}"
            
            max_wait = 600  # 10 minutes max
            poll_interval = 2  # Poll every 2 seconds
            waited = 0
            
            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval
                
                status_response = requests.get(status_url, timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status", "unknown")
                    progress = status_data.get("progress", 0.0)
                    
                    print(f"   Status: {status} ({progress*100:.1f}%)", end="\r")
                    
                    if status == "done":
                        print("\n✅ Generation completed!")
                        # Get result
                        result_response = requests.get(result_url, timeout=30)
                        if result_response.status_code == 200:
                            result = result_response.json()
                            
                            elapsed = time.time() - start_time
                            print(f"   ⏱️  Total time: {elapsed:.1f}s")
                            print(f"   🤖 Model: {result.get('model_used', 'N/A')}")
                            
                            # Save image
                            if result.get("image"):
                                image_b64 = result["image"]
                                print(f"   📊 Image: {len(image_b64):,} chars")
                                
                                if save_output:
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_path = OUTPUT_DIR / f"gen_{task_type}_{timestamp}.png"
                                    
                                    saved = _save_base64_image(image_b64, output_path)
                                    print(f"\n💾 Saved: {saved}")
                                    
                                    if open_output:
                                        open_image(saved)
                                    
                                    return saved
                            else:
                                print("   ⚠️  No image in result")
                                return None
                        break
                    elif status == "error":
                        print(f"\n❌ Generation failed: {status_data.get('error', 'Unknown error')}")
                        return None
                    elif status == "cancelled":
                        print("\n🚫 Generation cancelled")
                        return None
                
                if waited >= max_wait:
                    print(f"\n⏱️  Timeout after {max_wait}s")
                    return None
        else:
            print(f"\n❌ Error {response.status_code}")
            try:
                err = response.json()
                print(f"   {err.get('error', err.get('detail', response.text[:200]))}")
            except Exception:
                print(f"   {response.text[:300]}")
                
    except requests.exceptions.Timeout:
        print("\n❌ Timeout")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test ArtMancer generation on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Insertion (3 conditionals: object, mask, masked_bg)
  python test_generate.py
  python test_generate.py --prompt "a cute cat on a table"
  
  # Removal (3 conditionals: mask, masked_bg, mae)
  python test_generate.py --task removal --prompt "remove the object"
  
  # Custom conditional images (insertion)
  python test_generate.py --prompt "a dog" --object obj.png --mask mask.png --masked-bg bg.png
  
  # Custom conditional images (removal - no object needed)
  python test_generate.py --task removal --mask mask.png --masked-bg bg.png --mae mae.png
  
  python test_generate.py --steps 10 --no-open
  python test_generate.py --check-only
"""
    )
    
    parser.add_argument(
        "--prompt", "-p",
        default="yellow rubber duck, orange beak, black eyes, frontal perspective, white textured surface, photorealistic, ultra-detailed, correct white balance, soft neutral lighting",
        help="Generation prompt"
    )
    parser.add_argument(
        "--task", "-t",
        default="insertion",
        choices=["insertion", "removal", "white-balance"],
        help="Task type (default: insertion)"
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_image",
        help="Input image path"
    )
    parser.add_argument(
        "--object",
        dest="object_img",
        help="Object image (isolated object)"
    )
    parser.add_argument(
        "--mask",
        dest="mask_img",
        help="Mask image (white = edit area)"
    )
    parser.add_argument(
        "--masked-bg",
        dest="masked_bg_img",
        help="Masked background image"
    )
    parser.add_argument(
        "--mae",
        dest="mae_img",
        help="MAE image (for removal task only)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=28,
        help="Inference steps (default: 28)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--endpoint", "-e",
        help=f"Modal endpoint URL (default: {MODAL_HEAVY_ENDPOINT})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open output"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Health check only"
    )
    
    args = parser.parse_args()
    
    endpoint = args.endpoint or MODAL_HEAVY_ENDPOINT
    
    if args.check_only:
        print(f"🔎 Checking: {endpoint}")
        for path in ["/ping", "/api/health"]:
            url = endpoint.rstrip("/") + path
            print(f"\n📡 GET {url}")
            try:
                r = requests.get(url, timeout=30)
                print(f"   Status: {r.status_code}")
                try:
                    print(f"   Response: {r.json()}")
                except Exception:
                    print(f"   Response: {r.text[:200]}")
            except Exception as e:
                print(f"   Error: {e}")
    else:
        result = generate(
            prompt=args.prompt,
            task_type=args.task,
            input_image=args.input_image,
            object_img=args.object_img,
            mask_img=args.mask_img,
            masked_bg_img=args.masked_bg_img,
            mae_img=args.mae_img,
            num_steps=args.steps,
            seed=args.seed,
            endpoint_url=args.endpoint,
            save_output=not args.no_save,
            open_output=not args.no_open,
        )
        
        if result:
            print(f"\n🎉 Done! Output: {result}")
        else:
            print("\n⚠️  Generation failed or incomplete")


if __name__ == "__main__":
    main()
