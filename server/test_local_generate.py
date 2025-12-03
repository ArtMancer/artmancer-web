from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


# Load environment variables from .env file if exists
def load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file."""
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)


def _load_image_base64(path: str) -> str:
    """Load image file and convert to base64 string."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def build_sample_request() -> Dict[str, Any]:
    """
    Build sample payload matching GenerationRequest to test RunPod endpoint.
    """
    # TODO: Update these paths to match your machine
    # If files don't exist, you can use placeholder base64 strings for testing
    try:
        input_image_b64 = _load_image_base64("dataset/image.png")
        mask_image_b64 = _load_image_base64("dataset/condition-1.png")
    except FileNotFoundError:
        print("⚠️  Warning: Image files not found. Using placeholder values.")
        print("   Update paths in build_sample_request() or create test images.")
        # Placeholder - you should replace with actual image base64
        input_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="  # 1x1 red pixel
        mask_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # conditional_images: first element is mask, subsequent elements (if any) are extra conditionals
    conditional_images = [mask_image_b64]

    return {
        "prompt": "yellow rubber duck on white table, photorealistic",
        "input_image": input_image_b64,
        "num_inference_steps": 25,
        "guidance_scale": 4.0,
        "true_cfg_scale": 3.3,
        "task_type": "insertion",  # "insertion", "removal", or "white-balance"
        "seed": 42,
        "input_quality": "resized",  # "resized" or "original"
        "conditional_images": conditional_images,
    }


def get_auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """
    Get authentication headers for RunPod API requests.
    
    Args:
        api_key: RunPod API key (optional, can be set via RUNPOD_API_KEY env var)
    
    Returns:
        Dictionary with headers including Authorization if API key is provided
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("RUNPOD_API_KEY")
    
    if api_key:
        # RunPod uses Authorization: Bearer <API_KEY> format
        headers["Authorization"] = f"Bearer {api_key}"
        print("🔑 Using API key for authentication")
    else:
        print("⚠️  No API key provided - endpoint may require authentication")
    
    return headers


def health_check(endpoint_url: str, api_key: Optional[str] = None, max_retries: int = 5, delay: int = 10) -> bool:
    """
    Check if RunPod endpoint is healthy (handle cold start).
    
    Args:
        endpoint_url: Base URL of RunPod endpoint (e.g., https://xxx.api.runpod.ai)
        api_key: RunPod API key (optional)
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    
    Returns:
        True if endpoint is healthy, False otherwise
    """
    ping_url = f"{endpoint_url}/ping"
    print(f"🏥 Checking health at {ping_url}...")
    
    headers = get_auth_headers(api_key)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(ping_url, headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ Health check passed - endpoint is ready")
                return True
            elif response.status_code == 204:
                print(f"⏳ Endpoint initializing (attempt {attempt + 1}/{max_retries})...")
            elif response.status_code == 401:
                print("❌ Authentication failed (401) - API key may be required or invalid")
                print("   Set RUNPOD_API_KEY environment variable or pass --api-key argument")
                return False
            else:
                print(f"⚠️  Health check returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Health check attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            print(f"🔄 Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print("❌ Health check failed after all retries")
    return False


def test_runpod_endpoint(endpoint_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
    """
    Test RunPod endpoint with sample generation request.
    
    Args:
        endpoint_url: RunPod endpoint URL (defaults to environment variable or default)
        api_key: RunPod API key (optional, can be set via RUNPOD_API_KEY env var)
    """
    # Get endpoint URL from environment or use default
    if endpoint_url is None:
        endpoint_url = os.getenv(
            "RUNPOD_ENDPOINT_URL",
            "https://pov3ewvy1mejeo.api.runpod.ai"
        )
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("RUNPOD_API_KEY")
    
    # Remove trailing slash
    endpoint_url = endpoint_url.rstrip("/")
    
    print(f"🌐 Testing RunPod endpoint: {endpoint_url}")
    if api_key:
        print(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
    print("=" * 80)
    
    # Health check first (handle cold start)
    if not health_check(endpoint_url, api_key):
        print("❌ Endpoint is not ready. Please try again later.")
        return
    
    # Build request payload
    print("\n📦 Building sample request...")
    payload = build_sample_request()
    
    # Prepare API request with authentication headers
    api_url = f"{endpoint_url}/api/generate"
    headers = get_auth_headers(api_key)
    
    print(f"\n🚀 Sending request to {api_url}...")
    print(f"   Prompt: {payload['prompt']}")
    print(f"   Task type: {payload['task_type']}")
    print(f"   Inference steps: {payload['num_inference_steps']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=600,  # 10 minutes timeout for generation
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📡 Response received in {elapsed_time:.2f}s")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Generation successful!")
            print(f"   Generation time: {result.get('generation_time', 'N/A')}s")
            print(f"   Model used: {result.get('model_used', 'N/A')}")
            print(f"   Request ID: {result.get('request_id', 'N/A')}")
            
            params = result.get("parameters_used", {})
            if params:
                print(f"   Task type: {params.get('task_type', 'N/A')}")
                print(f"   Steps: {params.get('num_inference_steps', 'N/A')}")
                print(f"   CFG scale: {params.get('guidance_scale', 'N/A')}")
            
            # Check if image is in response
            if result.get("image"):
                image_b64 = result["image"]
                print(f"   Image size: {len(image_b64)} characters (base64)")
                print("   ✅ Image generated successfully!")
            else:
                print("   ⚠️  No image in response")
            
            # Debug path if available
            if result.get("debug_path"):
                print(f"   Debug path: {result['debug_path']}")
        elif response.status_code == 401:
            print("\n❌ Authentication failed (401 Unauthorized)")
            print("   This endpoint requires a RunPod API key.")
            print("   Set RUNPOD_API_KEY environment variable or pass --api-key argument")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data.get('error', error_data.get('detail', 'Unknown error'))}")
            except (ValueError, KeyError):
                print(f"   Response: {response.text[:200]}")
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', error_data.get('detail', 'Unknown error'))}")
                if error_data.get("error_type"):
                    print(f"   Error type: {error_data['error_type']}")
            except (ValueError, KeyError):
                print(f"   Response: {response.text[:200]}")
    
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout (exceeded 10 minutes)")
        print("   The generation may be taking too long or the endpoint is overloaded.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    # Load .env file if exists
    env_path = Path(__file__).parent / ".env"
    load_env_file(env_path)
    
    parser = argparse.ArgumentParser(description="Test RunPod endpoint for image generation")
    parser.add_argument(
        "endpoint_url",
        nargs="?",
        default=None,
        help="RunPod endpoint URL (defaults to RUNPOD_ENDPOINT_URL env var or default endpoint)"
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="RunPod API key (can also be set via RUNPOD_API_KEY env var or .env file)"
    )
    
    args = parser.parse_args()
    
    test_runpod_endpoint(args.endpoint_url, args.api_key)


