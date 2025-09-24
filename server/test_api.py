"""
Example usage of the ArtMancer Web API with presets and custom settings
"""

import requests
import base64
from pathlib import Path
import time

# API Configuration
API_BASE = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{API_BASE}/api/health")
    print("Health Check:", response.json())
    return response.status_code == 200


def test_models():
    """Get available models with capabilities"""
    response = requests.get(f"{API_BASE}/api/models")
    result = response.json()
    print("Available Models:")
    for model in result.get("models", []):
        print(f"  - {model['name']}: {model['info'].get('best_for', [])}")
    return result


def test_presets():
    """Get available presets"""
    response = requests.get(f"{API_BASE}/api/presets")
    result = response.json()
    print("Available Presets:")
    for name, preset in result.get("presets", {}).items():
        print(f"  - {name}: {preset.get('description', '')}")
    return result


def test_config():
    """Get API configuration"""
    response = requests.get(f"{API_BASE}/api/config")
    result = response.json()
    print("API Config:")
    print(f"  - Supported formats: {result['config']['available_formats']}")
    print(f"  - Available presets: {result['config']['available_presets']}")
    print(f"  - Default model: {result['config']['default_settings']['model']}")
    return result


def generate_image(prompt, settings=None, return_format="base64"):
    """Generate an image with optional custom settings"""

    payload = {"prompt": prompt, "return_format": return_format}

    if settings:
        payload["settings"] = settings

    response = requests.post(
        f"{API_BASE}/api/generate",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Generation successful!")
        print(f"Model used: {result['model_used']}")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print(f"Settings used: {result['settings_used']}")

        if result.get("image_base64"):
            # Save the image
            image_data = base64.b64decode(result["image_base64"])
            timestamp = int(time.time())
            output_path = Path(f"generated_{timestamp}.png")
            output_path.write_bytes(image_data)
            print(f"Image saved to: {output_path}")

        return result
    else:
        print(f"❌ Generation failed: {response.status_code}")
        print(response.json())
        return None


def generate_with_preset(prompt, preset_name, return_format="base64"):
    """Generate image using a predefined preset"""

    payload = {"prompt": prompt, "return_format": return_format}

    response = requests.post(
        f"{API_BASE}/api/generate/preset/{preset_name}",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Preset generation successful!")
        print(f"Preset used: {result.get('preset_used', 'N/A')}")
        print(f"Description: {result.get('preset_description', 'N/A')}")
        print(f"Model used: {result['model_used']}")
        print(f"Generation time: {result['generation_time']:.2f}s")

        if result.get("image_base64"):
            # Save the image
            image_data = base64.b64decode(result["image_base64"])
            timestamp = int(time.time())
            output_path = Path(f"preset_{preset_name}_{timestamp}.png")
            output_path.write_bytes(image_data)
            print(f"Image saved to: {output_path}")

        return result
    else:
        print(f"❌ Preset generation failed: {response.status_code}")
        print(response.json())
        return None


def example_basic_generation():
    """Basic image generation example"""
    print("\n=== Basic Generation Example ===")

    prompt = "Create a picture of a nano banana dish in a fancy restaurant with a Gemini theme"
    result = generate_image(prompt)
    return result


def example_preset_generation():
    """Preset-based generation examples"""
    print("\n=== Preset Generation Examples ===")

    prompt = "A futuristic nano banana smoothie with holographic garnish"

    # Try different presets
    presets = ["creative", "balanced", "artistic"]
    results = []

    for preset in presets:
        print(f"\n--- Using '{preset}' preset ---")
        result = generate_with_preset(prompt, preset)
        if result:
            results.append(result)
        time.sleep(1)  # Small delay between requests

    return results


def example_custom_settings():
    """Image generation with custom model settings"""
    print("\n=== Custom Settings Example ===")

    prompt = "A nano banana cocktail with glowing ice cubes and crystal decorations"

    custom_settings = {
        "model": "gemini-2.5-flash-image-preview",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_output_tokens": 1024,
    }

    result = generate_image(prompt, custom_settings)
    return result


def example_batch_generation():
    """Batch generation example with different presets"""
    print("\n=== Batch Generation Example ===")

    # Create requests with different presets applied manually
    requests_data = []

    prompts_and_presets = [
        ("Nano banana appetizer with gold leaf", "precise"),
        ("Nano banana dessert with crystal decorations", "artistic"),
        ("Nano banana cocktail with glowing ice cubes", "creative"),
    ]

    # Get preset settings and create requests
    presets_response = requests.get(f"{API_BASE}/api/presets")
    if presets_response.status_code == 200:
        available_presets = presets_response.json()["presets"]

        for prompt, preset_name in prompts_and_presets:
            if preset_name in available_presets:
                preset_settings = available_presets[preset_name].copy()
                preset_settings.pop("description", None)

                requests_data.append(
                    {
                        "prompt": prompt,
                        "settings": preset_settings,
                        "return_format": "base64",
                    }
                )

    if not requests_data:
        print("❌ Failed to prepare batch requests")
        return None

    response = requests.post(
        f"{API_BASE}/api/generate/batch",
        json=requests_data,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Batch generation completed!")
        print(f"Total requests: {result['total_requests']}")
        print(f"Batch ID: {result['batch_id']}")

        for item in result["results"]:
            if item["success"]:
                print(f"✅ Index {item['index']}: Success")
                # Save batch images
                if item["result"].get("image_base64"):
                    image_data = base64.b64decode(item["result"]["image_base64"])
                    timestamp = int(time.time())
                    output_path = Path(f"batch_{item['index']}_{timestamp}.png")
                    output_path.write_bytes(image_data)
                    print(f"   Image saved to: {output_path}")
            else:
                print(f"❌ Index {item['index']}: Failed - {item['error']}")

        return result
    else:
        print(f"❌ Batch generation failed: {response.status_code}")
        print(response.json())
        return None


if __name__ == "__main__":
    print("🚀 Starting ArtMancer API Tests")

    # Test API health
    if not test_health():
        print("❌ API is not healthy, please check if the server is running")
        exit(1)

    # Get models, presets, and config
    test_models()
    test_presets()
    test_config()

    # Run generation examples
    example_basic_generation()
    example_preset_generation()
    example_custom_settings()
    example_batch_generation()

    print("\n✅ All tests completed!")
    print("📁 Check the current directory for generated images!")
