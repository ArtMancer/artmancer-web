# Deprecated placeholder kept for backward compatibility.
# Actual configuration is handled in app/core/config.py
"""
Model configuration presets for Qwen Image Edit pipeline
"""

# Model presets with optimized settings for different scenarios
MODEL_PRESETS = {
    "creative": {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "true_cfg_scale": 4.0,
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "description": "High creativity, varied outputs",
    },
    "balanced": {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "true_cfg_scale": 3.5,
        "num_inference_steps": 35,
        "guidance_scale": 1.0,
        "description": "Balanced creativity and quality",
    },
    "precise": {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "true_cfg_scale": 2.5,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "description": "More controlled, precise edits",
    },
    "artistic": {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "true_cfg_scale": 5.0,
        "num_inference_steps": 50,
        "guidance_scale": 1.2,
        "description": "Maximum creativity for artistic works",
    },
    "fast": {
        "model": "Qwen/Qwen-Image-Edit-2509",
        "true_cfg_scale": 3.0,
        "num_inference_steps": 20,
        "guidance_scale": 1.0,
        "description": "Fast generation with good quality",
    },
}

# Available Qwen models
AVAILABLE_MODELS = [
    "Qwen/Qwen-Image-Edit-2509",
    "Qwen/Qwen-Image-Edit",
]

# Default settings
DEFAULT_SETTINGS = {
    "model": "Qwen/Qwen-Image-Edit-2509",
    "true_cfg_scale": 4.0,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "negative_prompt": "",
    "num_images_per_prompt": 1,
}

# Model-specific capabilities and limits
MODEL_INFO = {
    "Qwen/Qwen-Image-Edit-2509": {
        "supports_image_editing": True,
        "supports_image_generation": True,
        "supports_multi_image": True,
        "max_inference_steps": 100,
        "min_inference_steps": 10,
        "best_for": ["multi-image editing", "composition", "style transfer", "object modification"],
        "requires_input_image": True,
        "output_format": "PIL.Image",
        "version": "2509",
        "recommended_settings": {
            "true_cfg_scale": 4.0,
            "num_inference_steps": 40,
            "guidance_scale": 1.0
        }
    },
    "Qwen/Qwen-Image-Edit": {
        "supports_image_editing": True,
        "supports_image_generation": True,
        "supports_multi_image": False,
        "max_inference_steps": 100,
        "min_inference_steps": 10,
        "best_for": ["single image editing", "style transfer", "object modification", "background changes"],
        "requires_input_image": True,
        "output_format": "PIL.Image",
        "version": "original",
        "recommended_settings": {
            "true_cfg_scale": 4.0,
            "num_inference_steps": 50
        }
    },
}
