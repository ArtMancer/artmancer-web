"""
Model configuration presets for different use cases
"""

# Model presets with optimized settings for different scenarios
MODEL_PRESETS = {
    "creative": {
        "model": "gemini-2.5-flash-image-preview",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "description": "High creativity, varied outputs",
    },
    "balanced": {
        "model": "gemini-2.5-flash-image-preview",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "description": "Balanced creativity and consistency",
    },
    "precise": {
        "model": "gemini-2.5-flash-image-preview",
        "temperature": 0.3,
        "top_p": 0.6,
        "top_k": 10,
        "description": "More consistent, less random outputs",
    },
    "artistic": {
        "model": "gemini-2.5-flash-image-preview",
        "temperature": 1.2,
        "top_p": 0.9,
        "top_k": 50,
        "description": "Maximum creativity for artistic works",
    },
    "technical": {
        "model": "gemini-1.5-pro",
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": 5,
        "description": "Technical accuracy focused",
    },
}

# Available Gemini models
AVAILABLE_MODELS = [
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Default settings
DEFAULT_SETTINGS = {
    "model": "gemini-2.5-flash-image-preview",
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 2048,
}

# Model-specific capabilities and limits
MODEL_INFO = {
    "gemini-2.5-flash-image-preview": {
        "supports_image_generation": True,
        "max_tokens": 8192,
        "best_for": ["image generation", "creative tasks", "multimodal content"],
    },
    "gemini-2.0-flash-thinking-exp-1219": {
        "supports_image_generation": False,
        "max_tokens": 32768,
        "best_for": ["complex reasoning", "analysis", "step-by-step thinking"],
    },
    "gemini-2.0-flash-exp": {
        "supports_image_generation": True,
        "max_tokens": 8192,
        "best_for": ["experimental features", "latest capabilities"],
    },
    "gemini-1.5-flash": {
        "supports_image_generation": False,
        "max_tokens": 1048576,
        "best_for": ["long context", "document analysis", "fast responses"],
    },
    "gemini-1.5-pro": {
        "supports_image_generation": False,
        "max_tokens": 2097152,
        "best_for": ["complex tasks", "reasoning", "analysis"],
    },
}
