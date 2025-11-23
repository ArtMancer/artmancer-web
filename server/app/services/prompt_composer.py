"""
Prompt Composition System for Qwen Image Edit
Supports angle macros and background presets for better prompt generation.
Only used for insertion and removal tasks, not for white-balance.
"""

from typing import Optional

# Angle macros: Map English labels to Chinese commands that LoRA understands
ANGLE_MACROS = {
    "wide-angle": "广角镜头",
    "top-down": "俯视角度",
    "side-view": "侧面视角",
    "front-view": "正面视角",
    "close-up": "特写镜头",
    "low-angle": "低角度",
    "high-angle": "高角度",
    "bird-eye": "鸟瞰视角",
    "eye-level": "平视角度",
    "dutch-angle": "倾斜角度",
}

# Background presets: Short English fragments for common backgrounds
BACKGROUND_PRESETS = {
    "marble-surface": "marble surface",
    "soft-gray-studio": "soft gray studio",
    "white-background": "white background",
    "black-background": "black background",
    "wood-texture": "wood texture",
    "concrete-wall": "concrete wall",
    "outdoor-natural": "outdoor natural setting",
    "indoor-studio": "indoor studio lighting",
    "minimalist": "minimalist clean background",
    "textured": "textured background",
}


def compose_prompt(
    base_prompt: str,
    angle: Optional[str] = None,
    background_preset: Optional[str] = None,
    style_notes: Optional[str] = None,
) -> str:
    """
    Compose a prompt from base prompt, angle macro, background preset, and style notes.
    
    Uses " | " separator to keep clauses visually distinct while Qwen-Edit handles
    punctuation and yields consistent parsing without biasing toward the last clause.
    
    Args:
        base_prompt: The main prompt describing what to edit
        angle: Optional angle macro label (e.g., "wide-angle", "top-down")
        background_preset: Optional background preset name (e.g., "marble-surface")
        style_notes: Optional additional style notes
    
    Returns:
        Composed prompt string
    """
    parts = []
    
    # Add base prompt first (required)
    if base_prompt and base_prompt.strip():
        parts.append(base_prompt.strip())
    
    # Add angle macro if provided
    if angle and angle.strip():
        angle_lower = angle.strip().lower()
        if angle_lower in ANGLE_MACROS:
            parts.append(ANGLE_MACROS[angle_lower])
        else:
            # If not in macros, use as-is (might be custom angle)
            parts.append(angle.strip())
    
    # Add background preset if provided
    if background_preset and background_preset.strip():
        preset_lower = background_preset.strip().lower()
        if preset_lower in BACKGROUND_PRESETS:
            parts.append(BACKGROUND_PRESETS[preset_lower])
        else:
            # If not in presets, use as-is (might be custom background)
            parts.append(background_preset.strip())
    
    # Add style notes if provided
    if style_notes and style_notes.strip():
        parts.append(style_notes.strip())
    
    # Join with " | " separator
    if not parts:
        return ""  # Empty prompt if no parts
    
    return " | ".join(parts)


def get_available_angles() -> list[str]:
    """Get list of available angle macro labels."""
    return list(ANGLE_MACROS.keys())


def get_available_background_presets() -> list[str]:
    """Get list of available background preset names."""
    return list(BACKGROUND_PRESETS.keys())

