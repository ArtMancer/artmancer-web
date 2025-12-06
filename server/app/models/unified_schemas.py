"""
Unified Request Schema for 3 Tasks
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UnifiedGenerationRequest(BaseModel):
    """Unified request schema for insertion, removal, white_balance tasks."""
    
    task: Literal["insertion", "removal", "white_balance"] = Field(
        ...,
        description="Task type: 'insertion', 'removal', or 'white_balance'"
    )
    
    prompt: str = Field(..., description="Description of the desired edit")
    
    images: Dict[str, str] = Field(
        ...,
        description=(
            "Image dictionary with keys:\n"
            "- 'original': Base64 encoded original/input image (required for all tasks)\n"
            "- 'mask': Base64 encoded mask image (required for insertion/removal)\n"
            "- 'ref_img': Base64 encoded reference image (required for insertion)\n"
            "- 'optional': Additional optional images"
        )
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional parameters:\n"
            "- 'canny_thresholds': [low, high] for white_balance (default: [100, 200])\n"
            "- 'enable_mae': bool for removal (default: True)\n"
            "- 'num_inference_steps': int (default: 10)\n"
            "- 'guidance_scale': float (default: 4.0)\n"
            "- 'true_cfg_scale': float (default: 4.0)\n"
            "- 'negative_prompt': str\n"
            "- 'seed': int\n"
            "- 'width': int\n"
            "- 'height': int"
        )
    )

