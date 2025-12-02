from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

from app.models.schemas import GenerationRequest
from app.services.generation_service import GenerationService


def _load_image_base64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def build_sample_request() -> Dict[str, Any]:
    """
    Tạo payload mẫu giống GenerationRequest để test trực tiếp GenerationService.
    """
    # TODO: sửa lại các path này cho phù hợp với máy bạn
    input_image_b64 = _load_image_base64("dataset/image.png")
    mask_image_b64 = _load_image_base64("dataset/condition-1.png")

    # conditional_images: phần tử đầu tiên là mask, các phần tử sau (nếu có) là extra conditionals
    conditional_images = [mask_image_b64]

    return {
        "prompt": "yellow rubber duck on white table, photorealistic",
        "input_image": input_image_b64,
        "num_inference_steps": 25,
        "guidance_scale": 4.0,
        "true_cfg_scale": 3.3,
        "task_type": "object-insert",  # hoặc "object-removal" / "white-balance"
        "seed": 42,
        "input_quality": "medium",
        "conditional_images": conditional_images,
    }


def main() -> None:
    payload = build_sample_request()
    service = GenerationService()
    request = GenerationRequest.model_validate(payload)
    result = service.generate(request)
    print("Result keys:", list(result.keys()))
    if not result.get("success", False):
        print("Error:", result)
    else:
        print(
            "Success, generation_time:",
            result.get("generation_time"),
            "model_used:",
            result.get("model_used"),
            "task_type:",
            result.get("parameters_used", {}).get("task_type"),
        )


if __name__ == "__main__":
    main()


