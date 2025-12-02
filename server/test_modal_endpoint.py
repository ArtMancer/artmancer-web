"""Script minh hoạ cách gọi HTTP endpoint Modal từ local bằng requests."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

import requests


def _load_image_base64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def build_request_body() -> Dict[str, Any]:
    # TODO: đổi path ảnh cho phù hợp
    input_image_b64 = _load_image_base64("dataset/image.png")
    mask_image_b64 = _load_image_base64("dataset/condition-1.png")

    # conditional_images: phần tử đầu tiên là mask, các phần tử sau (nếu có) là extra conditionals
    conditional_images: list[str] = [mask_image_b64]

    return {
        "prompt": "yellow rubber duck on white table, photorealistic",
        "input_image": input_image_b64,
        "num_inference_steps": 25,
        "guidance_scale": 4.0,
        "true_cfg_scale": 3.3,
        "task_type": "object-insert",  # hoặc object-removal / white-balance
        "seed": 42,
        "input_quality": "medium",
        "conditional_images": conditional_images,
    }


def main() -> None:
    modal_url = "https://nxan2911--artmancer-qwen-generate.modal.run/"

    body = build_request_body()
    resp = requests.post(
        modal_url,
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        timeout=600,
    )

    print("Status:", resp.status_code)
    try:
        data = resp.json()
    except Exception:
        print("Raw response:", resp.text)
        return

    print("Keys:", list(data.keys()))
    if not data.get("success", False):
        print("Error:", data)
    else:
        print(
            "Success, generation_time:",
            data.get("generation_time"),
            "model_used:",
            data.get("model_used"),
        )


if __name__ == "__main__":
    main()


