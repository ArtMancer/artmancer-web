from __future__ import annotations

import logging
from typing import Any, Dict

import modal
from pydantic import ValidationError

from app.services.generation_service import GenerationService
from app.models.schemas import GenerationRequest

logger = logging.getLogger(__name__)

app = modal.App("artmancer-qwen")


# Image cho Modal: cài các thư viện cần thiết cho Qwen + LoRA bằng uv_pip_install
# và copy code Python local (package `app`) vào container để import được.
base_image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        # Web / Modal
        "fastapi[standard]>=0.115.0",
        "modal>=1.2.4",
        # Core ML stack
        "torch>=2.8.0",
        "torchvision>=0.23.0",
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "accelerate>=1.10.1",
        "safetensors>=0.4.0",
        "rembg>=2.0.68",
        # Phụ trợ
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.6.0",
        "python-dotenv>=1.0.0",
        "huggingface-hub[cli]>=0.35.1",
        "requests>=2.31.0",
        "hf-xet>=1.1.10",
        "torchmetrics>=1.0.0",
        "scikit-image>=0.21.0",
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.11.0.86",
        "pandas>=2.3.3",
    )
    # Đưa source code package `app` (trong thư mục server) vào image để import được trong container
    .add_local_python_source("app")
)

_service: GenerationService | None = None


def _get_service() -> GenerationService:
    """Khởi tạo và cache GenerationService cho worker Modal."""
    global _service
    if _service is None:
        logger.info("🔧 Khởi tạo GenerationService bên trong Modal worker")
        _service = GenerationService()
    return _service


@app.function(
    image=base_image,
    gpu="A10G",  # dùng A10G để tiết kiệm chi phí hơn A100
    timeout=600,
    min_containers=1,
)
@modal.fastapi_endpoint(method="POST")
def generate(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modal HTTP endpoint cho Qwen:

    - Input JSON: giống schema GenerationRequest (prompt, input_image, mask_image, reference_image, ...)
    - Output JSON: giống GenerationResponse (success, image base64, generation_time, model_used, parameters_used, request_id)
    """
    try:
        payload = body or {}
        if not isinstance(payload, dict):
            raise TypeError("Request body phải là JSON object.")

        request = GenerationRequest.model_validate(payload)
        service = _get_service()
        result = service.generate(request)

        # Đảm bảo luôn có success
        if "success" not in result:
            result["success"] = True

        return result
    except ValidationError as exc:
        logger.warning("Validation error trong Modal generate endpoint: %s", exc)
        return {
            "success": False,
            "error_type": "validation_error",
            "errors": exc.errors(),
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Lỗi không mong đợi trong Modal generate endpoint")
        return {
            "success": False,
            "error_type": "runtime_error",
            "error": str(exc),
        }


