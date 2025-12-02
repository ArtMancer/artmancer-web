"""ArtMancer backend package (core services + models for Qwen)."""

from __future__ import annotations

import importlib.machinery
import sys
import types

# Một số môi trường local có cài nhầm gói `triton` không tương thích với Torch/Diffusers,
# dẫn đến lỗi:
#   - module 'triton' has no attribute 'language'
#   - ValueError: triton.__spec__ is None
#
# Để tránh làm hỏng môi trường production, shim này chỉ chạy khi:
#   - Không có module `triton` trong sys.modules, HOẶC
#   - Module hiện tại không có thuộc tính `language`.
existing_triton = sys.modules.get("triton")
if existing_triton is None or not hasattr(existing_triton, "language"):
    triton_stub = types.ModuleType("triton")

    class _TritonLanguage:
        # Thuộc tính `dtype` chỉ cần tồn tại để Torch/Diffusers không crash;
        # ta không dùng Triton compiler trong flow inference này.
        dtype = object()

    triton_stub.language = _TritonLanguage()  # type: ignore[attr-defined]
    # Thiết lập __spec__ để importlib.util.find_spec("triton") không raise ValueError
    triton_stub.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)  # type: ignore[assignment]
    sys.modules["triton"] = triton_stub

__all__: list[str] = []

