"""
Async generation endpoints for job-based image generation.
Used by JobManagerService (CPU-only).
"""
from __future__ import annotations

import uuid
import time
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from typing import Dict, Any, List
from pathlib import Path
import base64
import io

from PIL import Image

from ...models import GenerationRequest
from ._helpers import (
    get_modal_imports,
    get_job_state_dictio,
    create_job_state,
    update_job_state_error,
    generate_sse_events,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generate", tags=["generation-async"])


@router.post("/async")
def submit_generation_job(request: GenerationRequest) -> Dict[str, Any]:
    """
    Submit async generation job.
    
    Returns task_id immediately, generation runs in background on A100.
    Use /api/generate/status/{task_id} to poll status.
    """
    worker, job_dictio, is_modal = get_modal_imports()
    
    if not is_modal or worker is None or job_dictio is None:
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment)"
        )
    
    task_id = str(uuid.uuid4())
    payload = request.model_dump()
    
    try:
        create_job_state(task_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Job state storage not available: {str(e)}"
        )
    
    # Spawn A100 worker function asynchronously
    try:
        worker.spawn(task_id, payload)
    except Exception as e:
        update_job_state_error(task_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn generation job: {str(e)}"
        )

    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Generation job submitted successfully"
    }


@router.get("/status/{task_id}")
def get_generation_status(task_id: str) -> Dict[str, Any]:
    """
    Get generation job status.
    
    Returns current status, progress, and error if any.
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_dictio[task_id]
    return {
        "task_id": task_id,
        "status": state.get("status", "unknown"),
        "progress": state.get("progress", 0.0),
        "error": state.get("error"),
    }


@router.get("/result/{task_id}")
def get_generation_result(task_id: str) -> Dict[str, Any]:
    """
    Get generation job result.
    
    Returns result only if status is "done".
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_dictio[task_id]
    status = state.get("status", "unknown")
    
    if status == "error":
        raise HTTPException(
            status_code=500,
            detail=state.get("error", "Generation failed")
        )
    
    if status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed yet. Current status: {status}"
        )
    
    result = state.get("result")
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Result not available"
        )
    
    return {
        "task_id": task_id,
        "status": "done",
        "image": result,
        "debug_info": state.get("debug_info"),
    }


@router.get("/result-image/{task_id}")
def get_generation_result_image(task_id: str):
    """
    Trả về ảnh kết quả dạng binary (ưu tiên WebP) cho Qwen Image Edit.
    
    - Đọc base64 ảnh đang lưu trong job_state_dictio
    - Decode và convert sang WebP (nếu có thể)
    - Trả về raw bytes với Content-Type: image/webp
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    state = job_dictio[task_id]
    status = state.get("status", "unknown")
    
    if status == "error":
        raise HTTPException(
            status_code=500,
            detail=state.get("error", "Generation failed")
        )
    
    if status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed yet. Current status: {status}"
        )
    
    result_b64 = state.get("result")
    if not result_b64:
        raise HTTPException(
            status_code=500,
            detail="Result image not available"
        )
    
    try:
        # Decode base64 từ worker (thường là PNG)
        image_bytes = base64.b64decode(result_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Optional: Resize if image is extremely large (>2560 in any dimension)
        # This rarely happens but provides extra safety
        max_dim = max(image.width, image.height)
        if max_dim > 2560:
            scale = 2560 / max_dim
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"🔽 Resized large image from {max_dim}px to {max(new_size)}px")
        
        # Encode sang WebP với optimization tối đa
        # Quality=80: Excellent visual quality, 40-50% smaller than quality=90
        # method=6: Best compression (slowest but smallest file)
        # exact=False: Allow encoder flexibility for better compression
        output = io.BytesIO()
        image.save(
            output, 
            format="WEBP", 
            quality=80, 
            method=6,
            exact=False,
        )
        output.seek(0)
        
        # Add cache headers for better performance
        return Response(
            content=output.getvalue(),
            media_type="image/webp",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",  # Cache 1 year
            }
        )
    except Exception as e:  # noqa: BLE001
        # Fallback: trả về PNG raw nếu convert WebP lỗi
        try:
            return Response(
                content=base64.b64decode(result_b64),
                media_type="image/png",
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to decode generation result: {e}"
            ) from e


@router.post("/async/cancel/{task_id}")
def cancel_async_generation(task_id: str) -> Dict[str, Any]:
    """
    Cancel an async generation job.
    
    Sets cancellation flag in job_state_dictio, which will be checked
    by the A100 worker during generation.
    
    Args:
        task_id: Task identifier from async job submission
    
    Returns:
        Success message
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    # Update job state to mark as cancelled
    state = job_dictio.get(task_id, {})
    state["cancelled"] = True
    state["status"] = "cancelled"
    job_dictio[task_id] = state
    
    return {
        "success": True,
        "message": f"Async generation task {task_id} marked for cancellation",
        "task_id": task_id
    }


@router.post("/multi-steps")
def generate_multi_steps(request: GenerationRequest) -> Dict[str, Any]:
    """
    Run generation sequentially for multiple inference steps (default: 5, 10, 25).
    Executes on GPU via qwen_worker.call (synchronous). Saves each image to Volume and returns base64 + path.
    """
    worker, _, is_modal = get_modal_imports()
    
    if not is_modal or worker is None:
        raise HTTPException(
            status_code=501,
            detail="Async generation not available (not running in Modal environment)"
        )

    # Get VOL_MOUNT_PATH
    from ._helpers import get_vol_mount_path
    vol_mount_path = get_vol_mount_path()
    
    steps: List[int] = [5, 10, 25]
    base_request_id = str(uuid.uuid4())
    results: List[Dict[str, Any]] = []

    # Ensure volume mount path exists
    save_root = Path(vol_mount_path) / "reports" / base_request_id
    save_root.mkdir(parents=True, exist_ok=True)

    for step in steps:
        # Override num_inference_steps per call
        payload = request.model_copy(update={"num_inference_steps": step}).model_dump()
        task_id = f"{base_request_id}_s{step}"

        try:
            # Call GPU worker synchronously
            result = worker.call(task_id, payload)
            image_b64 = result.get("image")
            if not image_b64:
                raise ValueError("Worker returned no image")

            # Save to volume
            save_path = save_root / f"step_{step}.png"
            try:
                raw = base64.b64decode(image_b64)
                with open(save_path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                # Continue even if save fails; report error in path field
                save_path = None
                result["save_error"] = str(e)

            results.append({
                    "step": step,
                    "image": image_b64,
                    "path": str(save_path) if save_path else None,
                    "request_id": result.get("request_id"),
                    "debug_info": result.get("debug_info"),
            })
        except Exception as e:
            results.append({
                    "step": step,
                    "error": str(e),
            })

    return {
        "base_request_id": base_request_id,
        "steps": steps,
        "results": results,
    }


@router.get("/stream/{task_id}")
async def stream_generation_progress(task_id: str):
    """
    Stream generation progress using Server-Sent Events (SSE).
    
    Returns real-time progress updates including:
    - status: queued, processing, done, error
    - progress: 0.0 to 1.0
    - current_step: current inference step
    - total_steps: total inference steps
    - error: error message if any
    - loading_message: loading stage message
    """
    return StreamingResponse(
        generate_sse_events(task_id, include_heartbeat=True, heartbeat_interval=5.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/stream")
async def stream_generation_direct(request: GenerationRequest):
    """
    Submit a generation job and stream progress using Server-Sent Events (SSE).

    This provides a single POST endpoint that:
    1) Submits the async job to H200 worker
    2) Streams job_state_dictio updates as SSE events until completion
    """
    # Reuse existing async submission logic to ensure consistent job_state handling
    job_info = submit_generation_job(request)
    task_id = job_info["task_id"]

    return StreamingResponse(
        generate_sse_events(task_id, include_heartbeat=True, heartbeat_interval=5.0),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
