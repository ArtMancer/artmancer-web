"""
Shared helper functions for API endpoints.

This module provides common utilities to reduce code duplication across endpoints:
- Modal app imports (lazy loading)
- Job state management
- SSE event generation
"""
from __future__ import annotations

import sys
import time
import json
import logging
from typing import Dict, Any, Optional, AsyncIterator, Tuple

logger = logging.getLogger(__name__)

# Global cache for Modal imports (lazy loading)
_MODAL_CACHE: Dict[str, Any] = {
    "qwen_worker": None,
    "job_state_dictio": None,
    "volume": None,
    "VOL_MOUNT_PATH": "/checkpoints",
    "MODAL_ENV": False,
}


def get_modal_imports() -> Tuple[Any, Optional[Dict[str, Any]], bool]:
    """
    Lazy import of Modal components.
    
    Returns:
        Tuple of (qwen_worker, job_state_dictio, is_modal_env)
    """
    if _MODAL_CACHE["qwen_worker"] is None:
        try:
            if "modal_app" not in sys.modules:
                import modal_app  # type: ignore
            else:
                modal_app = sys.modules["modal_app"]
            
            _MODAL_CACHE["qwen_worker"] = getattr(modal_app, "qwen_worker", None)
            _MODAL_CACHE["job_state_dictio"] = getattr(modal_app, "job_state_dictio", None)
            _MODAL_CACHE["volume"] = getattr(modal_app, "volume", None)
            _MODAL_CACHE["VOL_MOUNT_PATH"] = getattr(
                modal_app, "VOL_MOUNT_PATH", _MODAL_CACHE["VOL_MOUNT_PATH"]
            )
            _MODAL_CACHE["MODAL_ENV"] = (
                _MODAL_CACHE["qwen_worker"] is not None
                and _MODAL_CACHE["job_state_dictio"] is not None
            )
        except (ImportError, AttributeError):
            _MODAL_CACHE["MODAL_ENV"] = False
            _MODAL_CACHE["qwen_worker"] = None
            _MODAL_CACHE["job_state_dictio"] = None
            _MODAL_CACHE["volume"] = None
    
    return (
        _MODAL_CACHE["qwen_worker"],
        _MODAL_CACHE["job_state_dictio"],
        _MODAL_CACHE["MODAL_ENV"],
    )


def get_vol_mount_path() -> str:
    """
    Get Modal volume mount path.
    
    Returns:
        Volume mount path string (default: "/checkpoints")
    """
    get_modal_imports()  # Ensure cache is populated
    return _MODAL_CACHE["VOL_MOUNT_PATH"]


def get_job_state_dictio() -> Dict[str, Any]:
    """
    Get job state dictionary, raising HTTPException if not available.
    
    Raises:
        HTTPException: If job state storage is not available
    """
    _, job_dictio, _ = get_modal_imports()
    if job_dictio is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail="Job state storage not available"
        )
    return job_dictio


def create_job_state(task_id: str) -> None:
    """
    Initialize job state in job_state_dictio.
    
    Args:
        task_id: Unique task identifier
    """
    job_dictio = get_job_state_dictio()
    job_dictio[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_step": None,
        "total_steps": None,
        "error": None,
        "result": None,
        "debug_info": None,
        "created_at": time.time(),
    }
    
    # Add to tracking list (non-critical, ignore errors)
    tracking_key = "__job_tracking_list__"
    try:
        job_ids = job_dictio.get(tracking_key, [])
        if task_id not in job_ids:
            job_ids.append(task_id)
            job_dictio[tracking_key] = job_ids
    except Exception:
        pass  # Tracking list might not be available


def update_job_state_error(task_id: str, error: str) -> None:
    """
    Update job state to error status.
    
    Args:
        task_id: Task identifier
        error: Error message
    """
    job_dictio = get_job_state_dictio()
    existing_state = job_dictio.get(task_id, {})
    job_dictio[task_id] = {
        "status": "error",
        "progress": 0.0,
        "error": error,
        "result": None,
        "debug_info": None,
        "created_at": existing_state.get("created_at", time.time()),
    }


async def generate_sse_events(
    task_id: str,
    include_heartbeat: bool = False,
    heartbeat_interval: float = 5.0,
) -> AsyncIterator[str]:
    """
    Generate Server-Sent Events (SSE) for job progress streaming.
    
    Args:
        task_id: Task identifier to track
        include_heartbeat: Whether to send heartbeat messages to keep connection alive
        heartbeat_interval: Interval between heartbeats in seconds
    
    Yields:
        SSE-formatted event strings
    """
    job_dictio = get_job_state_dictio()
    
    if task_id not in job_dictio:
        yield f"data: {json.dumps({'error': f'Task {task_id} not found'})}\n\n"
        return
    
    last_step = -1
    last_status: Optional[str] = None
    last_loading_message: Optional[str] = None
    first_message = True
    last_heartbeat = time.time() if include_heartbeat else None
    
    import asyncio
    
    while True:
        try:
            state = job_dictio.get(task_id)
            if not state:
                yield f"data: {json.dumps({'error': f'Task {task_id} not found'})}\n\n"
                break
            
            status = state.get("status", "unknown")
            current_step = state.get("current_step")
            total_steps = state.get("total_steps")
            progress = state.get("progress", 0.0)
            error = state.get("error")
            loading_message = state.get("loading_message")
            
            # Determine if we should send an event
            should_send = (
                first_message
                or status != last_status
                or (current_step is not None and current_step != last_step)
                or (loading_message is not None and loading_message != last_loading_message)
            )
            
            # Heartbeat logic: send if no updates for a while (keep connection alive)
            if include_heartbeat and last_heartbeat is not None:
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    should_send = True
                    last_heartbeat = current_time
            
            if should_send:
                event_data: Dict[str, Any] = {
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "error": error,
                    "loading_message": loading_message,
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Update tracking variables
                last_step = current_step if current_step is not None else last_step
                last_status = status
                last_loading_message = loading_message
                first_message = False
            
            # Stop if done or error - send final message with result if available
            if status in ("done", "error", "cancelled"):
                final_event_data: Dict[str, Any] = {
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "error": error,
                    "loading_message": loading_message,
                }
                # Include result if done
                if status == "done" and state.get("result"):
                    final_event_data["result"] = state.get("result")
                yield f"data: {json.dumps(final_event_data)}\n\n"
                logger.info(f"✅ [SSE] Sent final status for task {task_id}: {status}")
                break
            
            # Poll every 200ms for updates
            await asyncio.sleep(0.2)
        
        except asyncio.CancelledError:
            # Connection cancelled - try to send final status before closing
            logger.warning(f"⚠️ [SSE] Connection cancelled for task {task_id}, sending final status...")
            try:
                final_state = job_dictio.get(task_id)
                if final_state:
                    final_status = final_state.get("status")
                    if final_status in ("done", "error", "cancelled"):
                        final_event_data = {
                            "task_id": task_id,
                            "status": final_status,
                            "progress": final_state.get("progress", 1.0),
                            "error": final_state.get("error"),
                            "result": final_state.get("result") if final_status == "done" else None,
                        }
                        yield f"data: {json.dumps(final_event_data)}\n\n"
                        logger.info(f"✅ [SSE] Sent final status on cancellation: {final_status}")
            except Exception as final_exc:
                logger.error(f"Failed to send final status on cancellation: {final_exc}")
            raise
        
        except Exception as loop_exc:
            logger.error(f"Error in SSE event loop for task {task_id}: {loop_exc}", exc_info=True)
            yield f"data: {json.dumps({'error': str(loop_exc), 'recoverable': True})}\n\n"
            # Continue loop to try to recover
            await asyncio.sleep(1.0)

