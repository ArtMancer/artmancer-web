"""Debug endpoints for viewing and managing debug sessions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ...services.debug_service import debug_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.get("/sessions")
def list_debug_sessions() -> Dict[str, Any]:
    """List all debug sessions."""
    if not debug_service.enabled:
        return {"enabled": False, "sessions": []}
    
    try:
        base_dir = debug_service.base_dir
        if not base_dir.exists():
            return {"enabled": True, "sessions": []}
        
        sessions = []
        for session_dir in sorted(base_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not session_dir.is_dir():
                continue
            
            # Read metadata if available
            metadata_path = session_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            
            sessions.append({
                "session_name": session_dir.name,
                "created_at": metadata.get("created_at", ""),
                "success": metadata.get("success"),
                "task_type": metadata.get("lora_info", {}).get("task_type"),
                "generation_time": metadata.get("parameters", {}).get("generation_time"),
                "num_images": len(metadata.get("images", {})),
            })
        
        return {
            "enabled": True,
            "sessions": sessions,
            "total": len(sessions),
        }
    
    except Exception as exc:
        logger.exception("Failed to list debug sessions")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/{session_name}")
def get_debug_session(session_name: str) -> Dict[str, Any]:
    """Get details of a specific debug session."""
    if not debug_service.enabled:
        raise HTTPException(status_code=404, detail="Debug service is not enabled")
    
    try:
        session_dir = debug_service.base_dir / session_name
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_name} not found")
        
        # Read metadata
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail=f"Metadata for session {session_name} not found")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Read LoRA log
        lora_log_path = session_dir / "lora_debug.log"
        lora_log = ""
        if lora_log_path.exists():
            with open(lora_log_path, "r", encoding="utf-8") as f:
                lora_log = f.read()
        
        # List all image files
        image_files = [f.name for f in session_dir.iterdir() if f.suffix == ".png"]
        
        return {
            "session_name": session_name,
            "metadata": metadata,
            "lora_log": lora_log,
            "image_files": sorted(image_files),
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get debug session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/{session_name}/images/{image_name}")
def get_debug_image(session_name: str, image_name: str):
    """Get a specific debug image."""
    if not debug_service.enabled:
        raise HTTPException(status_code=404, detail="Debug service is not enabled")
    
    try:
        image_path = debug_service.base_dir / session_name / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image {image_name} not found in session {session_name}")
        
        return FileResponse(image_path, media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get debug image")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cleanup")
def cleanup_debug_sessions(keep_last_n: int = 50) -> Dict[str, Any]:
    """Clean up old debug sessions, keeping only the most recent N."""
    if not debug_service.enabled:
        return {"enabled": False, "deleted": 0}
    
    try:
        deleted = debug_service.cleanup_old_sessions(keep_last_n=keep_last_n)
        return {
            "enabled": True,
            "deleted": deleted,
            "kept": keep_last_n,
        }
    except Exception as exc:
        logger.exception("Failed to cleanup debug sessions")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def get_debug_status() -> Dict[str, Any]:
    """Get debug service status."""
    status = {
        "enabled": debug_service.enabled,
        "base_dir": str(debug_service.base_dir) if debug_service.enabled else None,
    }
    
    if debug_service.enabled and debug_service.base_dir.exists():
        sessions = list(debug_service.base_dir.iterdir())
        status["total_sessions"] = len([s for s in sessions if s.is_dir()])
        
        # Calculate total disk usage
        total_size = sum(
            f.stat().st_size
            for session_dir in sessions
            if session_dir.is_dir()
            for f in session_dir.rglob("*")
            if f.is_file()
        )
        status["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return status

