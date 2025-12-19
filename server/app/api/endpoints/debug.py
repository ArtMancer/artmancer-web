"""Debug endpoints for viewing and managing debug sessions."""

from __future__ import annotations

import json
import logging
import zipfile
import io
from typing import Any, Dict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from ...services.debug_service import debug_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["debug"])


def _read_metadata(session_dir: Path) -> Dict[str, Any]:
    """
    Read metadata.json from session directory.
    
    Args:
        session_dir: Path to session directory
    
    Returns:
        Metadata dictionary (empty dict if not found or invalid)
    """
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_lora_log(session_dir: Path) -> str:
    """
    Read lora_debug.log from session directory.
    
    Args:
        session_dir: Path to session directory
    
    Returns:
        Log content as string (empty if not found)
    """
    lora_log_path = session_dir / "lora_debug.log"
    if not lora_log_path.exists():
        return ""
    
    try:
        with open(lora_log_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


@router.get("/sessions")
async def list_debug_sessions() -> Dict[str, Any]:
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
            
            metadata = _read_metadata(session_dir)
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
async def get_debug_session(session_name: str) -> Dict[str, Any]:
    """
    Get details of a specific debug session.
    
    ⚠️ IMPORTANT: Debug sessions are created in H200 worker containers and stored on
    local filesystem. They are NOT accessible from Job Manager Service (different container).
    
    This endpoint will only return sessions that were created in the same container
    instance as this Job Manager Service. Most debug sessions are created in H200
    worker containers and will return 404.
    
    To access debug sessions from H200 workers, you would need to:
    1. Store sessions in Modal Volume (shared storage)
    2. Or access them directly from H200 worker container
    """
    if not debug_service.enabled:
        raise HTTPException(
            status_code=404, 
            detail="Debug service is not enabled"
        )
    
    try:
        session_dir = debug_service.base_dir / session_name
        if not session_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=(
                    f"Debug session '{session_name}' not found. "
                    "Debug sessions are created in H200 worker containers and stored on local filesystem. "
                    "They are not accessible from Job Manager Service (different container). "
                    "This is expected behavior - debug sessions are container-local."
                )
            )
        
        metadata = _read_metadata(session_dir)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Metadata for session {session_name} not found"
            )
        
        lora_log = _read_lora_log(session_dir)
        image_files = sorted([f.name for f in session_dir.iterdir() if f.suffix == ".png"])
        
        return {
            "session_name": session_name,
            "metadata": metadata,
            "lora_log": lora_log,
            "image_files": image_files,
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get debug session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sessions/{session_name}/images/{image_name}")
async def get_debug_image(session_name: str, image_name: str):
    """Get a specific debug image."""
    if not debug_service.enabled:
        raise HTTPException(status_code=404, detail="Debug service is not enabled")
    
    try:
        image_path = debug_service.base_dir / session_name / image_name
        if not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Image {image_name} not found in session {session_name}"
            )
        
        return FileResponse(image_path, media_type="image/png")
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get debug image")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cleanup")
async def cleanup_debug_sessions(keep_last_n: int = 50) -> Dict[str, Any]:
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
async def get_debug_status() -> Dict[str, Any]:
    """Get debug service status."""
    status: Dict[str, Any] = {
        "enabled": debug_service.enabled,
        "base_dir": str(debug_service.base_dir) if debug_service.enabled else None,
    }
    
    if debug_service.enabled and debug_service.base_dir.exists():
        sessions = [s for s in debug_service.base_dir.iterdir() if s.is_dir()]
        status["total_sessions"] = len(sessions)
        
        # Calculate total disk usage
        total_size = sum(
            f.stat().st_size
            for session_dir in sessions
            for f in session_dir.rglob("*")
            if f.is_file()
        )
        status["total_size_mb"] = round(total_size / (1024 * 1024), 2)
    
    return status


@router.get("/sessions/{session_name}/download")
async def download_debug_session(session_name: str):
    """Download all debug files from a session as a ZIP archive."""
    if not debug_service.enabled:
        raise HTTPException(status_code=404, detail="Debug service is not enabled")
    
    try:
        session_dir = debug_service.base_dir / session_name
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail=f"Session {session_name} not found")
        
        if not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Session {session_name} is not a directory")
        
        # Collect all files in session directory
        files_to_add = [f for f in session_dir.rglob("*") if f.is_file()]
        
        if not files_to_add:
            raise HTTPException(status_code=404, detail=f"No files found in session {session_name}")
        
        logger.info(f"📦 Creating ZIP archive for session {session_name} with {len(files_to_add)} files")
        
        # Create ZIP archive in memory
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in files_to_add:
                    try:
                        arcname = file_path.relative_to(session_dir)
                        zip_file.write(file_path, arcname)
                        logger.debug(f"Added to ZIP: {arcname} ({file_path.stat().st_size} bytes)")
                    except Exception as e:
                        logger.warning(f"Failed to add {file_path} to ZIP: {e}")
                        # Continue with other files even if one fails
            
            zip_buffer.seek(0)
            zip_data = zip_buffer.getvalue()
            
            if len(zip_data) == 0:
                raise HTTPException(status_code=500, detail="ZIP archive is empty")
            
            logger.info(f"✅ Created ZIP archive: {len(zip_data)} bytes")
            
            return Response(
                content=zip_data,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{session_name}.zip"',
                    "Content-Length": str(len(zip_data)),
                }
            )
        finally:
            zip_buffer.close()
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to download debug session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
