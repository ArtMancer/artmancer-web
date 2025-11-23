"""
Visualization endpoint for downloading visualization images.
"""
import logging
import zipfile
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from ...core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visualization", tags=["visualization"])

# Base directory for visualizations
BASE_DIR = Path(__file__).resolve().parents[3]
VISUALIZATION_DIR = BASE_DIR / "visualizations"


@router.get("/{request_id}/download")
async def download_visualization(request_id: str, format: str = "zip"):
    """
    Download visualization images for a specific request_id.
    
    Args:
        request_id: The request ID from generation response
        format: Download format - "zip" (default) or "images" (individual files)
    
    Returns:
        ZIP file containing all visualization images, or individual image files
    """
    request_dir = VISUALIZATION_DIR / request_id
    
    if not request_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Visualization not found for request_id: {request_id}"
        )
    
    if format == "zip":
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all PNG files from the request directory
            for file_path in request_dir.rglob("*.png"):
                # Get relative path for ZIP structure
                arcname = file_path.relative_to(request_dir)
                zip_file.write(file_path, arcname)
            
            # Also add metadata.json if exists
            metadata_file = request_dir / "metadata.json"
            if metadata_file.exists():
                zip_file.write(metadata_file, "metadata.json")
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=visualization-{request_id}.zip"
            }
        )
    else:
        # Return individual files (not implemented for now)
        raise HTTPException(
            status_code=400,
            detail="Individual file download not yet implemented. Use format=zip"
        )


@router.get("/{request_id}/original")
async def get_original_visualization(request_id: str):
    """
    Get the original image from visualization.
    
    Args:
        request_id: The request ID from generation response
    
    Returns:
        Original image file
    """
    original_file = VISUALIZATION_DIR / request_id / "00_original.png"
    
    if not original_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Original visualization image not found for request_id: {request_id}"
        )
    
    return FileResponse(
        original_file,
        media_type="image/png",
        filename=f"original-{request_id}.png"
    )


@router.get("/{request_id}/generated")
async def get_generated_visualization(request_id: str):
    """
    Get the generated image from visualization.
    
    Args:
        request_id: The request ID from generation response
    
    Returns:
        Generated image file
    """
    generated_file = VISUALIZATION_DIR / request_id / "99_output.png"
    
    if not generated_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Generated visualization image not found for request_id: {request_id}"
        )
    
    return FileResponse(
        generated_file,
        media_type="image/png",
        filename=f"generated-{request_id}.png"
    )

