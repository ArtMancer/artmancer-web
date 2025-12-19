"""
Image caching service for temporary image storage during smart masking sessions.

Images are cached in memory with automatic cleanup after a timeout period.
Used for storing images temporarily during multi-step mask generation workflows.
"""

import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# In-memory cache: image_id -> (image_path, timestamp, base64_data)
_image_cache: dict[str, Tuple[str, float, str]] = {}
_cache_timeout = 3600  # 1 hour timeout


def cache_image(image_base64: str) -> str:
    """
    Cache an image and return a unique image_id.
    
    Args:
        image_base64: Base64 encoded image string (with or without data URL prefix)
    
    Returns:
        Unique image_id for the cached image
    
    Raises:
        ValueError: If image decoding or saving fails
    """
    # Generate unique ID
    image_id = str(uuid.uuid4())
    
    # Decode base64 to PIL Image
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Save to temporary file
        temp_dir = Path("/tmp" if Path("/tmp").exists() else Path.cwd() / "temp")
        temp_dir.mkdir(exist_ok=True)
        image_path = str(temp_dir / f"{image_id}.png")
        image.save(image_path, "PNG")
        
        # Store in cache with timestamp
        _image_cache[image_id] = (image_path, time.time(), image_base64)
        
        return image_id
    except Exception as e:
        raise ValueError(f"Failed to cache image: {str(e)}") from e


def get_cached_image(image_id: str) -> Optional[str]:
    """
    Get the file path of a cached image.
    
    Args:
        image_id: The image ID returned from cache_image
    
    Returns:
        File path to the cached image, or None if not found/expired
    
    Note:
        Automatically removes expired entries from cache.
    """
    if image_id not in _image_cache:
        return None
    
    image_path, timestamp, _ = _image_cache[image_id]
    
    # Check if expired
    if time.time() - timestamp > _cache_timeout:
        # Cleanup expired entry
        _cleanup_image_entry(image_id, image_path)
        return None
    
    # Check if file still exists
    if not Path(image_path).exists():
        del _image_cache[image_id]
        return None
    
    return image_path


def _cleanup_image_entry(image_id: str, image_path: str) -> None:
    """
    Cleanup a single image cache entry.
    
    Args:
        image_id: Image ID to remove
        image_path: Path to image file to delete
    """
    try:
        Path(image_path).unlink(missing_ok=True)
    except Exception:
        pass
    del _image_cache[image_id]


def cleanup_expired_images() -> int:
    """
    Remove expired images from cache.
    
    Returns:
        Number of images cleaned up
    """
    current_time = time.time()
    expired_ids: list[str] = []
    
    for image_id, (image_path, timestamp, _) in _image_cache.items():
        if current_time - timestamp > _cache_timeout:
            expired_ids.append(image_id)
            _cleanup_image_entry(image_id, image_path)
    
    if expired_ids:
        logger.debug(f"Cleaned up {len(expired_ids)} expired images from cache")
    
    return len(expired_ids)


def clear_cache() -> None:
    """
    Clear all cached images.
    
    Removes all entries from cache and deletes all cached image files.
    """
    for image_id, (image_path, _, _) in list(_image_cache.items()):
        _cleanup_image_entry(image_id, image_path)
    
    logger.debug("Cleared all images from cache")
