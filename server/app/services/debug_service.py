"""
Debug service for saving images and logs during generation process.

Folder structure:
debug_output/
├── YYYYMMDD_HHMMSS_uuid/
│   ├── metadata.json
│   ├── lora_debug.log
│   ├── 01_input_image.png
│   ├── 02_mask_image.png
│   ├── 03_mask_background.png
│   ├── 04_reference_image.png (optional)
│   ├── 05_canny_edge.png (optional)
│   ├── 06_mae_output.png (optional)
│   └── 07_generated_output.png
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

# numpy and PIL are imported lazily where needed to avoid requiring them in services that don't need them
if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class DebugSession:
    """Debug session for a single generation request."""
    
    def __init__(self, base_dir: str | Path = "debug_output", enabled: bool = True):
        """
        Initialize debug session.
        
        Args:
            base_dir: Base directory for all debug output
            enabled: Whether debug is enabled (set to False to disable)
        """
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session folder with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid4())[:8]
        self.session_name = f"{timestamp}_{session_id}"
        self.session_dir = self.base_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        self.metadata: Dict[str, Any] = {
            "session_id": session_id,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "images": {},
            "parameters": {},
            "lora_info": {},
        }
        
        # Initialize LoRA debug log
        self.lora_log_path = self.session_dir / "lora_debug.log"
        self.lora_log_path.touch()
        
        logger.info(f"🐛 Debug session created: {self.session_dir}")
    
    def log_lora(self, message: str) -> None:
        """Log LoRA-related debug information."""
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}\n"
        with open(self.lora_log_path, "a", encoding="utf-8") as f:
            f.write(log_line)
        logger.debug(f"LoRA debug: {message}")
    
    def save_image(
        self,
        image: Any | np.ndarray | None,  # type: ignore
        name: str,
        step: str,
        description: str = "",
    ) -> Optional[str]:
        """
        Save an image to the debug session.
        
        Args:
            image: PIL Image or numpy array to save
            name: Internal name for the image (e.g., "input_image")
            step: Step number prefix (e.g., "01", "02")
            description: Human-readable description
        
        Returns:
            Path to saved image or None if image is None or debug disabled
        """
        if not self.enabled or image is None:
            return None
        
        try:
            # Lazy import PIL and numpy to avoid requiring them in services that don't need them
            from PIL import Image
            import numpy as np
            
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                # Handle different array shapes
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Save image
            filename = f"{step}_{name}.png"
            filepath = self.session_dir / filename
            image.save(filepath)
            
            # Update metadata
            self.metadata["images"][name] = {
                "filename": filename,
                "step": step,
                "description": description,
                "size": list(image.size),
                "mode": image.mode,
            }
            
            logger.debug(f"💾 Saved debug image: {filename}")
            return str(filepath)
        
        except Exception as exc:
            logger.warning(f"Failed to save debug image '{name}': {exc}")
            return None
    
    def save_parameters(self, params: Dict[str, Any]) -> None:
        """Save generation parameters."""
        if not self.enabled:
            return
        self.metadata["parameters"] = params
        logger.debug(f"💾 Saved generation parameters: {len(params)} keys")
    
    def save_lora_info(self, info: Dict[str, Any]) -> None:
        """Save LoRA adapter information."""
        if not self.enabled:
            return
        self.metadata["lora_info"].update(info)
        logger.debug(f"💾 Saved LoRA info: {info}")
    
    def save_research_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save research metrics for worst case analysis."""
        if not self.enabled:
            return
        if "research_metrics" not in self.metadata:
            self.metadata["research_metrics"] = {}
        self.metadata["research_metrics"].update(metrics)
        logger.debug(f"💾 Saved research metrics: {len(metrics)} keys")
    
    def finalize(self, success: bool = True, error: Optional[str] = None) -> None:
        """
        Finalize the debug session and save metadata.
        
        Args:
            success: Whether generation was successful
            error: Error message if generation failed
        """
        if not self.enabled:
            return
        
        self.metadata["success"] = success
        self.metadata["error"] = error
        self.metadata["finalized_at"] = datetime.now().isoformat()
        
        # Save metadata as JSON
        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Debug session finalized: {self.session_dir}")
    
    def get_session_path(self) -> Optional[Path]:
        """Get the session directory path."""
        if not self.enabled:
            return None
        return self.session_dir


class DebugService:
    """Service for managing debug sessions."""
    
    def __init__(self, base_dir: str | Path = "debug_output", enabled: bool = True):
        """
        Initialize debug service.
        
        Args:
            base_dir: Base directory for all debug output
            enabled: Whether debug is enabled globally
        """
        self.base_dir = Path(base_dir)
        self.enabled = enabled
        
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🐛 Debug service initialized: {self.base_dir}")
    
    def create_session(self) -> DebugSession:
        """Create a new debug session."""
        return DebugSession(base_dir=self.base_dir, enabled=self.enabled)
    
    def cleanup_old_sessions(self, keep_last_n: int = 50) -> int:
        """
        Clean up old debug sessions, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent sessions to keep
        
        Returns:
            Number of sessions deleted
        """
        if not self.enabled or not self.base_dir.exists():
            return 0
        
        try:
            # Get all session directories
            sessions = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            
            # Delete old sessions
            deleted = 0
            for session in sessions[keep_last_n:]:
                try:
                    # Delete all files in session
                    for file in session.iterdir():
                        file.unlink()
                    session.rmdir()
                    deleted += 1
                except Exception as exc:
                    logger.warning(f"Failed to delete session {session}: {exc}")
            
            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} old debug sessions")
            
            return deleted
        
        except Exception as exc:
            logger.warning(f"Failed to cleanup debug sessions: {exc}")
            return 0



# Global debug service instance
# Can be controlled via DEBUG_GENERATION environment variable
DEBUG_ENABLED = os.getenv("DEBUG_GENERATION", "true").lower() in ("true", "1", "yes")
debug_service = DebugService(base_dir="debug_output", enabled=DEBUG_ENABLED)

