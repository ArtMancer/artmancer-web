"""
Container State Management Service

Tracks container state: cold, warm, hot.
Used for monitoring and optimization of container lifecycle.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContainerState:
    """
    Manages container state: cold, warm, hot.
    
    State transitions:
    - cold → warm: When model is loaded
    - warm → hot: When activity occurs within warm_threshold_seconds
    - hot → warm: When inactive for warm_threshold_seconds
    """
    
    def __init__(self) -> None:
        """Initialize container state manager."""
        self._state: str = "cold"
        self._model_load_time_ms: Optional[float] = None
        self._last_activity_time: Optional[float] = None
        self._warm_threshold_seconds = 300  # 5 minutes for hot state
        self._initialization_start = time.time()
        
    def mark_model_loaded(self, load_time_ms: float) -> None:
        """
        Mark that model has been loaded.
        
        Args:
            load_time_ms: Model load time in milliseconds
        """
        self._model_load_time_ms = load_time_ms
        if self._state == "cold":
            self._state = "warm"
            logger.info(
                f"✅ Container state: cold → warm "
                f"(model loaded in {load_time_ms:.2f}ms)"
            )
    
    def mark_activity(self) -> None:
        """
        Mark that container has activity (request processed).
        
        Transitions warm → hot if activity occurs within threshold.
        """
        self._last_activity_time = time.time()
        if self._state == "warm":
            # Check if we should transition to hot
            if (
                self._last_activity_time and
                (time.time() - self._last_activity_time) < self._warm_threshold_seconds
            ):
                self._state = "hot"
                logger.debug(
                    "Container state: warm → hot (active within 5 minutes)"
                )
    
    def get_state(self) -> str:
        """
        Get current container state.
        
        Automatically transitions hot → warm if inactive for threshold.
        
        Returns:
            Current state string ("cold", "warm", or "hot")
        """
        # Check if we should transition from hot to warm
        if self._state == "hot" and self._last_activity_time:
            time_since_activity = time.time() - self._last_activity_time
            if time_since_activity >= self._warm_threshold_seconds:
                self._state = "warm"
                logger.debug(
                    "Container state: hot → warm (inactive for 5+ minutes)"
                )
        
        return self._state
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed state information.
        
        Returns:
            Dictionary containing:
            - state: Current state string
            - uptime_seconds: Container uptime in seconds
            - model_load_time_ms: Model load time (if loaded)
            - last_activity_seconds_ago: Time since last activity (if any)
        """
        state = self.get_state()
        info: Dict[str, Any] = {
            "state": state,
            "uptime_seconds": round(time.time() - self._initialization_start, 2),
        }
        
        if self._model_load_time_ms is not None:
            info["model_load_time_ms"] = round(self._model_load_time_ms, 2)
        
        if self._last_activity_time:
            time_since_activity = time.time() - self._last_activity_time
            info["last_activity_seconds_ago"] = round(time_since_activity, 2)
        
        return info


# Global container state instance
_container_state = ContainerState()


def get_container_state() -> ContainerState:
    """
    Get global container state instance.
    
    Returns:
        Global ContainerState instance
    """
    return _container_state


def mark_model_loaded(load_time_ms: float) -> None:
    """
    Mark that model has been loaded.
    
    Args:
        load_time_ms: Model load time in milliseconds
    """
    _container_state.mark_model_loaded(load_time_ms)


def mark_activity() -> None:
    """Mark that container has activity."""
    _container_state.mark_activity()
