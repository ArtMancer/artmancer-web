"""HTTP client for calling other services."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class ServiceClient:
    """HTTP client for inter-service communication."""
    
    def __init__(self, service_url: str, timeout: float = 30.0):
        """
        Initialize service client.
        
        Args:
            service_url: Base URL of the service
            timeout: Request timeout in seconds
        """
        if httpx is None:
            raise ImportError("httpx is required for ServiceClient. Install it with: pip install httpx")
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        # Create httpx.Timeout object (httpx requires Timeout object, not float)
        # Set connect, read, write, and pool timeouts all to the same value
        timeout_obj = httpx.Timeout(timeout, connect=timeout, read=timeout, write=timeout, pool=timeout)
        # Enable redirect following for Modal's 303 redirects (with __modal_function_call_id)
        self._client = httpx.AsyncClient(timeout=timeout_obj, follow_redirects=True)
    
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request to service."""
        url = f"{self.service_url}{path}"
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if httpx is not None:
                if isinstance(e, httpx.HTTPStatusError):
                    # Log error for debugging
                    print(f"❌ [ServiceClient] GET {url} returned HTTP {e.response.status_code}")
                elif isinstance(e, httpx.RequestError):
                    # Log network error for debugging
                    error_type = type(e).__name__
                    print(f"❌ [ServiceClient] GET {url} failed: {error_type}: {str(e)}")
            # Re-raise with more context
            raise
    
    async def post(self, path: str, data: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request to service."""
        url = f"{self.service_url}{path}"
        logger.info(f"🌐 [ServiceClient] POST {url}")
        try:
            response = await self._client.post(url, data=data, json=json, follow_redirects=True)
            logger.info(f"📡 [ServiceClient] POST {url} returned HTTP {response.status_code}")
            
            # Handle 303 redirect manually if needed (Modal async function calls)
            # Modal returns 303 with location header containing __modal_function_call_id
            if response.status_code == 303:
                location = response.headers.get("location")
                if location:
                    logger.info(f"🔄 [ServiceClient] Following 303 redirect to: {location}")
                    # For Modal, we need to POST again to the redirect location (not GET)
                    # This is because Modal uses 303 to redirect to async function call endpoint
                    redirect_response = await self._client.post(
                        location,
                        data=data,
                        json=json,
                        follow_redirects=True
                    )
                    redirect_response.raise_for_status()
                    return redirect_response.json()
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if httpx is not None:
                if isinstance(e, httpx.HTTPStatusError):
                    # Log error for debugging with response body
                    error_body = e.response.text if hasattr(e.response, 'text') else "No response body"
                    logger.error(f"❌ [ServiceClient] POST {url} returned HTTP {e.response.status_code}: {error_body[:500]}")
                    print(f"❌ [ServiceClient] POST {url} returned HTTP {e.response.status_code}")
                    print(f"   Response headers: {dict(e.response.headers)}")
                    print(f"   Response body: {error_body[:500]}")
                elif isinstance(e, httpx.RequestError):
                    # Log network error for debugging
                    error_type = type(e).__name__
                    logger.error(f"❌ [ServiceClient] POST {url} failed: {error_type}: {str(e)}")
                    print(f"❌ [ServiceClient] POST {url} failed: {error_type}: {str(e)}")
            # Re-raise with more context
            raise
    
    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()


# Service URLs from environment
def get_service_url(service_name: str, default: str) -> str:
    """Get service URL from environment or use default."""
    env_key = f"{service_name.upper()}_SERVICE_URL"
    return os.getenv(env_key, default)


# Default service URLs (Modal endpoints)
SEGMENTATION_SERVICE_URL = get_service_url(
    "segmentation",
    os.getenv("SEGMENTATION_SERVICE_URL", "https://nxan2911--segmentation.modal.run")
)
IMAGE_UTILS_SERVICE_URL = get_service_url(
    "image_utils",
    os.getenv("IMAGE_UTILS_SERVICE_URL", "https://nxan2911--image-utils.modal.run")
)
JOB_MANAGER_SERVICE_URL = get_service_url(
    "job_manager",
    os.getenv("JOB_MANAGER_SERVICE_URL", "https://nxan2911--job-manager.modal.run")
)

