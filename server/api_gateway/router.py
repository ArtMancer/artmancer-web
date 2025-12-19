"""
API Gateway routing logic.

This module handles request routing to backend microservices with proper
error handling, SSE streaming, and service health checks.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, Optional, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from shared.clients.service_client import (
    ServiceClient,
    SEGMENTATION_SERVICE_URL,
    IMAGE_UTILS_SERVICE_URL,
    JOB_MANAGER_SERVICE_URL,
)

logger = logging.getLogger(__name__)


def _extract_request_error_details(error: Exception) -> str:
    """
    Extract detailed error information from httpx.RequestError.
    
    Args:
        error: Exception to extract details from
    
    Returns:
        Formatted error message string
    """
    if httpx is None or not isinstance(error, httpx.RequestError):  # type: ignore
        return str(error) if error else "Unknown error"
    
    error_type = type(error).__name__
    error_msg = str(error) if error else ""
    
    # Map specific error types to user-friendly messages
    error_messages: Dict[type, str] = {
        httpx.ConnectError: f"Connection error: Unable to connect to service. {error_msg}",  # type: ignore
        httpx.TimeoutException: f"Timeout error: Request timed out. {error_msg}",  # type: ignore
        httpx.NetworkError: f"Network error: {error_msg}",  # type: ignore
        httpx.ProxyError: f"Proxy error: {error_msg}",  # type: ignore
    }
    
    return error_messages.get(type(error), f"{error_type}: {error_msg}" if error_msg else f"{error_type}: Connection failed")


def _handle_service_error(
    error: Exception,
    service_name: str,
    service_url: str,
    endpoint: str
) -> HTTPException:
    """
    Handle errors when forwarding requests to services.
    
    Args:
        error: Exception that occurred
        service_name: Name of the service (for logging)
        service_url: Base URL of the service
        endpoint: Endpoint path
    
    Returns:
        HTTPException with appropriate status code and detail
    
    Raises:
        HTTPException: Always raises (never returns normally)
    """
    if httpx is None:
        logger.error(f"❌ [API Gateway] Error forwarding to {service_name}: {error}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(error),
                "service": service_name,
                "error_type": "gateway_error"
            }
        )
    
    # Handle HTTP status errors (forward service response)
    if isinstance(error, httpx.HTTPStatusError):  # type: ignore
        error_detail: Any = "Unknown error"
        try:
            error_detail = error.response.json()
        except Exception:
            error_detail = error.response.text or str(error)
        
        logger.error(f"❌ [API Gateway] {service_name} returned error: HTTP {error.response.status_code}")
        raise HTTPException(
            status_code=error.response.status_code,
            detail=error_detail
        )
    
    # Handle network/connection errors
    if isinstance(error, httpx.RequestError):  # type: ignore
        error_details = _extract_request_error_details(error)
        full_url = f"{service_url}{endpoint}"
        
        logger.error(f"❌ [API Gateway] Failed to connect to {service_name} at {full_url}: {error_details}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Service unavailable: {error_details}",
                "service": service_name,
                "url": full_url,
                "error_type": "service_unavailable"
            }
        )
    
    # Other unexpected errors
    error_trace = traceback.format_exc()
    logger.error(f"❌ [API Gateway] Unexpected error forwarding to {service_name}: {error_trace}")
    raise HTTPException(
        status_code=500,
        detail={
            "error": str(error),
            "service": service_name,
            "error_type": "gateway_error"
        }
    )


async def _forward_request(
    client: ServiceClient,
    request: Request,
    endpoint: str,
    service_name: str,
    service_url: str
) -> JSONResponse:
    """
    Forward HTTP request to backend service.
    
    Args:
        client: ServiceClient instance
        request: FastAPI Request object
        endpoint: Endpoint path to forward to
        service_name: Name of service (for logging)
        service_url: Base URL of service (for logging)
    
    Returns:
        JSONResponse with service response
    
    Raises:
        HTTPException: If forwarding fails
    """
    full_url = f"{service_url}{endpoint}"
    try:
        body = await request.json()
        logger.info(f"🔄 [API Gateway] Forwarding request to {full_url}")
        response = await client.post(endpoint, json=body)
        return JSONResponse(content=response)
    except Exception as e:
        _handle_service_error(e, service_name, service_url, endpoint)
        raise  # Never reached, but satisfies type checker


async def _stream_sse_events(
    task_id: str,
    job_manager_client: ServiceClient,
    service_url: str
) -> AsyncIterator[bytes]:
    """
    Stream Server-Sent Events from Job Manager with error recovery.
    
    Args:
        task_id: Task identifier
        job_manager_client: ServiceClient for Job Manager
        service_url: Base URL of Job Manager service
    
    Yields:
        SSE event bytes
    
    Note:
        Handles connection errors gracefully and attempts to send final status
        even if connection is interrupted.
    """
    if httpx is None:
        error_msg = "httpx library not available"
        yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()
        return
    
    endpoint = f"/api/generate/stream/{task_id}"
    full_url = f"{service_url}{endpoint}"
    
    async def _get_final_status() -> Optional[Dict[str, Any]]:
        """Get final status as fallback when stream fails."""
        try:
            status_response = await job_manager_client.get(f"/api/generate/status/{task_id}")
            if status_response and status_response.get("status") in ("done", "error", "cancelled"):
                return status_response
        except Exception as status_exc:
            logger.warning(f"⚠️ [API Gateway] Failed to get final status: {status_exc}")
        return None
    
    try:
        logger.info(f"🔄 [API Gateway] Forwarding SSE stream request to {full_url}")
        
        # Use httpx to stream the SSE response with longer timeout
        async with httpx.AsyncClient(  # type: ignore
            timeout=httpx.Timeout(1800.0, connect=30.0, read=1800.0),  # type: ignore
            follow_redirects=True
        ) as client:
            try:
                async with client.stream("GET", full_url) as response:
                    response.raise_for_status()
                    # Forward headers for SSE
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as stream_exc:  # type: ignore
                # Connection error during streaming - try to get final status
                error_msg = f"SSE stream connection error: {str(stream_exc)}"
                logger.warning(f"⚠️ [API Gateway] {error_msg}")
                
                final_status = await _get_final_status()
                if final_status:
                    yield f"data: {json.dumps(final_status)}\n\n".encode()
                    logger.info(f"✅ [API Gateway] Sent final status via fallback: {final_status.get('status')}")
                    return
                
                # Send error as SSE event
                yield f"data: {json.dumps({'error': error_msg, 'recoverable': True})}\n\n".encode()
    
    except Exception as e:
        error_msg = f"SSE connection error: {str(e)}"
        logger.error(f"❌ [API Gateway] {error_msg}")
        
        # Try to get final status as fallback
        final_status = await _get_final_status()
        if final_status:
            yield f"data: {json.dumps(final_status)}\n\n".encode()
            logger.info(f"✅ [API Gateway] Sent final status via fallback: {final_status.get('status')}")
            return
        
        # Send error as SSE event
        yield f"data: {json.dumps({'error': error_msg, 'recoverable': True})}\n\n".encode()


async def _stream_download(
    session_name: str,
    endpoint: str,
    service_url: str
) -> AsyncIterator[bytes]:
    """
    Stream ZIP file download from Job Manager.
    
    Args:
        session_name: Debug session name
        endpoint: Endpoint path
        service_url: Base URL of service
    
    Yields:
        File chunk bytes
    
    Raises:
        HTTPException: If download fails or httpx is not available
    """
    if httpx is None:
        raise HTTPException(status_code=500, detail="httpx library not available")
    
    full_url = f"{service_url}{endpoint}"
    logger.info(f"🔄 [API Gateway] Forwarding debug session download to {full_url}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:  # type: ignore
            async with client.stream("GET", full_url) as response:
                # Check status code immediately, before reading any data
                if response.status_code != 200:
                    error_detail = f"HTTP {response.status_code}: {response.reason_phrase or 'Unknown error'}"
                    logger.error(f"❌ [API Gateway] {error_detail}")
                    
                    # Try to read error message from response
                    try:
                        error_text = await response.aread()
                        if error_text:
                            try:
                                error_data = json.loads(error_text)
                                error_detail = error_data.get("detail", error_detail)
                            except json.JSONDecodeError:
                                # If not JSON, use the text as detail (truncated)
                                error_detail = error_text.decode('utf-8', errors='ignore')[:500]
                    except Exception as e:
                        logger.warning(f"⚠️ [API Gateway] Failed to read error response: {e}")
                    
                    raise HTTPException(status_code=response.status_code, detail=error_detail)
                
                # Status code is 200, proceed with streaming
                content_length = response.headers.get("Content-Length")
                if content_length:
                    logger.info(f"📦 [API Gateway] ZIP size: {content_length} bytes")
                
                # Stream the data
                chunk_count = 0
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
                
                logger.info(f"✅ [API Gateway] Streamed {chunk_count} chunks, {total_bytes} bytes total")
    
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:  # type: ignore
        error_detail = f"HTTP {e.response.status_code}: {e.response.reason_phrase or 'Unknown error'}"
        logger.error(f"❌ [API Gateway] {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        error_msg = f"Stream error: {str(e)}"
        logger.error(f"❌ [API Gateway] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


def create_router() -> APIRouter:
    """
    Create API Gateway router with all endpoints.
    
    Returns:
        Configured APIRouter instance
    """
    router = APIRouter(prefix="/api", tags=["gateway"])
    
    # Initialize service clients with appropriate timeouts
    # Segmentation service needs longer timeout for BiRefNet model loading and inference (5 minutes)
    segmentation_client = ServiceClient(SEGMENTATION_SERVICE_URL, timeout=300.0)  # 5 minutes for BiRefNet
    image_utils_client = ServiceClient(IMAGE_UTILS_SERVICE_URL)
    job_manager_client = ServiceClient(JOB_MANAGER_SERVICE_URL)
    
    # Health check
    @router.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "api_gateway"}
    
    @router.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint."""
        return {"message": "ArtMancer API Gateway", "version": "3.0.0"}
    
    # OPTIONS handler for CORS preflight - FastAPI CORS middleware should handle this,
    # but adding explicit handler as fallback
    @router.options("/{full_path:path}")
    async def options_handler(full_path: str) -> Any:  # type: ignore
        """Handle CORS preflight requests."""
        from fastapi.responses import Response
        return Response(status_code=200)
    
    # Async generation endpoints - forward to job manager
    @router.post("/generate/async")
    async def generate_async(request: Request) -> JSONResponse:
        """Forward async generation request to job manager."""
        return await _forward_request(
            job_manager_client,
            request,
            "/api/generate/async",
            "Job Manager Service",
            job_manager_client.service_url
        )
    
    @router.get("/generate/status/{task_id}")
    async def get_generation_status(task_id: str) -> JSONResponse:
        """Get generation job status from job manager."""
        try:
            response = await job_manager_client.get(f"/api/generate/status/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/generate/result/{task_id}")
    async def get_generation_result(task_id: str) -> JSONResponse:
        """Get generation result (JSON, base64) từ job manager."""
        try:
            response = await job_manager_client.get(f"/api/generate/result/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/generate/result-image/{task_id}")
    async def get_generation_result_image(task_id: str) -> StreamingResponse:
        """
        Proxy ảnh kết quả dạng binary (WebP/PNG) từ Job Manager.

        - Gọi /api/generate/result-image/{task_id} trên Job Manager
        - Stream thẳng bytes về client, giữ nguyên Content-Type
        """
        if httpx is None:
            raise HTTPException(status_code=500, detail="httpx library not available")

        endpoint = f"/api/generate/result-image/{task_id}"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"

        async def stream_image() -> AsyncIterator[bytes]:
            try:
                logger.info(f"🔄 [API Gateway] Forwarding result-image request to {full_url}")
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:  # type: ignore
                    async with client.stream("GET", full_url) as response:
                        # Nếu không 2xx thì ném HTTPException với nội dung chi tiết
                        if response.status_code != 200:
                            error_text = await response.aread()
                            detail: Any = f"HTTP {response.status_code}: {response.reason_phrase or 'Unknown error'}"
                            if error_text:
                                try:
                                    detail_json = json.loads(error_text)
                                    detail = detail_json.get("detail", detail)
                                except Exception:
                                    pass
                            logger.error(f"❌ [API Gateway] result-image error: {detail}")
                            raise HTTPException(status_code=response.status_code, detail=detail)

                        async for chunk in response.aiter_bytes():
                            yield chunk
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Result image download error: {str(e)}"
                logger.error(f"❌ [API Gateway] {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        # Tạm đặt content-type mặc định, sẽ được browser tự đoán nếu gateway không set
        # (httpx StreamingResponse không cho đọc header trước khi khởi tạo dễ dàng)
        return StreamingResponse(stream_image(), media_type="image/webp")
    
    @router.get("/generate/stream/{task_id}")
    async def stream_generation_progress(task_id: str) -> StreamingResponse:
        """
        Forward SSE stream for generation progress from job manager.
        
        Handles connection errors gracefully and ensures final status is sent
        even if connection is interrupted.
        """
        return StreamingResponse(
            _stream_sse_events(task_id, job_manager_client, job_manager_client.service_url),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "X-Content-Type-Options": "nosniff",
            }
        )
    
    # Helper function to forward segmentation requests
    async def _forward_segmentation_request(request: Request, endpoint: str) -> JSONResponse:
        """Forward request to segmentation service."""
        return await _forward_request(
            segmentation_client,
            request,
            endpoint,
            "Segmentation Service",
            segmentation_client.service_url
        )
    
    # Segmentation endpoints - forward to segmentation service
    @router.post("/smart-mask")
    async def smart_mask(request: Request) -> JSONResponse:
        """Forward smart mask request to segmentation service."""
        return await _forward_segmentation_request(request, "/api/smart-mask")
    
    @router.post("/smart-mask/detect")
    async def smart_mask_detect(request: Request) -> JSONResponse:
        """Forward smart mask detect request to segmentation service."""
        return await _forward_segmentation_request(request, "/api/smart-mask/detect")
    
    # Cancel endpoints - forward to appropriate services
    @router.post("/generate/async/cancel/{task_id}")
    async def cancel_async_generation(task_id: str) -> JSONResponse:
        """Cancel async generation task."""
        try:
            response = await job_manager_client.post(f"/api/generate/async/cancel/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(
                e,
                "Job Manager Service",
                job_manager_client.service_url,
                f"/api/generate/async/cancel/{task_id}"
            )
    
    @router.post("/smart-mask/cancel/{request_id}")
    async def cancel_smart_mask(request_id: str) -> JSONResponse:
        """Cancel smart mask segmentation request."""
        try:
            response = await segmentation_client.post(f"/api/smart-mask/cancel/{request_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(
                e,
                "Segmentation Service",
                segmentation_client.service_url,
                f"/api/smart-mask/cancel/{request_id}"
            )
    
    # Image utils endpoints - forward to image utils service
    @router.post("/image-utils/extract-object")
    async def extract_object(request: Request) -> JSONResponse:
        """Forward extract object request to image utils service."""
        return await _forward_request(
            image_utils_client,
            request,
            "/api/image-utils/extract-object",
            "Image Utils Service",
            image_utils_client.service_url
        )
    
    @router.post("/image-utils/add-shadow-preview")
    async def add_shadow_preview(request: Request) -> JSONResponse:
        """Forward add shadow preview request to image utils service."""
        return await _forward_request(
            image_utils_client,
            request,
            "/api/image-utils/add-shadow-preview",
            "Image Utils Service",
            image_utils_client.service_url
        )
    
    # Debug endpoints - forward to job manager (where debug sessions are created)
    @router.get("/debug/sessions")
    async def list_debug_sessions() -> JSONResponse:
        """List all debug sessions from job manager."""
        try:
            response = await job_manager_client.get("/api/debug/sessions")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(
                e,
                "Job Manager Service",
                job_manager_client.service_url,
                "/api/debug/sessions"
            )
    
    @router.get("/debug/sessions/{session_name}")
    async def get_debug_session(session_name: str) -> JSONResponse:
        """
        Get debug session details from job manager.
        
        Note: Debug sessions are created in H200 worker containers and may not be
        accessible if they were created in a different container instance.
        """
        try:
            response = await job_manager_client.get(f"/api/debug/sessions/{session_name}")
            return JSONResponse(content=response)
        except HTTPException as e:
            # Improve error message for 404 to explain why session might not be found
            if e.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Debug session '{session_name}' not found. "
                        "Sessions are created in H200 worker containers and may not be accessible "
                        "from Job Manager Service if they were created in a different container instance."
                    )
                )
            raise
        except Exception as e:
            raise _handle_service_error(
                e,
                "Job Manager Service",
                job_manager_client.service_url,
                f"/api/debug/sessions/{session_name}"
            )
    
    @router.get("/debug/sessions/{session_name}/download")
    async def download_debug_session(session_name: str) -> StreamingResponse:
        """Download debug session ZIP from job manager."""
        endpoint = f"/api/debug/sessions/{session_name}/download"
        return StreamingResponse(
            _stream_download(session_name, endpoint, job_manager_client.service_url),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{session_name}.zip"',
                "Cache-Control": "no-cache",
            }
        )
    
    @router.get("/debug/sessions/{session_name}/images/{image_name}")
    async def get_debug_image(session_name: str, image_name: str) -> StreamingResponse:
        """Get debug image from job manager."""
        if httpx is None:
            raise HTTPException(status_code=500, detail="httpx library not available")
        
        endpoint = f"/api/debug/sessions/{session_name}/images/{image_name}"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        async def stream_image() -> AsyncIterator[bytes]:
            """Stream image from Job Manager to client."""
            try:
                logger.info(f"🔄 [API Gateway] Forwarding debug image request to {full_url}")
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:  # type: ignore
                    async with client.stream("GET", full_url) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                error_msg = f"Image download error: {str(e)}"
                logger.error(f"❌ [API Gateway] {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
        
        return StreamingResponse(stream_image(), media_type="image/png")
    
    @router.get("/debug/status")
    async def get_debug_status() -> JSONResponse:
        """Get debug service status from job manager."""
        try:
            response = await job_manager_client.get("/api/debug/status")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(
                e,
                "Job Manager Service",
                job_manager_client.service_url,
                "/api/debug/status"
            )
    
    @router.post("/debug/cleanup")
    async def cleanup_debug_sessions(request: Request) -> JSONResponse:
        """Clean up old debug sessions in job manager."""
        try:
            body = await request.json()
            response = await job_manager_client.post("/api/debug/cleanup", json=body)
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(
                e,
                "Job Manager Service",
                job_manager_client.service_url,
                "/api/debug/cleanup"
            )
    
    # System endpoints - forward to appropriate service
    @router.get("/system/health")
    async def system_health() -> JSONResponse:
        """Get system health from all services with timeout."""
        import asyncio
        
        # Check all services with individual timeouts
        services: Dict[str, ServiceClient] = {
            "segmentation": segmentation_client,
            "image_utils": image_utils_client,
            "job_manager": job_manager_client,
        }
        
        async def check_service_health(name: str, client: ServiceClient, timeout: float = 2.0) -> tuple[str, Dict[str, Any]]:
            """
            Check health of a single service with timeout.
            
            Args:
                name: Service name
                client: ServiceClient instance
                timeout: Timeout in seconds
            
            Returns:
                Tuple of (service_name, health_status_dict)
            """
            try:
                health = await asyncio.wait_for(
                    client.get("/api/health"),
                    timeout=timeout
                )
                return name, {"status": "healthy", **health}
            except asyncio.TimeoutError:
                return name, {"status": "timeout", "error": f"Health check timed out after {timeout}s"}
            except Exception as e:
                return name, {"status": "unhealthy", "error": str(e)}
        
        # Check all services concurrently with individual timeouts
        tasks = [
            check_service_health(name, client, timeout=2.0)
            for name, client in services.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build health status dict
        health_status: Dict[str, Dict[str, Any]] = {}
        service_names = list(services.keys())
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = service_names[i]
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
            elif isinstance(result, tuple) and len(result) == 2:
                name, status = result
                health_status[name] = status
            else:
                service_name = service_names[i]
                health_status[service_name] = {
                    "status": "unknown",
                    "error": "Unexpected result format"
                }
        
        # Determine overall status
        all_healthy = all(s.get("status") == "healthy" for s in health_status.values())
        overall_status = "healthy" if all_healthy else "degraded"
        
        return JSONResponse(content={
            "status": overall_status,
            "services": health_status,
        })
    
    return router
