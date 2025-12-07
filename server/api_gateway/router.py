"""API Gateway routing logic."""

from __future__ import annotations

import json
import traceback
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

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


def _extract_request_error_details(error: Exception) -> str:
    """Extract detailed error information from httpx.RequestError."""
    if httpx is None:
        return str(error) if error else "Unknown error"
    
    if not isinstance(error, httpx.RequestError):
        return str(error) if error else "Unknown error"
    
    error_type = type(error).__name__
    error_msg = str(error) if error else ""
    
    # Extract more details based on error type
    if isinstance(error, httpx.ConnectError):
        return f"Connection error: Unable to connect to service. {error_msg}"
    elif isinstance(error, httpx.TimeoutException):
        return f"Timeout error: Request timed out. {error_msg}"
    elif isinstance(error, httpx.NetworkError):
        return f"Network error: {error_msg}"
    elif isinstance(error, httpx.ProxyError):
        return f"Proxy error: {error_msg}"
    else:
        return f"{error_type}: {error_msg}" if error_msg else f"{error_type}: Connection failed"


def _handle_service_error(
    error: Exception,
    service_name: str,
    service_url: str,
    endpoint: str
) -> HTTPException:
    """Handle errors when forwarding requests to services."""
    if httpx is not None:
        if isinstance(error, httpx.HTTPStatusError):
            # Forward the actual error from the service
            error_detail = "Unknown error"
            try:
                error_detail = error.response.json()
            except Exception:
                error_detail = error.response.text or str(error)
            
            print(f"❌ [API Gateway] {service_name} returned error: HTTP {error.response.status_code}")
            raise HTTPException(
                status_code=error.response.status_code,
                detail=error_detail
            )
        elif isinstance(error, httpx.RequestError):
            # Network or connection error - extract detailed info
            error_details = _extract_request_error_details(error)
            full_url = f"{service_url}{endpoint}"
            
            print(f"❌ [API Gateway] Failed to connect to {service_name} at {full_url}: {error_details}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": f"Service unavailable: {error_details}",
                    "service": service_name,
                    "url": full_url,
                    "error_type": "service_unavailable"
                }
            )
    
    # Other errors
    error_trace = traceback.format_exc()
    print(f"❌ [API Gateway] Unexpected error forwarding to {service_name}: {error_trace}")
    raise HTTPException(
        status_code=500,
        detail={
            "error": str(error),
            "service": service_name,
            "error_type": "gateway_error"
        }
    )


def create_router() -> APIRouter:
    """Create API Gateway router."""
    router = APIRouter(prefix="/api", tags=["gateway"])
    
    # Initialize service clients with appropriate timeouts
    # Segmentation service needs longer timeout for BiRefNet model loading and inference (5 minutes)
    segmentation_client = ServiceClient(SEGMENTATION_SERVICE_URL, timeout=300.0)  # 5 minutes for BiRefNet
    image_utils_client = ServiceClient(IMAGE_UTILS_SERVICE_URL)
    job_manager_client = ServiceClient(JOB_MANAGER_SERVICE_URL)
    
    # Health check
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "api_gateway"}
    
    @router.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "ArtMancer API Gateway", "version": "3.0.0"}
    
    # OPTIONS handler for CORS preflight - FastAPI CORS middleware should handle this,
    # but adding explicit handler as fallback
    @router.options("/{full_path:path}")
    async def options_handler(full_path: str):
        """Handle CORS preflight requests."""
        from fastapi.responses import Response
        return Response(status_code=200)
    
    # Async generation endpoints - forward to job manager
    @router.post("/generate/async")
    async def generate_async(request: Request):
        """Forward async generation request to job manager."""
        endpoint = "/api/generate/async"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        try:
            body = await request.json()
            print(f"🔄 [API Gateway] Forwarding async generation request to {full_url}")
            response = await job_manager_client.post(endpoint, json=body)
            return JSONResponse(content=response)
        except Exception as e:
            _handle_service_error(e, "Job Manager Service", service_url, endpoint)
    
    @router.get("/generate/status/{task_id}")
    async def get_generation_status(task_id: str):
        """Get generation job status from job manager."""
        try:
            response = await job_manager_client.get(f"/api/generate/status/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/generate/result/{task_id}")
    async def get_generation_result(task_id: str):
        """Get generation result from job manager."""
        try:
            response = await job_manager_client.get(f"/api/generate/result/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/generate/stream/{task_id}")
    async def stream_generation_progress(task_id: str):
        """Forward SSE stream for generation progress from job manager."""
        from fastapi.responses import StreamingResponse
        import asyncio
        
        endpoint = f"/api/generate/stream/{task_id}"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        async def stream_events():
            """Stream events from Job Manager to client."""
            try:
                print(f"🔄 [API Gateway] Forwarding SSE stream request to {full_url}")
                # Use httpx to stream the SSE response
                async with httpx.AsyncClient(timeout=1800.0, follow_redirects=True) as client:
                    async with client.stream("GET", full_url) as response:
                        response.raise_for_status()
                        # Forward headers for SSE
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                error_msg = f"SSE connection error: {str(e)}"
                print(f"❌ [API Gateway] {error_msg}")
                # Send error as SSE event
                yield f"data: {json.dumps({'error': error_msg})}\n\n".encode()
        
        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
    
    # Helper function to forward segmentation requests
    async def _forward_segmentation_request(request: Request, endpoint: str) -> JSONResponse:
        """Forward request to segmentation service."""
        service_url = segmentation_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        try:
            body = await request.json()
            print(f"🔄 [API Gateway] Forwarding segmentation request to {full_url}")
            response = await segmentation_client.post(endpoint, json=body)
            return JSONResponse(content=response)
        except Exception as e:
            # _handle_service_error raises HTTPException, so this will never return normally
            raise _handle_service_error(e, "Segmentation Service", service_url, endpoint)
    
    # Segmentation endpoints - forward to segmentation service
    @router.post("/smart-mask")
    async def smart_mask(request: Request):
        """Forward smart mask request to segmentation service."""
        return await _forward_segmentation_request(request, "/api/smart-mask")
    
    @router.post("/smart-mask/detect")
    async def smart_mask_detect(request: Request):
        """Forward smart mask detect request to segmentation service."""
        return await _forward_segmentation_request(request, "/api/smart-mask/detect")
    
    # Cancel endpoints - forward to appropriate services
    @router.post("/generate/async/cancel/{task_id}")
    async def cancel_async_generation(task_id: str):
        """Cancel async generation task."""
        try:
            response = await job_manager_client.post(f"/api/generate/async/cancel/{task_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Job Manager Service", job_manager_client.service_url, f"/api/generate/async/cancel/{task_id}")
    
    @router.post("/smart-mask/cancel/{request_id}")
    async def cancel_smart_mask(request_id: str):
        """Cancel smart mask segmentation request."""
        try:
            response = await segmentation_client.post(f"/api/smart-mask/cancel/{request_id}")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Segmentation Service", segmentation_client.service_url, f"/api/smart-mask/cancel/{request_id}")
    
    # Image utils endpoints - forward to image utils service
    @router.post("/image-utils/extract-object")
    async def extract_object(request: Request):
        """Forward extract object request to image utils service."""
        endpoint = "/api/image-utils/extract-object"
        service_url = image_utils_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        try:
            body = await request.json()
            print(f"🔄 [API Gateway] Forwarding extract-object request to {full_url}")
            response = await image_utils_client.post(endpoint, json=body)
            return JSONResponse(content=response)
        except Exception as e:
            _handle_service_error(e, "Image Utils Service", service_url, endpoint)
    
    # Debug endpoints - forward to job manager (where debug sessions are created)
    @router.get("/debug/sessions")
    async def list_debug_sessions():
        """List all debug sessions from job manager."""
        try:
            response = await job_manager_client.get("/api/debug/sessions")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Job Manager Service", job_manager_client.service_url, "/api/debug/sessions")
    
    @router.get("/debug/sessions/{session_name}")
    async def get_debug_session(session_name: str):
        """Get debug session details from job manager."""
        try:
            response = await job_manager_client.get(f"/api/debug/sessions/{session_name}")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Job Manager Service", job_manager_client.service_url, f"/api/debug/sessions/{session_name}")
    
    @router.get("/debug/sessions/{session_name}/download")
    async def download_debug_session(session_name: str):
        """Download debug session ZIP from job manager."""
        from fastapi.responses import StreamingResponse
        
        if httpx is None:
            raise HTTPException(status_code=500, detail="httpx library not available")
        
        endpoint = f"/api/debug/sessions/{session_name}/download"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        print(f"🔄 [API Gateway] Forwarding debug session download to {full_url}")
        
        # Create streaming generator that checks status BEFORE yielding any data
        # This allows us to raise HTTPException before StreamingResponse starts sending data
        async def stream_download():
            """Stream ZIP file from Job Manager to client."""
            try:
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    async with client.stream("GET", full_url) as response:
                        # Check status code immediately, before reading any data
                        if response.status_code != 200:
                            error_detail = f"HTTP {response.status_code}: {response.reason_phrase or 'Unknown error'}"
                            print(f"❌ [API Gateway] {error_detail}")
                            
                            # Try to read error message from response
                            try:
                                error_text = await response.aread()
                                if error_text:
                                    try:
                                        error_data = json.loads(error_text)
                                        error_detail = error_data.get("detail", error_detail)
                                    except json.JSONDecodeError:
                                        # If not JSON, use the text as detail
                                        error_detail = error_text.decode('utf-8', errors='ignore')[:500]
                            except Exception as e:
                                print(f"⚠️ [API Gateway] Failed to read error response: {e}")
                            
                            # Raise HTTPException - this will be caught and converted to proper HTTP response
                            # BEFORE any data is yielded, so StreamingResponse hasn't started yet
                            raise HTTPException(status_code=response.status_code, detail=error_detail)
                        
                        # Status code is 200, proceed with streaming
                        # Get content length from response if available
                        content_length = response.headers.get("Content-Length")
                        if content_length:
                            print(f"📦 [API Gateway] ZIP size: {content_length} bytes")
                        
                        # Stream the data
                        chunk_count = 0
                        total_bytes = 0
                        async for chunk in response.aiter_bytes():
                            chunk_count += 1
                            total_bytes += len(chunk)
                            yield chunk
                        
                        print(f"✅ [API Gateway] Streamed {chunk_count} chunks, {total_bytes} bytes total")
            except HTTPException:
                # Re-raise HTTPException - FastAPI will handle it properly
                # Since no data has been yielded yet, this is safe
                raise
            except httpx.HTTPStatusError as e:
                error_detail = f"HTTP {e.response.status_code}: {e.response.reason_phrase or 'Unknown error'}"
                print(f"❌ [API Gateway] {error_detail}")
                raise HTTPException(status_code=e.response.status_code, detail=error_detail)
            except Exception as e:
                error_msg = f"Stream error: {str(e)}"
                print(f"❌ [API Gateway] {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
        
        # Create StreamingResponse - errors should be caught before this point
        return StreamingResponse(
            stream_download(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{session_name}.zip"',
                "Cache-Control": "no-cache",
            }
        )
    
    @router.get("/debug/sessions/{session_name}/images/{image_name}")
    async def get_debug_image(session_name: str, image_name: str):
        """Get debug image from job manager."""
        from fastapi.responses import StreamingResponse
        
        endpoint = f"/api/debug/sessions/{session_name}/images/{image_name}"
        service_url = job_manager_client.service_url
        full_url = f"{service_url}{endpoint}"
        
        async def stream_image():
            """Stream image from Job Manager to client."""
            try:
                print(f"🔄 [API Gateway] Forwarding debug image request to {full_url}")
                # Use httpx to stream the response
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    async with client.stream("GET", full_url) as response:
                        response.raise_for_status()
                        # Forward headers for image
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                error_msg = f"Image download error: {str(e)}"
                print(f"❌ [API Gateway] {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)
        
        return StreamingResponse(
            stream_image(),
            media_type="image/png"
        )
    
    @router.get("/debug/status")
    async def get_debug_status():
        """Get debug service status from job manager."""
        try:
            response = await job_manager_client.get("/api/debug/status")
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Job Manager Service", job_manager_client.service_url, "/api/debug/status")
    
    @router.post("/debug/cleanup")
    async def cleanup_debug_sessions(request: Request):
        """Clean up old debug sessions in job manager."""
        try:
            body = await request.json()
            response = await job_manager_client.post("/api/debug/cleanup", json=body)
            return JSONResponse(content=response)
        except Exception as e:
            raise _handle_service_error(e, "Job Manager Service", job_manager_client.service_url, "/api/debug/cleanup")
    
    # System endpoints - forward to appropriate service
    @router.get("/system/health")
    async def system_health():
        """Get system health from all services with timeout."""
        import asyncio
        
        # Check all services with individual timeouts
        services = {
            "segmentation": segmentation_client,
            "image_utils": image_utils_client,
            "job_manager": job_manager_client,
        }
        
        async def check_service_health(name: str, client: ServiceClient, timeout: float = 3.0):
            """Check health of a single service with timeout."""
            try:
                # Use asyncio.wait_for to add timeout
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
        service_names = list(services.keys())
        tasks = [
            check_service_health(name, client, timeout=3.0)
            for name, client in services.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build health status dict
        health_status = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # If exception occurred, mark service as unhealthy
                service_name = service_names[i]
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
            elif isinstance(result, tuple) and len(result) == 2:
                # Normal result: (name, status)
                name, status = result
                health_status[name] = status
            else:
                # Unexpected result format
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

