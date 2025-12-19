"""
API Gateway - Main entry point for all requests.

This module creates the FastAPI application with CORS middleware and
request logging for the API Gateway service.
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .router import create_router


def _parse_allowed_origins() -> List[str]:
    """
    Parse allowed origins from environment variable.
    
    Returns:
        List of allowed origin strings
    """
    allowed_origins_str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,https://localhost:3000"
    )
    return [
        origin.strip()
        for origin in allowed_origins_str.split(",")
        if origin.strip()
    ]


def _add_cors_middleware(app: FastAPI, allowed_origins: List[str]) -> None:
    """
    Add CORS middleware to FastAPI app.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origin strings
    
    Note:
        Middleware runs in reverse order (LIFO), so this will run first.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (POST, GET, PUT, DELETE, OPTIONS, etc.)
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"],  # Expose all headers in response
        max_age=3600,  # Cache preflight requests for 1 hour
    )


def _add_cors_logging_middleware(app: FastAPI, allowed_origins: List[str]) -> None:
    """
    Add middleware to log CORS requests and ensure CORS headers on all responses.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origin strings
    
    Note:
        This middleware ensures CORS headers are present on all responses,
        including redirects (important for Modal's 303 redirects).
    """
    @app.middleware("http")
    async def log_cors_requests(request: Request, call_next) -> Response:
        """
        Log CORS requests and ensure CORS headers on all responses.
        
        Args:
            request: FastAPI Request object
            call_next: Next middleware/route handler
        
        Returns:
            Response with CORS headers if needed
        """
        origin = request.headers.get("origin")
        if origin:
            print(f"🌐 [API Gateway] CORS request from origin: {origin}")
        
        response = await call_next(request)
        
        # Ensure CORS headers are present on all responses (including redirects)
        # This is important for Modal's 303 redirects
        if origin and origin in allowed_origins:
            cors_headers = {
                "access-control-allow-origin": origin,
                "access-control-allow-credentials": "true",
                "access-control-allow-methods": "*",
                "access-control-allow-headers": "*",
            }
            
            for header_name, header_value in cors_headers.items():
                if header_name not in response.headers:
                    response.headers[header_name] = header_value
        
        # Log CORS headers in response
        cors_response_headers = {
            k: v for k, v in response.headers.items()
            if k.lower().startswith("access-control")
        }
        if cors_response_headers:
            print(f"🌐 [API Gateway] CORS response headers: {cors_response_headers}")
        
        return response


def _add_security_middleware(app: FastAPI) -> None:
    """
    Add security middleware to remove server header.
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def remove_server_header(request: Request, call_next) -> Response:
        """
        Remove server header for security.
        
        Args:
            request: FastAPI Request object
            call_next: Next middleware/route handler
        
        Returns:
            Response with server header removed
        """
        response = await call_next(request)
        if "server" in response.headers:
            del response.headers["server"]
        return response


def create_app() -> FastAPI:
    """
    Create FastAPI app for API Gateway.
    
    Returns:
        Configured FastAPI application instance
    """
    allowed_origins = _parse_allowed_origins()
    print(f"🌐 [API Gateway] CORS configured with allowed origins: {allowed_origins}")
    
    app = FastAPI(
        title="ArtMancer API Gateway",
        description="API Gateway for routing requests to microservices",
        version="3.0.0",
    )
    
    # Include routers first
    app.include_router(create_router())
    
    # Add middleware (order matters - last added runs first due to LIFO)
    _add_cors_logging_middleware(app, allowed_origins)
    _add_security_middleware(app)
    _add_cors_middleware(app, allowed_origins)  # Must be added LAST (runs FIRST)
    
    return app


# For Modal deployment
app = create_app()
