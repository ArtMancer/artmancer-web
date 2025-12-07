"""API Gateway - Main entry point for all requests."""

from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .router import create_router


def create_app() -> FastAPI:
    """Create FastAPI app for API Gateway."""
    # Get allowed origins from environment
    allowed_origins_str = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,https://localhost:3000"
    )
    allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
    
    # Log CORS configuration for debugging
    print(f"🌐 [API Gateway] CORS configured with allowed origins: {allowed_origins}")

    app = FastAPI(
        title="ArtMancer API Gateway",
        description="API Gateway for routing requests to microservices",
        version="3.0.0",
    )

    # Include routers first
    app.include_router(create_router())
    
    # Log CORS requests for debugging and ensure CORS headers on all responses
    @app.middleware("http")
    async def log_cors_requests(request, call_next):
        origin = request.headers.get("origin")
        if origin:
            print(f"🌐 [API Gateway] CORS request from origin: {origin}")
        response = await call_next(request)
        
        # Ensure CORS headers are present on all responses (including redirects)
        # This is important for Modal's 303 redirects
        if origin and origin in allowed_origins:
            if "access-control-allow-origin" not in response.headers:
                response.headers["access-control-allow-origin"] = origin
            if "access-control-allow-credentials" not in response.headers:
                response.headers["access-control-allow-credentials"] = "true"
            if "access-control-allow-methods" not in response.headers:
                response.headers["access-control-allow-methods"] = "*"
            if "access-control-allow-headers" not in response.headers:
                response.headers["access-control-allow-headers"] = "*"
        
        # Log CORS headers in response
        cors_headers = {
            k: v for k, v in response.headers.items() 
            if k.lower().startswith("access-control")
        }
        if cors_headers:
            print(f"🌐 [API Gateway] CORS response headers: {cors_headers}")
        return response
    
    # Remove server header for security
    @app.middleware("http")
    async def remove_server_header(request, call_next):
        response = await call_next(request)
        if "server" in response.headers:
            del response.headers["server"]
        return response
    
    # Configure CORS with explicit settings - MUST be added LAST (runs FIRST due to LIFO)
    # FastAPI middleware runs in reverse order (LIFO), so this will run first
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (POST, GET, PUT, DELETE, OPTIONS, etc.)
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"],  # Expose all headers in response
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    return app


# For Modal deployment
app = create_app()

