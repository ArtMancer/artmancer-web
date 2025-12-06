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

    # Configure CORS with explicit settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (POST, GET, PUT, DELETE, OPTIONS, etc.)
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"],  # Expose all headers in response
    )
    
    # Log CORS requests for debugging
    @app.middleware("http")
    async def log_cors_requests(request, call_next):
        origin = request.headers.get("origin")
        if origin:
            print(f"🌐 [API Gateway] CORS request from origin: {origin}")
        response = await call_next(request)
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

    # Include routers
    app.include_router(create_router())

    return app


# For Modal deployment
app = create_app()

