"""
API Endpoints Package

Endpoints are imported individually to avoid loading unnecessary dependencies.

Import Strategy:
- Heavy Service (A100 Worker): from app.api.endpoints import generation
- Light Service (CPU/T4): from app.api.endpoints import smart_mask, image_utils

Available Endpoints:
- generation: Async and sync image generation endpoints
- generation_async: Async job-based generation endpoints
- smart_mask: Smart mask segmentation using FastSAM/BiRefNet
- image_utils: Image processing utilities (extract object, etc.)
- system: Health checks and system information
- debug: Debugging endpoints for development
"""
