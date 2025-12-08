"""
Core package for ArtMancer backend.

This package contains core functionality including:
- config: Application settings and configuration
- pipeline: Model pipeline loading and device management
- qwen_loader: Qwen model loading and adapter management

Import Strategy:
Core modules are imported individually to avoid loading unnecessary dependencies.
- Heavy Service: from app.core.pipeline import load_pipeline
- Light Service: from app.core.config import settings
"""
