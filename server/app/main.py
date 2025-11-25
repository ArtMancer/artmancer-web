from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.endpoints import evaluation, generation, system, white_balance, smart_mask, visualization, benchmark
from .core.config import settings
from .core.pipeline import clear_pipeline_cache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting ArtMancer backend")
    logger.info(f"🌐 Server configured to run on {settings.host}:{settings.port}")
    
    # Optionally preload models on startup for faster first request
    if settings.preload_models:
        logger.info("📥 Preloading models on startup (this may take a moment)...")
        try:
            from .core.pipeline import load_pipeline
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            # Preload models in background thread to not block startup
            def preload_models_sync():
                try:
                    # Load models sequentially to avoid memory spike
                    logger.info("📥 Preloading insertion model...")
                    load_pipeline(task_type="insertion")
                    logger.info("✅ Insertion model loaded")
                    
                    logger.info("📥 Preloading removal model...")
                    load_pipeline(task_type="removal")
                    logger.info("✅ Removal model loaded")
                    
                    logger.info("📥 Preloading white-balance model...")
                    load_pipeline(task_type="white-balance")
                    logger.info("✅ White-balance model loaded")
                    
                    logger.info("✅ All models preloaded successfully")
                except Exception as e:
                    logger.error(f"❌ Error preloading models: {e}")
            
            # Run preload in thread pool to not block async startup
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            loop.run_in_executor(executor, preload_models_sync)
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload models: {e}. Models will be loaded on first use.")
    else:
        logger.info("💡 Models will be loaded on first use (lazy loading). Set PRELOAD_MODELS=true to preload on startup.")
    
    yield
    clear_pipeline_cache()
    logger.info("🛑 Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=settings.description,
        version=settings.version,
        lifespan=lifespan,
    )

    # CORS configuration
    # Use configured origins from .env, or default to localhost:3000
    cors_origins = settings.allowed_origins if settings.allowed_origins else ["http://localhost:3000"]
    
    # In debug mode, automatically add common localhost ports for development convenience
    if settings.debug:
        common_ports = [3000, 3001, 3002, 3003, 3300, 5173, 5174]  # common dev ports
        for port in common_ports:
            origin = f"http://localhost:{port}"
            if origin not in cors_origins:
                cors_origins.append(origin)
        logger.info(f"🌐 CORS configured for origins (debug mode): {cors_origins}")
    else:
        logger.info(f"🌐 CORS configured for origins: {cors_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods including OPTIONS
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    # Include routers
    app.include_router(system.router)
    app.include_router(generation.router)
    app.include_router(evaluation.router)
    app.include_router(white_balance.router)
    app.include_router(smart_mask.router)
    app.include_router(visualization.router)
    app.include_router(benchmark.router)
    
    logger.info("✅ Routers registered:")
    logger.info("   - System: /api/health, /api/clear-cache")
    logger.info("   - Generation: /api/generate")
    logger.info("   - Evaluation: /api/evaluate")
    logger.info("   - White Balance: /api/white-balance")
    logger.info("   - Smart Mask: /api/smart-mask")
    logger.info("   - Visualization: /api/visualization")
    logger.info("   - Benchmark: /api/benchmark/run, /api/benchmark/upload")

    return app

