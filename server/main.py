from app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings

    print(f"🚀 Starting ArtMancer backend on {settings.host}:{settings.port}")
    print(f"📚 API docs: http://localhost:{settings.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
