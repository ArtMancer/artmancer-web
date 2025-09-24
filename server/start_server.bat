@echo off
REM Startup script for ArtMancer Web Server (Windows)

echo 🚀 Starting ArtMancer Web Server...

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating from .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo ✅ .env file created. Please add your GEMINI_API_KEY.
        echo 📝 Edit .env file and add your Gemini API key, then run this script again.
        pause
        exit /b 1
    ) else (
        echo ❌ .env.example file not found. Please create .env file manually.
        pause
        exit /b 1
    )
)

REM Check if GEMINI_API_KEY is set (basic check)
findstr /C:"GEMINI_API_KEY=" ".env" >nul
if errorlevel 1 (
    echo ❌ GEMINI_API_KEY not found in .env file.
    echo 📝 Please add your Gemini API key to the .env file.
    pause
    exit /b 1
)

REM Install dependencies if needed
echo 📦 Installing dependencies...
uv sync

REM Start the server
echo 🌐 Starting FastAPI server...
echo 📋 Server will be available at: http://localhost:8000
echo 📚 API docs will be available at: http://localhost:8000/docs
echo 🔗 ReDoc docs will be available at: http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop the server
echo.

uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000