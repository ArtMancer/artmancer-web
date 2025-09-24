#!/bin/bash
# Startup script for ArtMancer Web Server

echo "🚀 Starting ArtMancer Web Server..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ .env file created. Please add your GEMINI_API_KEY."
        echo "📝 Edit .env file and add your Gemini API key, then run this script again."
        exit 1
    else
        echo "❌ .env.example file not found. Please create .env file manually."
        exit 1
    fi
fi

# Check if GEMINI_API_KEY is set
if ! grep -q "^GEMINI_API_KEY=.*[^[:space:]]" .env; then
    echo "❌ GEMINI_API_KEY not found in .env file."
    echo "📝 Please add your Gemini API key to the .env file."
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
uv sync

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📋 Server will be available at: http://localhost:8000"
echo "📚 API docs will be available at: http://localhost:8000/docs"
echo "🔗 ReDoc docs will be available at: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000