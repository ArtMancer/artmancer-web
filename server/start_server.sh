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

# Load environment variables from .env file
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
fi

# Set defaults if not specified in .env
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DEBUG=${DEBUG:-True}

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📋 Server will be available at: http://localhost:${PORT}"
echo "📚 API docs will be available at: http://localhost:${PORT}/docs"
echo "🔗 ReDoc docs will be available at: http://localhost:${PORT}/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run uvicorn main:app --reload --host ${HOST} --port ${PORT}