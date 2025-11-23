#!/bin/bash
# Startup script for ArtMancer Web Server

echo "🚀 Starting ArtMancer Web Server..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  .env file not found. Creating from .env.example..."
        cp .env.example .env
        echo "✅ .env file created. Update MODEL_FILE if your checkpoint is stored elsewhere."
    else
        echo "⚠️ .env file not found. Creating a minimal one..."
        cat <<'EOF' > .env
MODEL_FILE=./qwen_2509_object_insertion_512_000002750.safetensors
EOF
        echo "✅ Created .env with default MODEL_FILE."
    fi
fi

# Ensure model files exist
MODEL_FILE_INSERTION=$(grep "^MODEL_FILE_INSERTION=" .env | cut -d'=' -f2-)
MODEL_FILE_INSERTION=${MODEL_FILE_INSERTION:-./qwen_2509_object_insertion_512_000002750.safetensors}
MODEL_FILE_REMOVAL=$(grep "^MODEL_FILE_REMOVAL=" .env | cut -d'=' -f2-)
MODEL_FILE_REMOVAL=${MODEL_FILE_REMOVAL:-./qwen2509_object_removal_512_000002500.safetensors}

if [ ! -f "$MODEL_FILE_INSERTION" ]; then
    echo "❌ Insertion model file not found at $MODEL_FILE_INSERTION"
    echo "📝 Update MODEL_FILE_INSERTION in .env to point to your safetensors checkpoint."
    exit 1
fi

if [ ! -f "$MODEL_FILE_REMOVAL" ]; then
    echo "❌ Removal model file not found at $MODEL_FILE_REMOVAL"
    echo "📝 Update MODEL_FILE_REMOVAL in .env to point to your safetensors checkpoint."
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
PORT=${PORT:-8003}
DEBUG=${DEBUG:-True}

# Build uvicorn command
UVICORN_CMD="uv run uvicorn main:app --host ${HOST} --port ${PORT}"

# Add --reload only if DEBUG is True
if [ "${DEBUG}" = "True" ] || [ "${DEBUG}" = "true" ] || [ "${DEBUG}" = "1" ]; then
    UVICORN_CMD="${UVICORN_CMD} --reload"
    echo "🔧 Running in DEBUG mode (auto-reload enabled)"
else
    echo "🚀 Running in PRODUCTION mode (no auto-reload)"
fi

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📋 Server will be available at: http://localhost:${PORT}"
echo "📚 API docs will be available at: http://localhost:${PORT}/docs"
echo "🔗 ReDoc docs will be available at: http://localhost:${PORT}/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

eval $UVICORN_CMD