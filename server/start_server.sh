#!/bin/bash
# Startup script for ArtMancer Web Server

echo "🚀 Starting ArtMancer Web Server..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  .env file not found. Creating from .env.example..."
        cp .env.example .env
        echo "✅ .env file created. Update MODEL_FILE_INSERTION and MODEL_FILE_REMOVAL if your checkpoints are stored elsewhere."
    else
        echo "⚠️ .env file not found. Creating a minimal one..."
        cat <<'EOF' > .env
MODEL_FILE_INSERTION=./checkpoints/qwen_2509_object_insertion_512_000002750.safetensors
MODEL_FILE_REMOVAL=./checkpoints/qwen_2509_object_removal_512_000007500.safetensors
EOF
        echo "✅ Created .env with default MODEL_FILE_INSERTION and MODEL_FILE_REMOVAL."
    fi
fi

# Ensure model files exist
MODEL_FILE_INSERTION=$(grep "^MODEL_FILE_INSERTION=" .env | cut -d'=' -f2-)
MODEL_FILE_INSERTION=${MODEL_FILE_INSERTION:-./checkpoints/qwen_2509_object_insertion_512_000002750.safetensors}
MODEL_FILE_REMOVAL=$(grep "^MODEL_FILE_REMOVAL=" .env | cut -d'=' -f2-)
MODEL_FILE_REMOVAL=${MODEL_FILE_REMOVAL:-./checkpoints/qwen_2509_object_removal_512_000007500.safetensors}

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

# Fix virtual environment location for WSL (avoid permission issues on Windows filesystem)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == /mnt/* ]] && [ -d "$SCRIPT_DIR/.venv" ]; then
    # Check if .venv is a symlink (already fixed) or real directory (needs fixing)
    if [ ! -L "$SCRIPT_DIR/.venv" ]; then
        echo "⚠️  Virtual environment on Windows filesystem detected."
        echo "🔧 Moving .venv to Linux filesystem to avoid permission issues..."
        VENV_LINUX="$HOME/.cache/artmancer-web-venv"
        rm -rf "$SCRIPT_DIR/.venv"
        ln -sf "$VENV_LINUX" "$SCRIPT_DIR/.venv"
        export UV_PROJECT_ENVIRONMENT="$VENV_LINUX"
        echo "✅ Using Linux filesystem for .venv: $VENV_LINUX"
    fi
elif [[ "$SCRIPT_DIR" == /mnt/* ]]; then
    # Project on Windows filesystem but no .venv yet - use Linux filesystem
    VENV_LINUX="$HOME/.cache/artmancer-web-venv"
    export UV_PROJECT_ENVIRONMENT="$VENV_LINUX"
    echo "📦 Virtual environment will be created on Linux filesystem: $VENV_LINUX"
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
