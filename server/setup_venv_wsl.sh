#!/bin/bash
# Setup virtual environment on Linux filesystem for WSL
# Run this once to fix permission issues when installing packages like torchvision

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SCRIPT_DIR" != /mnt/* ]]; then
    echo "✅ Project is already on Linux filesystem. No action needed."
    exit 0
fi

echo "🔧 Setting up virtual environment on Linux filesystem for WSL..."

VENV_LINUX="$HOME/.cache/artmancer-web-venv"

# Remove old .venv if it exists on Windows filesystem
if [ -d "$SCRIPT_DIR/.venv" ] && [ ! -L "$SCRIPT_DIR/.venv" ]; then
    echo "🗑️  Removing old .venv from Windows filesystem..."
    rm -rf "$SCRIPT_DIR/.venv"
fi

# Create symlink to Linux filesystem
if [ ! -L "$SCRIPT_DIR/.venv" ]; then
    echo "🔗 Creating symlink to Linux filesystem .venv..."
    ln -sf "$VENV_LINUX" "$SCRIPT_DIR/.venv"
fi

# Set environment variable for uv
export UV_PROJECT_ENVIRONMENT="$VENV_LINUX"

echo "✅ Virtual environment will be at: $VENV_LINUX"
echo ""
echo "📦 Now run: uv sync"
echo "   This will install all dependencies on Linux filesystem (no permission issues)"

