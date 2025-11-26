#!/bin/bash
# Fix virtual environment location for WSL
# Moves .venv from Windows filesystem to Linux filesystem to avoid permission issues

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_WINDOWS="$SCRIPT_DIR/.venv"

# Check if we're on Windows filesystem (starts with /mnt/)
if [[ "$SCRIPT_DIR" == /mnt/* ]]; then
    echo "⚠️  Project is on Windows filesystem. Moving .venv to Linux filesystem..."
    
    # Use Linux filesystem for .venv (in user's home)
    VENV_LINUX="$HOME/.cache/artmancer-web-venv"
    
    # Remove old .venv if it exists
    if [ -d "$VENV_WINDOWS" ]; then
        echo "🗑️  Removing old .venv from Windows filesystem..."
        rm -rf "$VENV_WINDOWS"
    fi
    
    # Create symlink from project to Linux filesystem venv
    echo "🔗 Creating symlink to Linux filesystem .venv..."
    ln -sf "$VENV_LINUX" "$VENV_WINDOWS"
    
    # Set UV_PROJECT_ENVIRONMENT to use Linux filesystem
    export UV_PROJECT_ENVIRONMENT="$VENV_LINUX"
    
    echo "✅ Virtual environment will be created at: $VENV_LINUX"
    echo "📝 Run 'uv sync' to install dependencies on Linux filesystem"
else
    echo "✅ Project is already on Linux filesystem. No action needed."
fi

