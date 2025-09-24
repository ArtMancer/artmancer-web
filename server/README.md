# ArtMancer Web - Server

FastAPI server for Gemini-powered image generation with customizable model settings.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Set up environment variables:

```bash
cp .env.example .env
```

3. Add your Gemini API key to the `.env` file:

```
GEMINI_API_KEY=your_actual_api_key_here
```

## Running the Server

Development mode with auto-reload:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using the built-in runner:

```bash
uv run python main.py
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - Available Gemini models
- `GET /api/config` - API configuration
- `POST /api/generate` - Generate single image
- `POST /api/generate/batch` - Generate multiple images

## API Documentation

Once running, visit:

- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

- `GEMINI_API_KEY` - Your Gemini API key (required)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: True)
- `ALLOWED_ORIGINS` - CORS allowed origins (default: localhost:3000,3001)
