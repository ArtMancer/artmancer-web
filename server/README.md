# ArtMancer Web - Server

FastAPI backend for local Qwen image-editing with mask-aware conditioning.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create `.env` (if not present) and point `MODEL_FILE` to your safetensors checkpoint:

```
MODEL_FILE=./qwen_2509_object_insertion_512_000002750.safetensors
```

3. Start the server:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8003
```

or use the helper script:

```bash
./start_server.sh
```

## API Endpoints

- `GET /api/health` – Device + pipeline status
- `POST /api/generate` – Edit an image using prompt + mask with 3 conditional inputs (mask/background/object)
- `POST /api/clear-cache` – Release GPU / CPU memory

## Request requirements

- `input_image`: base64 original image from frontend
- `mask_image`: base64 mask coming from frontend
- The backend splits the original/mask into:
  - mask RGB
  - background (original minus mask)
  - foreground (mask * original)
  which are passed to the Qwen model as conditional inputs.

## Environment Variables

- `MODEL_FILE` – path to the `.safetensors` checkpoint (default: `./qwen_2509_object_insertion_512_000002750.safetensors`)
- `HOST`, `PORT`, `DEBUG` – optional overrides for server process
- `ALLOWED_ORIGINS` – comma-separated list for CORS (defaults to `http://localhost:3000`)
