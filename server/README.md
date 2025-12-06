# ArtMancer Web - Server

FastAPI backend for local Qwen image-editing with mask-aware conditioning.

## Local Development Setup

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

## Modal Deployment

### Prerequisites

- Modal account (sign up at [modal.com](https://modal.com))
- Checkpoint files (insertion_cp.safetensors, removal_cp.safetensors, wb_cp.safetensors)
- Python 3.12+

### Step 1: Install and Setup Modal

1. Install Modal:

   ```bash
   pip install modal
   ```

   Or if using uv:

   ```bash
   uv pip install modal
   ```

2. Authenticate with Modal:

   ```bash
   modal setup
   ```

   This will open a browser to authenticate. Follow the instructions to complete setup.

### Step 2: Upload Checkpoints to Modal Volume

1. **Create and upload checkpoints to Modal Volume**:

   ```bash
   cd server

   # Start a Modal shell to upload files
   modal run modal_app.py::volume

   # Or use Modal CLI to upload files
   # First, create a temporary function to upload files
   ```

   **Alternative: Use Modal Volume API directly**

   Create a temporary script to upload checkpoints:

   ```python
   # upload_checkpoints_modal.py
   import modal

   app = modal.App("upload-checkpoints")
   volume = modal.Volume.from_name("artmancer-checkpoints", create_if_missing=True)

   @app.function(volumes={"/checkpoints": volume})
   def upload_files():
       import shutil
       from pathlib import Path
       
       local_checkpoints = Path("./checkpoints")
       modal_checkpoints = Path("/checkpoints/checkpoints")
       modal_checkpoints.mkdir(parents=True, exist_ok=True)
       
       for file in ["insertion_cp.safetensors", "removal_cp.safetensors", "wb_cp.safetensors"]:
           local_file = local_checkpoints / file
           if local_file.exists():
               shutil.copy2(local_file, modal_checkpoints / file)
               print(f"Uploaded {file}")
           else:
               print(f"Warning: {file} not found")
       
       volume.commit()

   if __name__ == "__main__":
       with app.run():
           upload_files.remote()
   ```

   Run the upload script:

   ```bash
   modal run upload_checkpoints_modal.py
   ```

   **Or use Modal Volume mount locally**:

   ```bash
   # Mount volume locally (if supported)
   modal volume mount artmancer-checkpoints /tmp/modal-volume
   cp checkpoints/*.safetensors /tmp/modal-volume/checkpoints/
   modal volume unmount artmancer-checkpoints
   ```

### Step 3: Deploy to Modal

Deploy both endpoints:

```bash
cd server
modal deploy modal_app.py
```

This will deploy two web endpoints:
- **Heavy endpoint** (`fastapi_app_heavy`): A100 80GB GPU for `/api/generate`
- **Light endpoint** (`fastapi_app_light`): T4 GPU for lightweight tasks

### Step 4: Get Endpoint URLs

After deployment, Modal will provide URLs for both endpoints. You can find them:

1. In the terminal output after `modal deploy`
2. In the Modal dashboard at [modal.com](https://modal.com)
3. Using Modal CLI:

   ```bash
   modal app list
   modal app show artmancer
   ```

Example URLs:
- Heavy endpoint: `https://your-username--artmancer-fastapi-app-heavy.modal.run`
- Light endpoint: `https://your-username--artmancer-fastapi-app-light.modal.run`

### Step 5: Test Deployment

1. Test health check endpoint:

   ```bash
   curl https://YOUR_HEAVY_ENDPOINT_URL/ping
   ```

   Should return `{"status": "healthy"}`.

2. Test generation endpoint (heavy):

   ```bash
   curl -X POST https://YOUR_HEAVY_ENDPOINT_URL/api/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "test", "input_image": "base64..."}'
   ```

3. Test light endpoint:

   ```bash
   curl https://YOUR_LIGHT_ENDPOINT_URL/api/health
   ```

### Step 6: Update Frontend

Update your frontend to use the Modal endpoints. Set environment variables in `.env.local`:

```env
# Heavy endpoint for generation
NEXT_PUBLIC_API_GENERATE_URL=https://YOUR_HEAVY_ENDPOINT_URL

# Light endpoint for other tasks
NEXT_PUBLIC_API_LIGHT_URL=https://YOUR_LIGHT_ENDPOINT_URL

# Or use a single base URL and route appropriately
NEXT_PUBLIC_API_URL=https://YOUR_HEAVY_ENDPOINT_URL
```

### Local Testing with Modal

Test the Modal app locally before deploying:

```bash
# Serve locally (development mode)
modal serve modal_app.py

# Or run a specific function
modal run modal_app.py::fastapi_app_heavy
```

### Modal-Specific Notes

- **Auto-scaling**: Modal automatically scales endpoints based on traffic
- **Cold starts**: Sub-second cold starts with Modal's optimized containers
- **GPU availability**: Modal pools capacity across major clouds for better availability
- **Pricing**: Pay per second of GPU/CPU usage
- **Volumes**: Modal Volumes are persistent and shared across function invocations
- **Concurrency**: 
  - Heavy endpoint: 3 concurrent requests (A100 is expensive)
  - Light endpoint: 20 concurrent requests (T4 is cheaper)
- **Timeouts**:
  - Heavy endpoint: 10 minutes (600 seconds)
  - Light endpoint: 5 minutes (300 seconds)

### Updating Checkpoints

To update checkpoints in the Modal Volume:

```bash
# Use the upload script again, or
modal volume ls artmancer-checkpoints
modal volume download artmancer-checkpoints /path/to/local
# Make changes, then upload back
``` 

## API Endpoints

The backend is deployed as two separate Modal endpoints:

### Heavy Endpoint (A100 80GB)
- `GET /ping` – Health check endpoint
- `GET /api/health` – Device + pipeline status (detailed health info)
- `POST /api/generate` – Edit an image using prompt + mask with 3 conditional inputs (mask/background/object) - **Requires A100 80GB**

### Light Endpoint (T4/CPU)
- `GET /ping` – Health check endpoint
- `GET /api/health` – Device + pipeline status
- `GET /api/` – Root endpoint
- `POST /api/clear-cache` – Release GPU / CPU memory
- `POST /api/smart-mask` – Generate smart mask using FastSAM
- `POST /api/image-utils/*` – Image utilities (background removal, etc.)
- `POST /api/debug/*` – Debug session management

## Request requirements

- `input_image`: base64 original image from frontend
- `mask_image`: base64 mask coming from frontend
- The backend splits the original/mask into:
  - mask RGB
  - background (original minus mask)
  - foreground (mask \* original)
    which are passed to the Qwen model as conditional inputs.

## Environment Variables

- `MODEL_FILE` – path to the `.safetensors` checkpoint (default: `./qwen_2509_object_insertion_512_000002750.safetensors`)
- `HOST`, `PORT`, `DEBUG` – optional overrides for server process
- `ALLOWED_ORIGINS` – comma-separated list for CORS (defaults to `http://localhost:3000`)
