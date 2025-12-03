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

## RunPod Serverless Deployment

### Prerequisites

- RunPod account with API access
- Docker Hub or GitHub Container Registry account
- Checkpoint files (insertion_cp.safetensors, removal_cp.safetensors, wb_cp.safetensors)

### Step 1: Setup Network Volume

1. Create a network volume on RunPod console for checkpoints
2. Upload checkpoint files to the volume:
   - `insertion_cp.safetensors`
   - `removal_cp.safetensors`
   - `wb_cp.safetensors`
3. Note the volume mount path (default: `/runpod-volume/`)

### Step 2: Build Docker Image

**Option A: Build locally and push to Docker Hub**

```bash
cd server  # Important: build from server/ directory
docker build --platform linux/amd64 -t yourusername/artmancer:latest .

# Push to Docker Hub
docker push yourusername/artmancer:latest
```

**Option B: Deploy directly from GitHub (RunPod)**

When deploying from GitHub on RunPod:

1. Repository: `ArtMancer/artmancer-web`
2. Branch: `main` (or your branch)
3. **Docker Build Context**: Set to `server` (important!)
4. Dockerfile Path: `server/Dockerfile` (or just `Dockerfile` if context is `server`)

### Step 3: Deploy on RunPod

1. Go to RunPod Console → Serverless → New Endpoint
2. Select **Load Balancer** endpoint type (not Queue)
3. Configure endpoint:

   - **Container Image**: `docker.io/yourusername/artmancer:latest`
   - **GPU**: A100-80GB (or test with smaller GPU for cost optimization)
   - **Network Volumes**: Mount your checkpoint volume at `/runpod-volume/`
   - **Environment Variables**:
     - `PORT=80`
     - `PORT_HEALTH=80`
     - `PYTORCH_ALLOC_CONF=expandable_segments:True`
     - `MODEL_FILE_INSERTION=/runpod-volume/checkpoints/insertion_cp.safetensors`
     - `MODEL_FILE_REMOVAL=/runpod-volume/checkpoints/removal_cp.safetensors`
     - `MODEL_FILE_WHITE_BALANCE=/runpod-volume/checkpoints/wb_cp.safetensors`
   - **Expose HTTP Ports**: Port 80
   - **FlashBoot**: Enable (reduces cold start <200ms)
   - **Worker Configuration**:
     - Min workers: 0 or 1
     - Max workers: 3-5
     - Timeout: 600 seconds

4. Click **Deploy Endpoint**

### Step 4: Test Deployment

1. Test health check endpoint:

   ```bash
   curl https://YOUR_ENDPOINT_ID.api.runpod.ai/ping
   ```

   Should return `{"status": "healthy"}` when ready, or `204` when initializing.

2. Test generation endpoint:
   ```bash
   curl -X POST https://YOUR_ENDPOINT_ID.api.runpod.ai/api/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "test", "input_image": "base64..."}'
   ```

### Step 5: Update Frontend (Optional)

The frontend is already configured to use the default RunPod endpoint: `https://pov3ewvy1mejeo.api.runpod.ai`

To use a different endpoint, set environment variable in `.env.local`:

```
NEXT_PUBLIC_RUNPOD_GENERATE_URL=https://YOUR_ENDPOINT_ID.api.runpod.ai
```

Or use `NEXT_PUBLIC_API_URL` to override all endpoints.

### Local Testing

Test the RunPod handler locally:

```bash
# Install uvicorn if not already installed
pip install uvicorn

# Run the handler
uvicorn rp_handler:app --host 0.0.0.0 --port 80

# Test health check
curl http://localhost:80/ping

# Test API endpoint
curl http://localhost:80/api/health
```

### RunPod-Specific Notes

- **Timeouts**:
  - Request timeout: 2 minutes (no worker available) → 400 error
  - Processing timeout: 5.5 minutes → 524 error
  - Misconfigured timeout: 8 minutes → 502 error
- **Payload Limit**: 30 MB for both requests and responses
- **Cold Start**: Workers need time to initialize. Frontend includes retry logic.
- **Health Check**: `/ping` endpoint is required by RunPod Load Balancer
  - Returns `200` when healthy
  - Returns `204` when initializing
  - Returns error status when unhealthy

## API Endpoints

- `GET /ping` – Health check endpoint (required for RunPod Load Balancer)
- `GET /api/health` – Device + pipeline status (detailed health info)
- `POST /api/generate` – Edit an image using prompt + mask with 3 conditional inputs (mask/background/object)
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
