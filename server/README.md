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
   - Go to Storage → New Network Volume
   - Select datacenter (EU-RO-1 recommended for S3 API support)
   - Set size (e.g., 10GB for 3 checkpoint files ~850MB total)
   - Note: S3-compatible API is available for: EUR-IS-1, EU-RO-1, EU-CZ-1, US-KS-2, US-CA-2
   - Reference: https://docs.runpod.io/storage/network-volumes

2. **Create S3 API Key** (separate from your regular RunPod API key):
   - Go to RunPod Console → Settings → S3 API Keys
   - Click "Create an S3 API key"
   - Save the **access key** (user_XXXXX) and **secret** (rps_XXXXX)
   - The access key is your User ID (found in the key description)
   - Reference: https://docs.runpod.io/storage/s3-api#setup-and-authentication

3. Upload checkpoint files to the volume using S3-compatible API:

   **Option A: Use the provided Python upload script (recommended)**

   ```bash
   cd server

   # Create .env file with S3 API credentials
   echo 'RUNPOD_S3_USER_ID=user_XXXXX' > .env
   echo 'RUNPOD_S3_SECRET=rps_XXXXX' >> .env
   echo 'RUNPOD_S3_BUCKET=YOUR_VOLUME_ID' >> .env

   # Run upload script
   uv run python upload_checkpoints_to_runpod.py
   ```

   Or set environment variables directly:

   ```bash
   export RUNPOD_S3_USER_ID='user_XXXXX'
   export RUNPOD_S3_SECRET='rps_XXXXX'
   export RUNPOD_S3_BUCKET='YOUR_VOLUME_ID'
   uv run python upload_checkpoints_to_runpod.py
   ```

   **Option B: Manual upload with AWS CLI**

   ```bash
   # Install AWS CLI if not already installed
   pip install awscli

   # Configure AWS CLI with S3 API credentials
   aws configure
   # AWS Access Key ID: Enter your User ID (user_XXXXX)
   # AWS Secret Access Key: Enter your S3 API secret (rps_XXXXX)
   # Default region: eu-ro-1 (or your datacenter)
   # Default output format: json

   # Upload each file (replace YOUR_VOLUME_ID with your Network Volume ID)
   aws s3 cp checkpoints/insertion_cp.safetensors \
     s3://YOUR_VOLUME_ID/checkpoints/insertion_cp.safetensors \
     --region eu-ro-1 \
     --endpoint-url https://s3api-eu-ro-1.runpod.io

   aws s3 cp checkpoints/removal_cp.safetensors \
     s3://YOUR_VOLUME_ID/checkpoints/removal_cp.safetensors \
     --region eu-ro-1 \
     --endpoint-url https://s3api-eu-ro-1.runpod.io

   aws s3 cp checkpoints/wb_cp.safetensors \
     s3://YOUR_VOLUME_ID/checkpoints/wb_cp.safetensors \
     --region eu-ro-1 \
     --endpoint-url https://s3api-eu-ro-1.runpod.io

   # Verify upload
   aws s3 ls --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io s3://YOUR_VOLUME_ID/checkpoints/
   ```

3. **Important**: Files uploaded to `s3://bucket/checkpoints/` will be accessible at `/runpod-volume/checkpoints/` on Serverless workers

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

### RunPod healthcheck helper (SDK-only)

I added a helper script `runpod_healthcheck.py` at the repo root (`server/runpod_healthcheck.py`). It:
- Uses the official `runpod` Python SDK and calls `Endpoint.health()` to verify endpoint status
- Does not perform any HTTP fallback or DNS/TCP checks — using the SDK ensures consistent behavior and helps match the RunPod control plane

Note: This tool requires `runpod` SDK to be installed in your environment. Install with:

```bash
# from server directory
uv run python -m pip install runpod
```

Usage:

```bash
# Load the env or export variables manually
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_URL="https://pov3ewvy1mejeo.api.runpod.ai"
python runpod_healthcheck.py
```

This is useful to quickly verify endpoint status using the official SDK.

Auth scheme options for `test_generate.py`:

- `key` (default): `Authorization: Key <API_KEY>`
- `bearer`: `Authorization: Bearer <API_KEY>`
- `raw`: `Authorization: <API_KEY>` (no 'Key' prefix)
- `x-api-key`: `x-api-key: <API_KEY>`

If your endpoint returns 401, try alternate schemes using the `--auth-scheme` flag:

```bash
# from server directory
uv run python test_generate.py --auth-scheme raw --api-key $(sed -n 's/^RUNPOD_API_KEY=//p' .env)
``` 

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
