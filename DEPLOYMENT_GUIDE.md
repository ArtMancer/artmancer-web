# Deployment Guide - Multi-Service Architecture

## Tổng quan

Hệ thống mới sử dụng kiến trúc microservices với API Gateway làm entry point duy nhất. Các service độc lập, có thể scale riêng biệt.

## Prerequisites

- Modal account và token đã được cấu hình
- Environment variables đã được set
- Python 3.12+

## Deployment Steps

### 1. Deploy API Gateway

API Gateway là service always-on, CPU-only, rẻ nhất.

```python
# Trong modal_app.py, thêm:

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --no-cache-dir "
        "fastapi[standard]>=0.123.4 "
        "uvicorn[standard]>=0.23.0 "
        "httpx>=0.27.0 "
        "pydantic>=2.7.0 "
        "python-multipart "
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir("api_gateway", "/root/api_gateway")
    .add_local_dir("shared", "/root/shared")
)

@app.cls(
    image=gateway_image,
    cpu=1,  # Minimal CPU
    timeout=60,
    min_containers=1,  # Always-on
    scaledown_window=300,  # Scale down after 5 minutes of inactivity
)
class APIGatewayService:
    @modal.enter()
    def prepare(self):
        """Container startup."""
        print("🚀 [APIGatewayService] Container starting up...")
        print("✅ [APIGatewayService] Container ready!")
    
    @modal.asgi_app(label="api-gateway")
    def serve(self):
        """API Gateway endpoint."""
        from api_gateway.main import create_app
        return create_app()
```

Deploy:
```bash
modal deploy modal_app.py
```

URL sẽ là: `https://<username>--api-gateway.modal.run`

### 2. Cấu hình Environment Variables

API Gateway cần biết URLs của các service:

```bash
# Trong Modal dashboard hoặc .env
export GENERATION_SERVICE_URL=https://nxan2911--qwen.modal.run
export SEGMENTATION_SERVICE_URL=https://nxan2911--fastsam.modal.run
export IMAGE_UTILS_SERVICE_URL=https://nxan2911--image-utils.modal.run
export JOB_MANAGER_SERVICE_URL=https://nxan2911--job-manager.modal.run
```

Hoặc set trong `modal_app.py`:
```python
gateway_image = gateway_image.env({
    "GENERATION_SERVICE_URL": "https://nxan2911--qwen.modal.run",
    "SEGMENTATION_SERVICE_URL": "https://nxan2911--fastsam.modal.run",
    "IMAGE_UTILS_SERVICE_URL": "https://nxan2911--image-utils.modal.run",
    "JOB_MANAGER_SERVICE_URL": "https://nxan2911--job-manager.modal.run",
})
```

### 3. Deploy các Service hiện có

Các service hiện có (QwenService, FastSAMService, ImageUtilsService, JobManagerService) giữ nguyên, chỉ cần đảm bảo:

1. Có `/api/health` endpoint
2. Không có wake-up logic
3. Scale-to-zero (trừ JobManagerService là always-on)

### 4. Cập nhật Frontend

#### 4.1. Cập nhật API Base URL

```typescript
// client/src/services/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 
  'https://nxan2911--api-gateway.modal.run';
```

#### 4.2. Xóa wake-up logic

- Xóa `client/src/app/api/wake-up/route.ts`
- Xóa `client/src/components/BackendWarmer.tsx`
- Xóa `client/src/components/SmartWakeUp.tsx`
- Refactor `client/src/contexts/ServerContext.tsx` để chỉ check health

#### 4.3. Cập nhật Server Status

```typescript
// client/src/app/api/server-status/route.ts
export async function GET() {
  const GATEWAY_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 
    'https://nxan2911--api-gateway.modal.run';
  
  try {
    const response = await fetch(`${GATEWAY_URL}/api/system/health`, {
      method: 'GET',
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });
    
    if (response.ok) {
      const data = await response.json();
      return NextResponse.json({
        status: data.status === 'healthy' ? 'online' : 'offline',
        services: data.services,
      });
    }
  } catch (error) {
    console.error('Health check failed:', error);
  }
  
  return NextResponse.json({ status: 'offline' });
}
```

### 5. Testing

#### 5.1. Test API Gateway

```bash
# Health check
curl https://nxan2911--api-gateway.modal.run/api/health

# System health
curl https://nxan2911--api-gateway.modal.run/api/system/health
```

#### 5.2. Test Routing

```bash
# Generation
curl -X POST https://nxan2911--api-gateway.modal.run/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "input_image": "..."}'

# Smart mask
curl -X POST https://nxan2911--api-gateway.modal.run/api/smart-mask \
  -H "Content-Type: application/json" \
  -d '{"image": "...", "points": [[100, 100]]}'
```

## Cost Optimization

### API Gateway
- **Type**: CPU-only, always-on
- **Cost**: ~$0.0001/hour (rất rẻ)
- **Min containers**: 1 (always-on để giảm latency)

### Generation Service (A100)
- **Type**: A100 GPU, scale-to-zero
- **Cost**: ~$1.10/hour khi active
- **Min containers**: 0 (scale-to-zero để tiết kiệm)

### Segmentation Service (T4)
- **Type**: T4 GPU, scale-to-zero
- **Cost**: ~$0.20/hour khi active
- **Min containers**: 0 (scale-to-zero)

### Image Utils Service
- **Type**: CPU-only, scale-to-zero
- **Cost**: ~$0.0001/hour khi active
- **Min containers**: 0 (scale-to-zero)

### Job Manager Service
- **Type**: CPU-only, always-on
- **Cost**: ~$0.0001/hour
- **Min containers**: 1 (always-on để quản lý jobs)

## Monitoring

### Health Checks

API Gateway cung cấp aggregated health check:
```bash
GET /api/system/health
```

Response:
```json
{
  "status": "healthy",
  "services": {
    "generation": {"status": "healthy", ...},
    "segmentation": {"status": "healthy", ...},
    "image_utils": {"status": "healthy", ...},
    "job_manager": {"status": "healthy", ...}
  }
}
```

### Logs

Mỗi service có logs riêng trong Modal dashboard:
- API Gateway: `modal logs api-gateway`
- Generation: `modal logs qwen`
- Segmentation: `modal logs fastsam`
- Image Utils: `modal logs image-utils`
- Job Manager: `modal logs job-manager`

## Troubleshooting

### Service không khả dụng

1. Check service health:
```bash
curl https://nxan2911--api-gateway.modal.run/api/system/health
```

2. Check service logs:
```bash
modal logs <service-name>
```

3. Check environment variables:
```bash
modal env get <service-name>
```

### API Gateway không route được

1. Check service URLs trong environment variables
2. Check service có `/api/health` endpoint không
3. Check CORS configuration

### Cold start latency

- API Gateway: Always-on, không có cold start
- Các service khác: Scale-to-zero, có cold start 1-3s
- Có thể set `min_containers=1` cho service quan trọng (tăng cost)

## Rollback Plan

Nếu cần rollback về kiến trúc cũ:

1. Frontend: Đổi `API_BASE_URL` về service URLs cũ
2. Không cần thay đổi backend services (chúng vẫn hoạt động độc lập)
3. Xóa API Gateway service

## Next Steps

1. ✅ Deploy API Gateway
2. ⏳ Cập nhật frontend
3. ⏳ Testing
4. ⏳ Monitor và optimize

