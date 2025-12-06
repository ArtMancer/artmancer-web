# Backend Refactor - Complete Documentation

## Tổng quan

Refactor toàn bộ backend từ kiến trúc light/heavy containers với wake-up logic sang kiến trúc microservices độc lập với API Gateway.

## Kiến trúc mới

```
┌─────────────┐
│  Frontend   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   API Gateway    │  ← Entry point duy nhất
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    │         │          │              │
    ▼         ▼          ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
│Generation│ │Segmentation│ │Image Utils│ │Job Manager │
│ Service │ │  Service   │ │  Service  │ │  Service   │
│ (A100)  │ │   (T4)     │ │  (CPU)    │ │   (CPU)    │
└────────┘ └──────────┘ └──────────┘ └─────────────┘
```

## Files mới

### 1. API Gateway

- `server/api_gateway/__init__.py`
- `server/api_gateway/main.py` - FastAPI app chính
- `server/api_gateway/router.py` - Routing logic

### 2. Shared Modules

- `server/shared/__init__.py`
- `server/shared/utils/__init__.py`
- `server/shared/clients/__init__.py`
- `server/shared/clients/service_client.py` - HTTP client cho inter-service communication
- `server/shared/schemas/__init__.py`

### 3. Services (cần refactor tiếp)

- `server/services/generation/` - Từ generation_service.py
- `server/services/segmentation/` - Từ mask_segmentation_service.py
- `server/services/image_utils/` - Từ image_utils endpoints
- `server/services/job_manager/` - Từ generation_async.py

## Files cần xóa

### Frontend:

1. `client/src/app/api/wake-up/route.ts` - Wake-up API route
2. `client/src/components/BackendWarmer.tsx` - Backend warmer component
3. `client/src/components/SmartWakeUp.tsx` - Smart wake-up component
4. `client/src/app/api/server-status/route.ts` - Server status check (refactor để dùng API Gateway)

### Backend:

1. `server/modal_app.py` - Function `warmup_services()` (dòng 842-887)

## Files cần thay đổi

### Frontend:

1. `client/src/contexts/ServerContext.tsx` - Loại bỏ wake-up logic, chỉ giữ health check
2. `client/src/services/api.ts` - Cập nhật base URL sang API Gateway
3. `client/src/components/ServerControl.tsx` - Loại bỏ wake-up logic

### Backend:

1. `server/modal_app.py` - Thêm API Gateway service, loại bỏ warmup logic
2. Tất cả service endpoints - Đảm bảo có `/api/health` endpoint

## Hướng dẫn deploy

### 1. Deploy API Gateway

```bash
# Trong modal_app.py, thêm:
@app.cls(
    image=cpu_image,  # Hoặc tạo image riêng cho gateway
    cpu=1,
    timeout=60,
    min_containers=1,  # Always-on
)
class APIGatewayService:
    @modal.asgi_app(label="api-gateway")
    def serve(self):
        from api_gateway.main import create_app
        return create_app()
```

### 2. Deploy các service

Các service hiện có (QwenService, FastSAMService, ImageUtilsService, JobManagerService) giữ nguyên, chỉ cần đảm bảo có `/api/health` endpoint.

### 3. Cấu hình environment variables

```bash
# API Gateway cần biết URLs của các service
export GENERATION_SERVICE_URL=https://nxan2911--qwen.modal.run
export SEGMENTATION_SERVICE_URL=https://nxan2911--fastsam.modal.run
export IMAGE_UTILS_SERVICE_URL=https://nxan2911--image-utils.modal.run
export JOB_MANAGER_SERVICE_URL=https://nxan2911--job-manager.modal.run
```

### 4. Cập nhật frontend

```typescript
// client/src/services/api.ts
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
  "https://nxan2911--api-gateway.modal.run";
```

## Backward Compatibility

API Gateway giữ nguyên tất cả endpoint paths:

- `/api/generate` → Generation Service
- `/api/generate/async` → Job Manager
- `/api/generate/status/{task_id}` → Job Manager
- `/api/generate/result/{task_id}` → Job Manager
- `/api/smart-mask` → Segmentation Service
- `/api/image-utils/*` → Image Utils Service
- `/api/system/health` → Aggregate health từ tất cả services

Frontend chỉ cần đổi base URL, không cần thay đổi endpoint paths.

## Migration Steps

1. ✅ Tạo cấu trúc thư mục mới
2. ✅ Tạo API Gateway
3. ✅ Tạo shared modules
4. ⏳ Refactor các service vào cấu trúc mới
5. ⏳ Xóa code wake-up
6. ⏳ Cập nhật frontend
7. ⏳ Cập nhật Modal deployment
8. ⏳ Testing

## Notes

- API Gateway là always-on service (CPU-only, rẻ)
- Các service khác vẫn scale-to-zero (cold start)
- Không còn wake-up logic, services tự động start khi có request
- Frontend chỉ cần gọi API Gateway, không cần biết service URLs
