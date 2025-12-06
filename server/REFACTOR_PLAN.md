# Refactor Plan: Backend Architecture Migration

## Mục tiêu

Chuyển từ kiến trúc light/heavy containers với wake-up logic sang kiến trúc microservices độc lập với API Gateway.

## Cấu trúc mới

```
server/
├── api_gateway/           # API Gateway - entry point
│   ├── __init__.py
│   ├── main.py           # FastAPI app chính
│   ├── router.py         # Routing logic
│   └── middleware.py     # CORS, logging, etc.
│
├── services/             # Các service độc lập
│   ├── generation/       # Qwen image generation
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── endpoints.py
│   ├── inference/        # Model inference (có thể merge với generation)
│   ├── segmentation/     # FastSAM mask generation
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── endpoints.py
│   ├── image_utils/      # Image processing utilities
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── endpoints.py
│   ├── job_manager/      # Async job management
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── endpoints.py
│   └── upload/           # File upload (nếu cần)
│
├── shared/               # Shared code
│   ├── utils/            # Common utilities
│   ├── clients/          # HTTP clients cho inter-service communication
│   └── schemas/          # Pydantic models
│
└── modal_app.py          # Modal deployment config
```

## Files cần xóa

### Frontend:

- `client/src/app/api/wake-up/route.ts`
- `client/src/components/BackendWarmer.tsx`
- `client/src/components/SmartWakeUp.tsx`
- `client/src/app/api/server-status/route.ts` (hoặc refactor)
- `client/src/contexts/ServerContext.tsx` (refactor để loại bỏ wake-up logic)

### Backend:

- `server/modal_app.py` - function `warmup_services()`
- Tất cả logic wake-up trong các service

## Files mới

1. `server/api_gateway/main.py` - API Gateway FastAPI app
2. `server/api_gateway/router.py` - Routing logic
3. `server/shared/clients/service_client.py` - HTTP client cho inter-service calls
4. `server/shared/schemas/` - Unified schemas

## Migration Steps

1. ✅ Phân tích codebase hiện tại
2. ⏳ Tạo cấu trúc thư mục mới
3. ⏳ Tạo API Gateway
4. ⏳ Refactor các service vào cấu trúc mới
5. ⏳ Tạo shared modules
6. ⏳ Xóa code cũ
7. ⏳ Cập nhật frontend
8. ⏳ Cập nhật Modal deployment
9. ⏳ Tạo diagram và docs
