# Files Changes Summary

## Files Mới (Created)

### API Gateway

1. `server/api_gateway/__init__.py`
2. `server/api_gateway/main.py` - FastAPI app chính cho API Gateway
3. `server/api_gateway/router.py` - Routing logic để forward requests

### Shared Modules

4. `server/shared/__init__.py`
5. `server/shared/utils/__init__.py`
6. `server/shared/clients/__init__.py`
7. `server/shared/clients/service_client.py` - HTTP client cho inter-service communication
8. `server/shared/schemas/__init__.py`

### Documentation

9. `server/REFACTOR_PLAN.md` - Kế hoạch refactor
10. `server/REFACTOR_STATUS.md` - Trạng thái refactor
11. `REFACTOR_COMPLETE.md` - Tài liệu hoàn chỉnh về refactor
12. `ARCHITECTURE_DIAGRAM.md` - Diagram kiến trúc mới
13. `FILES_CHANGES.md` - File này

## Files Cần Xóa (To Delete)

### Frontend

1. `client/src/app/api/wake-up/route.ts` - Wake-up API route
2. `client/src/components/BackendWarmer.tsx` - Backend warmer component
3. `client/src/components/SmartWakeUp.tsx` - Smart wake-up component

### Backend

4. `server/modal_app.py` - Function `warmup_services()` (dòng 842-887) - Xóa function này

## Files Cần Thay Đổi (To Modify)

### Frontend

1. `client/src/contexts/ServerContext.tsx`

   - Loại bỏ wake-up logic
   - Loại bỏ `toggleServer` function
   - Chỉ giữ health check
   - Cập nhật để dùng API Gateway

2. `client/src/app/api/server-status/route.ts`

   - Refactor để gọi API Gateway `/api/system/health` thay vì ping từng service
   - Loại bỏ wake-up logic

3. `client/src/services/api.ts`

   - Cập nhật `API_BASE_URL` để trỏ đến API Gateway
   - Loại bỏ `healthCheckWithRetry` logic (không cần wake-up)
   - Cập nhật tất cả service URLs

4. `client/src/components/ServerControl.tsx`
   - Loại bỏ wake-up logic
   - Chỉ hiển thị status, không có toggle on/off

### Backend

5. `server/modal_app.py`

   - Thêm API Gateway service
   - Xóa function `warmup_services()`
   - Đảm bảo tất cả services có `/api/health` endpoint

6. `server/pyproject.toml`
   - Thêm `httpx>=0.27.0` dependency (cho ServiceClient)

## Files Cần Refactor Tiếp (Future Work)

### Services Structure

Các service hiện tại cần được refactor vào cấu trúc mới:

1. `server/services/generation/`

   - Di chuyển từ `app/services/generation_service.py`
   - Di chuyển từ `app/api/endpoints/generation.py`
   - Di chuyển từ `app/api/endpoints/generation_sync.py`

2. `server/services/segmentation/`

   - Di chuyển từ `app/services/mask_segmentation_service.py`
   - Di chuyển từ `app/api/endpoints/smart_mask.py`

3. `server/services/image_utils/`

   - Di chuyển từ `app/api/endpoints/image_utils.py`

4. `server/services/job_manager/`
   - Di chuyển từ `app/api/endpoints/generation_async.py`
   - Di chuyển từ `app/services/job_cleanup.py`

## Summary

- **Files mới**: 13 files
- **Files cần xóa**: 4 files/functions
- **Files cần thay đổi**: 6 files
- **Files cần refactor tiếp**: 4 service modules
