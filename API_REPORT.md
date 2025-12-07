# Báo Cáo API Backend - ArtMancer

## Tổng Quan

Backend ArtMancer cung cấp các API chính cho việc tạo ảnh với AI:

1. **Generation APIs** - Tạo ảnh với AI (async)
2. **Smart Mask APIs** - Tạo mask thông minh với FastSAM
3. **Image Utils APIs** - Xử lý ảnh cơ bản

---

## 1. Generation APIs

### 1.1 Generate Image (Async)

- **Endpoint**: `POST /api/generate`
- **Mô tả**: Tạo ảnh với AI (tự động chuyển sang async mode nếu trong Modal environment)
- **Request Body** (`GenerationRequest`):
  ```json
  {
    "prompt": "remove the object",
    "input_image": "base64_encoded_image",
    "conditional_images": ["base64_mask"],
    "reference_image": "base64_ref_image",
    "reference_mask_R": "base64_mask",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 10,
    "guidance_scale": 1.0,
    "true_cfg_scale": 4.0,
    "negative_prompt": "blurry, low quality",
    "seed": 42,
    "task_type": "insertion",
    "angle": "wide-angle",
    "background_preset": "marble-surface",
    "input_quality": "original",
    "enable_flowmatch_scheduler": false
  }
  ```
- **Response**:
  ```json
  {
    "task_id": "uuid",
    "status": "queued",
    "message": "Generation job submitted successfully..."
  }
  ```

### 1.2 Submit Async Generation Job

- **Endpoint**: `POST /api/generate/async`
- **Mô tả**: Submit job tạo ảnh async (chạy trên A100 worker)
- **Request**: Tương tự `POST /api/generate`
- **Response**:
  ```json
  {
    "task_id": "uuid",
    "status": "queued",
    "message": "Generation job submitted successfully"
  }
  ```

### 1.3 Get Generation Status

- **Endpoint**: `GET /api/generate/status/{task_id}`
- **Mô tả**: Lấy trạng thái của generation job
- **Response**:
  ```json
  {
    "task_id": "uuid",
    "status": "processing",
    "progress": 0.65,
    "current_step": 6,
    "total_steps": 10,
    "error": null
  }
  ```
- **Status values**: `queued`, `processing`, `done`, `error`, `cancelled`

### 1.4 Get Generation Result

- **Endpoint**: `GET /api/generate/result/{task_id}`
- **Mô tả**: Lấy kết quả generation (chỉ khi status = "done")
- **Response**:
  ```json
  {
    "task_id": "uuid",
    "status": "done",
    "image": "base64_encoded_image",
    "debug_info": {
      "conditional_images": [...],
      "conditional_labels": [...],
      "input_image_size": "1024x1024",
      "output_image_size": "1024x1024",
      "lora_adapter": "adapter_name",
      "loaded_adapters": [...]
    }
  }
  ```

### 1.5 Stream Generation Progress (SSE)

- **Endpoint**: `GET /api/generate/stream/{task_id}`
- **Mô tả**: Stream real-time progress updates qua Server-Sent Events
- **Response**: SSE stream với format:
  ```
  data: {"task_id": "uuid", "status": "processing", "progress": 0.5, "current_step": 5, "total_steps": 10, "loading_message": "Generating..."}
  ```
- **Headers**:
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`

### 1.6 Cancel Async Generation

- **Endpoint**: `POST /api/generate/async/cancel/{task_id}`
- **Mô tả**: Hủy generation job đang chạy
- **Response**:
  ```json
  {
    "success": true,
    "message": "Async generation task {task_id} marked for cancellation",
    "task_id": "uuid"
  }
  ```

---

## 2. Smart Mask APIs

### 2.1 Generate Smart Mask

- **Endpoint**: `POST /api/smart-mask`
- **Mô tả**: Tạo mask thông minh sử dụng FastSAM
- **Request Body** (`SmartMaskRequest`):
  ```json
  {
    "image": "base64_encoded_image",
    "image_id": "cached_image_id",
    "bbox": [x_min, y_min, x_max, y_max],
    "points": [[x1, y1], [x2, y2]],
    "border_adjustment": 5,
    "use_blur": false,
    "auto_detect": true,
    "mask": "base64_guidance_mask"
  }
  ```
- **Response** (`SmartMaskResponse`):
  ```json
  {
    "success": true,
    "mask_base64": "base64_encoded_mask",
    "image_id": "cached_image_id",
    "request_id": "uuid",
    "error": null
  }
  ```
- **Lưu ý**:
  - Có thể dùng `image` (lần đầu) hoặc `image_id` (các lần sau)
  - Cần có `bbox`, `points`, hoặc `auto_detect=true`
  - `auto_detect`: tự động phát hiện object chính

### 2.2 Cancel Smart Mask

- **Endpoint**: `POST /api/smart-mask/cancel/{request_id}`
- **Mô tả**: Hủy request tạo smart mask
- **Response**:
  ```json
  {
    "success": true,
    "message": "Smart mask request {request_id} marked for cancellation",
    "request_id": "uuid"
  }
  ```

---

## 3. Image Utils APIs

### 3.1 Extract Object

- **Endpoint**: `POST /api/image-utils/extract-object`
- **Mô tả**: Trích xuất object từ ảnh sử dụng mask (tạo PNG với background trong suốt)
- **Request** (`ExtractObjectRequest`):
  ```json
  {
    "image": "base64_encoded_image",
    "mask": "base64_encoded_mask"
  }
  ```
- **Response** (`ExtractObjectResponse`):
  ```json
  {
    "success": true,
    "extracted_image": "base64_encoded_png",
    "error": null
  }
  ```
- **Lưu ý**:
  - Mask: white (255) = giữ lại, black (0) = xóa (transparent)
  - Kết quả là PNG với alpha channel

---

## Models & Schemas

### GenerationRequest

- `prompt`: Mô tả chỉnh sửa mong muốn
- `input_image`: Ảnh gốc (base64)
- `conditional_images`: Danh sách ảnh điều kiện (mask, background, etc.)
- `reference_image`: Ảnh reference (cho insertion task)
- `reference_mask_R`: Mask R cho two-source workflow
- `width`, `height`: Kích thước output (256-2048)
- `num_inference_steps`: Số bước inference (1-100, default: 10)
- `guidance_scale`: Guidance scale (0.5-20, default: 1.0)
- `true_cfg_scale`: True CFG scale (0.5-15, default: 4.0)
- `negative_prompt`: Negative prompt
- `seed`: Random seed
- `task_type`: "insertion", "removal", "white-balance"
- `angle`: Angle macro label
- `background_preset`: Background preset name
- `input_quality`: "resized" hoặc "original"
- `enable_flowmatch_scheduler`: Sử dụng FlowMatch scheduler

### GenerationResponse

- `success`: Thành công hay không
- `image`: Ảnh kết quả (base64)
- `generation_time`: Thời gian generation (seconds)
- `model_used`: Model được sử dụng
- `parameters_used`: Các parameters đã dùng
- `request_id`: ID để truy cập visualization images
- `debug_info`: Thông tin debug (conditional images, labels, etc.)

---

## Error Handling

Tất cả APIs trả về HTTP status codes chuẩn:

- `200`: Success
- `400`: Bad Request (validation error)
- `404`: Not Found (task/session không tồn tại)
- `500`: Internal Server Error
- `501`: Not Implemented (feature không available)
- `503`: Service Unavailable (service không kết nối được)

Error response format:

```json
{
  "detail": "Error message hoặc error object"
}
```

---

## Notes

1. **Async Generation**: Tất cả generation jobs chạy async trên A100 workers (Modal environment)
2. **Image Caching**: Smart mask API cache images để tối ưu performance
3. **SSE Streaming**: Generation progress được stream real-time qua SSE
4. **Cancellation**: Hỗ trợ cancel cho cả generation và smart mask requests
5. **Task Types**: Hỗ trợ 3 loại task: `insertion`, `removal`, `white-balance`
