# Architecture Diagram - New Backend Structure

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    (Next.js Client)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Requests
                            │ (Single Entry Point)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
│                  (Always-on, CPU-only)                      │
│                                                              │
│  - Routes requests to appropriate service                   │
│  - Aggregates health checks                                 │
│  - Handles CORS, security headers                           │
└───────────┬───────────┬───────────┬───────────┬─────────────┘
            │           │           │           │
            │           │           │           │
    ┌───────▼───┐ ┌─────▼────┐ ┌───▼────┐ ┌───▼──────────┐
    │ Generation│ │Segmentation│ │Image Utils│ │Job Manager  │
    │  Service  │ │  Service   │ │  Service │ │  Service   │
    │           │ │            │ │          │ │            │
    │  A100 GPU │ │   T4 GPU    │ │  CPU    │ │    CPU     │
    │           │ │            │ │          │ │            │
    │ Scale-to- │ │ Scale-to-   │ │ Scale-to-│ │ Always-on  │
    │   zero    │ │   zero      │ │   zero   │ │ (min=1)    │
    └───────────┘ └────────────┘ └──────────┘ └────────────┘
```

## Request Flow

### Synchronous Generation

```
Frontend → API Gateway → Generation Service (A100) → Response
```

### Asynchronous Generation

```
Frontend → API Gateway → Job Manager → A100 Worker → Job Manager → Response
```

### Smart Mask

```
Frontend → API Gateway → Segmentation Service (T4) → Response
```

### Image Utils

```
Frontend → API Gateway → Image Utils Service (CPU) → Response
```

## Service Details

### API Gateway

- **Type**: CPU-only, always-on
- **Responsibilities**:
  - Request routing
  - Health check aggregation
  - CORS handling
  - Security headers
- **Endpoints**: All `/api/*` paths

### Generation Service

- **Type**: A100 GPU, scale-to-zero
- **Responsibilities**:
  - Qwen image generation
  - Model inference
- **Endpoints**: `/api/generate`

### Segmentation Service

- **Type**: T4 GPU, scale-to-zero
- **Responsibilities**:
  - FastSAM mask generation
  - Smart mask segmentation
- **Endpoints**: `/api/smart-mask`

### Image Utils Service

- **Type**: CPU-only, scale-to-zero
- **Responsibilities**:
  - Image processing (resize, crop, encode)
  - Object extraction
  - MAE generation
  - Canny edge detection
- **Endpoints**: `/api/image-utils/*`

### Job Manager Service

- **Type**: CPU-only, always-on (min_containers=1)
- **Responsibilities**:
  - Async job management
  - Job state tracking
  - Queue management
- **Endpoints**: `/api/generate/async`, `/api/generate/status/*`, `/api/generate/result/*`

## Data Flow

### Job State Management

```
Job Manager (Modal Dict) ← A100 Worker updates state
         ↓
Frontend polls status
```

### Model Loading

```
Modal Volume → Service Container (direct mount, no copy)
```

## Key Differences from Old Architecture

### Old Architecture

- Light/Heavy containers with wake-up logic
- Frontend pings services to warm up
- Complex state management for on/off

### New Architecture

- Independent microservices
- API Gateway as single entry point
- No wake-up logic (services auto-scale)
- Clean separation of concerns
