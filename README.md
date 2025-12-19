# 🎨 Artmancer Web

## AI-Powered Art Generation Platform

Artmancer is a modern full-stack web application that enables users to generate stunning AI artwork through intuitive text prompts and image references. Built with Next.js 16 and FastAPI, featuring a sleek dark theme with purple accents and powered by Qwen AI models.

![Next.js](https://img.shields.io/badge/Next.js-16.0.7-black)
![React](https://img.shields.io/badge/React-19.2.0-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-4.x-38bdf8)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.4-009688)
![Python](https://img.shields.io/badge/Python-3.12+-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c)

## ✨ Features

### Frontend
- 🎨 **AI Art Generation** - Create stunning artwork from text descriptions
- 📷 **Image Upload** - Use reference images for enhanced generation
- 🎛️ **Customization Panel** - Control style, quality, and generation parameters
- 📱 **Responsive Design** - Optimized for desktop and mobile devices
- 🌙 **Dark Theme** - Modern dark interface with purple accent colors
- 🔄 **Dynamic Sizing** - Multiple output resolutions (512x512, 768x768, 1024x1024)
- ⚡ **Fast Performance** - Built with Next.js 16 and Turbopack
- 🖼️ **Image Comparison** - Side-by-side comparison slider for before/after results
- 📊 **Real-time Progress** - Server-sent events for generation progress tracking

### Backend
- 🤖 **Qwen AI Models** - Advanced image editing with Qwen-Image-Edit-2509
- 🔄 **Async Generation** - Non-blocking job processing with status tracking
- 🎯 **Smart Mask Segmentation** - Intelligent object detection and masking (FastSAM & BiRefNet)
- 🖼️ **Image Processing** - Advanced image utilities and transformations
- ☁️ **Modal Cloud Deployment** - Scalable microservices architecture
- 💾 **Persistent Storage** - Modal Volumes for model checkpoints and cache
- 🚀 **Cold Boot Optimization** - Fast startup with direct volume loading (1-3s vs 20-60s)
- 🔧 **Multiple Operations** - Image insertion, removal, and white balance adjustment

## 🏗️ Architecture

### System Overview

Artmancer follows a microservices architecture deployed on Modal:

```
┌─────────────────┐
│   Next.js App   │ (Frontend - Port 3000)
│   (Client)      │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  API Gateway    │ (CPU - Entry Point)
│  (Modal)        │
└────────┬────────┘
         │
    ┌────┴────┬──────────────┬─────────────┐
    ▼         ▼              ▼             ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Job     │ │ Smart    │ │ Image    │ │ A100     │
│ Manager │ │ Mask     │ │ Utils    │ │ Worker   │
│ (CPU)   │ │ (T4 GPU)│ │ (CPU)    │ │ (A100)   │
└─────────┘ └──────────┘ └──────────┘ └──────────┘
```

### Services

1. **API Gateway** (CPU)
   - Single entry point for all requests
   - Routes requests to appropriate services
   - Handles CORS and authentication

2. **Job Manager** (CPU)
   - Coordinates async generation jobs
   - Dispatches tasks to A100 workers
   - Manages job state with Modal Dictio
   - Cold boot enabled (scales to zero)

3. **Segmentation Service** (T4 GPU)
   - Smart mask generation using FastSAM
   - BiRefNet for advanced object detection
   - Handles image segmentation requests

4. **Image Utils Service** (CPU)
   - Image processing utilities
   - Format conversion and transformations
   - Base64 encoding/decoding

5. **A100 Worker** (A100 GPU)
   - Async image generation
   - Qwen model inference
   - Spawned by Job Manager (not public endpoint)

## 🚀 Quick Start

### Prerequisites

**Frontend:**
- Node.js 18.0 or higher
- npm 8.0 or higher (or yarn 1.22+)

**Backend:**
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Modal account and CLI configured
- CUDA-capable GPU (for local development) or Modal account (for cloud deployment)

### Installation

#### 1. Clone the repository

   ```bash
   git clone https://github.com/nxank4/artmancer-web.git
   cd artmancer-web
   ```

#### 2. Frontend Setup

   ```bash
   cd client
   npm install
   ```

#### 3. Backend Setup

   ```bash
cd server
uv sync
```

#### 4. Environment Configuration

**Frontend** - Create `client/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8080
```

**Backend** - Create `server/.env`:

```env
# Model checkpoint paths (for local development)
MODEL_FILE_INSERTION=./checkpoints/insertion_cp.safetensors
MODEL_FILE_REMOVAL=./checkpoints/removal_cp.safetensors
MODEL_FILE_WHITE_BALANCE=./checkpoints/wb_cp.safetensors

# Input quality settings
INPUT_QUALITY=resized  # or "original"
INPUT_QUALITY_WARNING_PX=2048

# Scheduler settings
ENABLE_FLOWMATCH_SCHEDULER=false
SCHEDULER_SHIFT=3.0

# Modal deployment (if using Modal)
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

#### 5. Start Development Servers

**Frontend:**
```bash
cd client
   npm run dev
   ```

**Backend (Local):**
```bash
cd server
uv run uvicorn main:app --reload --port 8080
```

**Backend (Modal - Production):**
```bash
cd server
modal deploy modal_app.py
```

#### 6. Open your browser

   Navigate to [http://localhost:3000](http://localhost:3000)

## 🛠️ Development

### Available Scripts

#### Frontend (`client/`)

- **`npm run dev`** - Starts the development server with Turbopack
- **`npm run build`** - Builds the app for production
- **`npm run start`** - Runs the built app in production mode
- **`npm run lint`** - Runs ESLint
- **`npm run lint:fix`** - Fixes ESLint errors automatically

#### Backend (`server/`)

- **`uv run uvicorn main:app --reload`** - Start local development server
- **`uv run python -m pytest`** - Run tests
- **`modal deploy modal_app.py`** - Deploy to Modal cloud
- **`modal run modal_app.py::setup_volume`** - Setup Modal volume with models

### Project Structure

```
artmancer-web/
├── client/                      # Next.js frontend application
│   ├── public/                 # Static assets
│   │   ├── favicon.ico
│   │   └── logo.svg
│   ├── src/
│   │   ├── app/                # Next.js App Router
│   │   │   ├── globals.css     # Global styles & theme
│   │   │   ├── layout.tsx      # Root layout
│   │   │   └── page.tsx        # Main application page
│   │   ├── components/         # React components
│   │   ├── contexts/           # React contexts
│   │   ├── hooks/              # Custom React hooks
│   │   ├── services/           # API service clients
│   │   ├── theme/              # Theme configuration
│   │   └── utils/              # Utility functions
│   ├── package.json
│   └── next.config.ts
│
├── server/                      # FastAPI backend application
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints/      # API route handlers
│   │   │       ├── generation_async.py
│   │   │       ├── smart_mask.py
│   │   │       ├── image_utils.py
│   │   │       ├── system.py
│   │   │       └── debug.py
│   │   ├── core/               # Core configuration
│   │   │   ├── config.py       # Settings and environment
│   │   │   └── pipeline.py     # Model loading and pipeline
│   │   ├── models/             # Pydantic models
│   │   │   └── schemas.py      # Request/response schemas
│   │   └── services/           # Business logic
│   │       ├── generation_service.py
│   │       ├── mask_segmentation_service.py
│   │       ├── image_processing.py
│   │       └── debug_service.py
│   ├── api_gateway/            # API Gateway service
│   │   ├── main.py
│   │   └── router.py
│   ├── shared/                 # Shared utilities
│   │   └── clients/
│   │       └── service_client.py
│   ├── modal_app.py            # Modal deployment configuration
│   ├── main.py                 # Local development entry point
│   ├── pyproject.toml          # Python dependencies
│   └── uv.lock                 # Dependency lock file
│
└── README.md
```

### Technology Stack

#### Frontend

- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **TypeScript 5** - Type safety
- **Tailwind CSS 4** - Utility-first CSS framework
- **Turbopack** - Fast bundler for development
- **Material-UI (MUI)** - Component library
- **Lucide React** - Icon library
- **@img-comparison-slider/react** - Image comparison component

#### Backend

- **FastAPI** - Modern Python web framework
- **Python 3.12+** - Programming language
- **PyTorch 2.9** - Deep learning framework
- **Diffusers** - Hugging Face diffusion models library
- **Transformers** - Hugging Face transformers library
- **Qwen** - Alibaba Cloud AI models
- **Modal** - Cloud infrastructure platform
- **Pydantic** - Data validation
- **Pillow** - Image processing
- **Ultralytics** - YOLO models for detection
- **FastSAM** - Fast Segment Anything Model
- **BiRefNet** - Bidirectional Reference Network

#### Infrastructure

- **Modal** - Serverless cloud platform
- **Modal Volumes** - Persistent storage for models
- **Modal Dictio** - Distributed key-value store for job state
- **AWS S3** - Object storage (via boto3)

## 🎨 Design Features

### Color Palette

- **Primary Background**: `#0b0b0d` (Deep dark)
- **Secondary Background**: `#1a1a1f` (Lighter dark)
- **Primary Accent**: `#6a0dad` (Purple)
- **Highlight Accent**: `#9d4edd` (Light purple)
- **Text Primary**: `#e0e0e0` (Light gray)
- **Text Secondary**: `#a0a0a8` (Medium gray)

### UI Components

- **Header**: Logo, search input, and action buttons
- **Art Display**: Dynamic canvas with size options
- **Customize Panel**: Collapsible settings sidebar
- **Image Upload**: Drag-and-drop or click-to-upload
- **Progress Indicator**: Real-time generation progress
- **Image Comparison**: Side-by-side slider for results

## 📡 API Documentation

### Base URL

- **Local**: `http://localhost:8080/api`
- **Modal**: `https://<username>--api-gateway.modal.run/api`

### Endpoints

#### Health Check

```http
GET /api/health
```

Returns service health status.

#### Async Generation

```http
POST /api/generate/async
Content-Type: application/json

{
  "prompt": "a beautiful sunset over mountains",
  "image": "base64_encoded_image",
  "mask": "base64_encoded_mask",
  "operation": "insertion",  // "insertion" | "removal" | "white_balance"
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 7.5
}
```

Returns:
```json
{
  "task_id": "uuid",
  "status": "pending"
}
```

#### Generation Status

```http
GET /api/generate/status/{task_id}
```

Returns:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "result": {
    "image": "base64_encoded_image"
  }
}
```

#### Stream Progress

```http
GET /api/generate/stream/{task_id}
```

Server-sent events stream for real-time progress updates.

#### Smart Mask

```http
POST /api/smart-mask
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "prompt": "person",
  "model_type": "segmentation"  // "segmentation" | "birefnet"
}
```

#### Cancel Generation

```http
POST /api/generate/async/cancel/{task_id}
```

Cancels an in-progress generation task.

### Generation Operations

1. **Insertion** - Add objects to images based on text prompts
2. **Removal** - Remove objects from images using masks
3. **White Balance** - Adjust white balance of images

## 🔧 Configuration

### Environment Variables

#### Frontend (`client/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8080
```

#### Backend (`server/.env`)

```env
# Model Configuration
MODEL_FILE_INSERTION=./checkpoints/insertion_cp.safetensors
MODEL_FILE_REMOVAL=./checkpoints/removal_cp.safetensors
MODEL_FILE_WHITE_BALANCE=./checkpoints/wb_cp.safetensors

# Input Quality
INPUT_QUALITY=resized  # "resized" | "original"
INPUT_QUALITY_WARNING_PX=2048

# Scheduler
ENABLE_FLOWMATCH_SCHEDULER=false
SCHEDULER_SHIFT=3.0

# Modal (for cloud deployment)
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
```

### Modal Deployment

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate**:
   ```bash
   modal token set
   ```

3. **Setup Volume** (first time):
   ```bash
   cd server
   modal run modal_app.py::setup_volume
   ```

4. **Deploy Services**:
   ```bash
   modal deploy modal_app.py
   ```

5. **Get Service URLs**:
   ```bash
   modal app list
   ```

## 📱 Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🧪 Testing

### Frontend

```bash
cd client
npm run lint
```

### Backend

```bash
cd server
uv run pytest
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port already in use

   ```bash
# Frontend
   npx kill-port 3000
   npm run dev

# Backend
npx kill-port 8080
uv run uvicorn main:app --reload --port 8080
   ```

#### 2. Node modules issues

   ```bash
cd client
   rm -rf node_modules package-lock.json
   npm install
   ```

#### 3. Python dependencies issues

   ```bash
cd server
rm -rf .venv uv.lock
uv sync
```

#### 4. Modal deployment errors

- Ensure Modal CLI is authenticated: `modal token set`
- Check Modal volume exists: `modal volume list`
- Verify model files are uploaded to volume

#### 5. Model loading errors

- Verify checkpoint files exist in `server/checkpoints/`
- Check file paths in `.env` configuration
- Ensure sufficient disk space for model files

#### 6. CORS errors

- Verify `ALLOWED_ORIGINS` includes your frontend URL
- Check API Gateway CORS configuration
- Ensure frontend `NEXT_PUBLIC_API_URL` matches backend URL

## 🚧 Roadmap

### Phase 1 (Completed)

- ✅ Frontend UI/UX implementation
- ✅ Image upload functionality
- ✅ Responsive design
- ✅ Theme customization
- ✅ Backend API development
- ✅ AI model integration (Qwen)
- ✅ Async job processing
- ✅ Smart mask segmentation

### Phase 2 (In Progress)

- 🔄 User authentication
- 🔄 Generation history
- 🔄 Advanced editing tools
- 🔄 Batch processing

### Phase 3 (Future)

- 📋 Social features
- 📋 Premium subscriptions
- 📋 Mobile app
- 📋 Real-time collaboration
- 📋 Model fine-tuning interface

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Use TypeScript for all frontend code
- Use Python type hints for all backend code
- Follow existing code style and conventions
- Add appropriate comments for complex logic
- Test your changes across different screen sizes
- Ensure accessibility standards are met
- Update documentation for new features
- Write tests for new functionality

### Code Style

- **Frontend**: ESLint with React and TypeScript rules
- **Backend**: Black formatter, mypy type checking
- **Commits**: Conventional commits format

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Developer**: [nxank4](https://github.com/nxank4)
- **Project**: Artmancer Web Application

## 🙏 Acknowledgments

- [Next.js](https://nextjs.org/) for the amazing React framework
- [FastAPI](https://fastapi.tiangolo.com/) for the modern Python web framework
- [Modal](https://modal.com/) for cloud infrastructure
- [Hugging Face](https://huggingface.co/) for AI models and libraries
- [Qwen](https://qwenlm.github.io/) for the powerful image editing models
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS
- The open-source community for inspiration and tools

---

### Built with ❤️ and ☕ by the Artmancer team

For questions or support, please open an issue on GitHub.
