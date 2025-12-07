# ArtMancer System Architecture Diagram

## PlantUML Source

See `ARCHITECTURE_DIAGRAM.puml` for the complete PlantUML source code.

## Draw.io Instructions

### Components to Create:

1. **Frontend (React/Next.js)**

   - Rectangle box
   - Label: "Frontend Client"
   - Color: Light blue (#E6F3FF)

2. **API Gateway (CPU)**

   - Rectangle box
   - Label: "API Gateway"
   - Subtitle: "CPU, Cold-start: <1s"
   - Color: Light green (#E6FFE6)

3. **Modal Volumes (Storage)**

   - Cylinder/database shape
   - Three volumes:
     - "Qwen Model Cache"
     - "FastSAM Model Cache"
     - "BiRefNet Model Cache"
   - Color: Light yellow (#FFF9E6)

4. **Segmentation Service (T4 GPU)**

   - Rectangle box
   - Label: "Segmentation Service"
   - Subtitle: "T4 GPU, 16GB VRAM"
   - Details: "FastSAM: 1-3s, BiRefNet: 3-5s"
   - Color: Light orange (#FFE6E6)

5. **Image Utils Service (CPU)**

   - Rectangle box
   - Label: "Image Utils Service"
   - Subtitle: "CPU"
   - Color: Light green (#E6FFE6)

6. **Job Manager Service**

   - Rectangle box
   - Label: "Job Manager"
   - Subtitle: "Task coordination"
   - Color: Light purple (#F0E6FF)

7. **A100 GPU Workers**
   - Multiple rectangle boxes
   - Label: "A100 Worker 1/2/N"
   - Subtitle: "A100 GPU, 40/80GB VRAM"
   - Details: "FP16 inference, 15-30s typical"
   - Color: Light red (#FFE6E6)

### Connections (Arrows):

1. **Frontend → API Gateway**

   - Label: "HTTP/HTTPS (Base64 images, JSON requests)"
   - Arrow: Solid line

2. **API Gateway → Services**

   - To Segmentation Service: "POST /api/smart-mask"
   - To Image Utils Service: "POST /api/image-utils/extract-object"
   - To Job Manager: "POST /api/generate/async"
   - Arrow: Solid lines

3. **Job Manager → A100 Workers**

   - Label: "Task assignment"
   - Arrow: Solid lines

4. **Services → Modal Volumes**

   - Segmentation Service → FastSAM/BiRefNet volumes
   - A100 Workers → Qwen volume
   - Arrow: Dashed lines (dependencies)

5. **Services → API Gateway → Frontend**
   - Response flow (JSON, SSE)
   - Arrow: Solid lines

### Data Flow Sequence (Optional):

You can add numbered labels to show the workflow:

1. Mask Generation (Frontend → Gateway → Segmentation)
2. Object Extraction (Gateway → Image Utils)
3. Image Generation (Gateway → Job Manager → A100 Workers)
4. Progress Updates (A100 → Job Manager → Gateway → Frontend via SSE)
5. Final Result (Job Manager → Gateway → Frontend)

### Legend Box:

Create a legend showing:

- **Hardware Allocation**: CPU, T4 GPU, A100 GPU
- **Performance**: Cold-start times, typical workflow times
- **Optimizations**: Modal Volumes, FP16 inference, Async processing

### Layout Suggestions:

- Top: Frontend
- Middle: API Gateway (centered)
- Left side: Segmentation Service, Image Utils Service
- Right side: Job Manager, A100 Workers (stacked vertically)
- Bottom: Modal Volumes (horizontal row)

### Color Scheme:

- Frontend: Light blue (#E6F3FF)
- API Gateway: Light green (#E6FFE6)
- Segmentation Service: Light orange (#FFE6E6)
- Image Utils: Light green (#E6FFE6)
- Job Manager: Light purple (#F0E6FF)
- A100 Workers: Light red (#FFE6E6)
- Modal Volumes: Light yellow (#FFF9E6)
