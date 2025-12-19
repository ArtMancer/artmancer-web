/**
 * BrushPreviewLayer Component
 * Hiển thị preview brush size khi đang trong masking mode
 */

import { useEffect, useRef } from "react";
import { getAbsoluteLayerStyle } from "../utils";
import { Z_INDEX } from "../constants";

interface BrushPreviewLayerProps {
  isMaskingMode: boolean;
  maskBrushSize: number;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  transform: { scale: number };
  viewportZoom: number;
  imageContainerRef: React.RefObject<HTMLDivElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  maskToolType: "brush" | "box" | "eraser";
}

export default function BrushPreviewLayer({
  isMaskingMode,
  maskBrushSize,
  imageDimensions,
  displayScale,
  transform,
  viewportZoom,
  imageContainerRef,
  maskCanvasRef,
  maskToolType,
}: BrushPreviewLayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mousePosRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (
      !isMaskingMode ||
      (maskToolType !== "brush" && maskToolType !== "eraser") ||
      !imageDimensions ||
      !maskCanvasRef.current
    ) {
      mousePosRef.current = null;
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    // Set canvas size
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;

    const handleMouseMove = (e: MouseEvent) => {
      // Use maskCanvasRef to get accurate bounding rect (same as mask canvas)
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas || !canvas) return;

      // Get the mask canvas bounding rect (same calculation as getCanvasCoordinates)
      const canvasRect = maskCanvas.getBoundingClientRect();

      // Calculate relative position (có thể < 0 hoặc > 1 khi nằm ngoài canvas)
      const relativeX = (e.clientX - canvasRect.left) / canvasRect.width;
      const relativeY = (e.clientY - canvasRect.top) / canvasRect.height;

      // Nếu trỏ chuột đã ra khỏi vùng canvas chính thì ẩn preview hẳn
      if (relativeX < 0 || relativeX > 1 || relativeY < 0 || relativeY > 1) {
        mousePosRef.current = null;
        const previewCanvas = canvasRef.current;
        if (previewCanvas) {
          const previewCtx = previewCanvas.getContext("2d", {
            willReadFrequently: true,
          });
          if (previewCtx) {
            previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
          }
        }
        return;
      }

      // Convert to canvas internal coordinates
      const canvasX = relativeX * imageDimensions.width;
      const canvasY = relativeY * imageDimensions.height;

      mousePosRef.current = { x: canvasX, y: canvasY };

      // Redraw preview
      drawPreview(ctx);
    };

    const handleMouseLeave = () => {
      mousePosRef.current = null;
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    };

    const drawPreview = (ctx: CanvasRenderingContext2D) => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!mousePosRef.current) return;

      // Interpret maskBrushSize as brush RADIUS in pixels (matches UI "px")
      // Scale up for better visual feedback on high-res images
      const brushRadius = (maskBrushSize || 1) * 2;

      const { x, y } = mousePosRef.current;

      // Clamp coordinates to canvas bounds for preview display
      // This allows preview to show even when mouse is outside canvas,
      // but clamped to the edge of the canvas
      const clampedX = Math.max(0, Math.min(x, imageDimensions.width));
      const clampedY = Math.max(0, Math.min(y, imageDimensions.height));

      // Draw preview circle with strong contrasting visuals for better accessibility
      ctx.globalCompositeOperation = "source-over";

      // Semi-transparent fill to show affected area clearly
      ctx.beginPath();
      ctx.arc(clampedX, clampedY, brushRadius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 0, 0, 0.15)";
      ctx.fill();

      // Outer black border (shadow effect for visibility)
      ctx.beginPath();
      ctx.arc(clampedX, clampedY, brushRadius, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Inner white solid border (main preview)
      ctx.beginPath();
      ctx.arc(clampedX, clampedY, brushRadius, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Center dot for precision
      ctx.beginPath();
      ctx.arc(clampedX, clampedY, 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const container = imageContainerRef.current;
    if (!container) return;

    // Attach to both container and document to support strokes outside canvas
    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("mouseleave", handleMouseLeave);
    document.addEventListener("mousemove", handleMouseMove);

    // Initial draw
    drawPreview(ctx);

    return () => {
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("mouseleave", handleMouseLeave);
      document.removeEventListener("mousemove", handleMouseMove);
    };
  }, [
    isMaskingMode,
    maskBrushSize,
    imageDimensions,
    displayScale,
    transform.scale,
    viewportZoom,
    imageContainerRef,
    maskCanvasRef,
    maskToolType,
  ]);

  if (
    !isMaskingMode ||
    (maskToolType !== "brush" && maskToolType !== "eraser") ||
    !imageDimensions
  ) {
    return null;
  }

  const style: React.CSSProperties = {
    ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.MASK + 1), // Above mask layer
    pointerEvents: "none", // Don't block mouse events
  };

  return <canvas ref={canvasRef} style={style} />;
}
