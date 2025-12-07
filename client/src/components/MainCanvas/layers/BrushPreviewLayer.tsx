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
  maskToolType: "brush" | "box";
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
      maskToolType !== "brush" ||
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

      // Calculate relative position (0-1)
      const relativeX = (e.clientX - canvasRect.left) / canvasRect.width;
      const relativeY = (e.clientY - canvasRect.top) / canvasRect.height;

      // Convert to canvas internal coordinates
      const canvasX = relativeX * imageDimensions.width;
      const canvasY = relativeY * imageDimensions.height;

      // Check if mouse is within image bounds
      if (
        canvasX >= 0 &&
        canvasX <= imageDimensions.width &&
        canvasY >= 0 &&
        canvasY <= imageDimensions.height
      ) {
        mousePosRef.current = { x: canvasX, y: canvasY };
      } else {
        mousePosRef.current = null;
      }

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

      // Calculate brush size in canvas coordinates
      // Dynamic brush size: scale from 0.5% to 10% of base image size
      const baseImageSize = Math.min(
        imageDimensions.width,
        imageDimensions.height
      );
      const brushSize = (maskBrushSize / 100) * (baseImageSize / 5);

      const { x, y } = mousePosRef.current;

      // Draw preview circle with contrasting colors (white + black border for visibility)
      ctx.globalCompositeOperation = "source-over";

      // Outer black border (shadow effect for visibility)
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Inner white dashed border (main preview)
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]); // Dashed line
      ctx.stroke();
      ctx.setLineDash([]); // Reset

      // Center dot for precision
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 0, 0, 0.8)";
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const container = imageContainerRef.current;
    if (!container) return;

    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("mouseleave", handleMouseLeave);

    // Initial draw
    drawPreview(ctx);

    return () => {
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("mouseleave", handleMouseLeave);
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

  if (!isMaskingMode || maskToolType !== "brush" || !imageDimensions) {
    return null;
  }

  const style: React.CSSProperties = {
    ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.MASK + 1), // Above mask layer
    pointerEvents: "none", // Don't block mouse events
  };

  return <canvas ref={canvasRef} style={style} />;
}
