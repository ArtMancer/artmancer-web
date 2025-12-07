"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  X,
  Paintbrush,
  Wand2,
  Check,
  Trash2,
  Plus,
  Square,
} from "lucide-react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Button,
  ButtonGroup,
  Box,
  Slider,
  Typography,
  Tooltip,
  Divider,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  CircularProgress,
} from "@mui/material";

interface ReferenceImageEditorProps {
  isOpen: boolean;
  imageData: string; // Base64 image data
  onClose: () => void;
  onSubmit: (processedImage: string, maskData?: string | null) => void; // Return processed image + mask for main canvas
  initialMaskData?: string | null; // Optional: existing mask data to load
  modelType?: "segmentation" | "birefnet"; // Model type for mask detection
  onModelTypeChange?: (modelType: "segmentation" | "birefnet") => void; // Callback for model type change
  borderAdjustment?: number; // Border adjustment for mask detection
  onBorderAdjustmentChange?: (value: number) => void; // Callback for border adjustment change
}

export default function ReferenceImageEditor({
  isOpen,
  imageData,
  onClose,
  onSubmit,
  initialMaskData,
  modelType = "segmentation",
  onModelTypeChange,
  borderAdjustment: propBorderAdjustment = 0,
  onBorderAdjustmentChange,
}: ReferenceImageEditorProps) {
  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null); // Display canvas (shows combined mask)
  const pendingBrushCanvasRef = useRef<HTMLCanvasElement>(null); // Hidden canvas for new brush strokes
  const containerRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null); // Context for pending brush
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const confirmedMaskRef = useRef<ImageData | null>(null); // Stores confirmed SAM mask
  const mousePositionRef = useRef<{ x: number; y: number } | null>(null); // Ref to track mouse position
  const cursorRef = useRef<HTMLDivElement | null>(null); // Ref for cursor element to update directly

  // Box drawing state
  const [toolType, setToolType] = useState<"brush" | "box">("brush");
  const boxStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const boxCurrentPosRef = useRef<{ x: number; y: number } | null>(null);
  const initialBoxStateRef = useRef<ImageData | null>(null);

  // State
  const [imageDimensions, setImageDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [loadedImage, setLoadedImage] = useState<HTMLImageElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(30); // UI value 0-100
  const [hasMask, setHasMask] = useState(false);
  const [hasPendingBrush, setHasPendingBrush] = useState(false); // Track if user drew new strokes (brush or box)
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [extractedObject, setExtractedObject] = useState<string | null>(null);
  // Use prop borderAdjustment if provided, otherwise use local state
  const [localBorderAdjustment, setLocalBorderAdjustment] = useState(0);
  const borderAdjustment =
    propBorderAdjustment !== undefined
      ? propBorderAdjustment
      : localBorderAdjustment;
  const setBorderAdjustment =
    onBorderAdjustmentChange || setLocalBorderAdjustment;

  // Helper: get current combined mask (confirmed + pending) as binary PNG (data URL)
  const getCurrentMaskDataUrl = useCallback((): string | null => {
    if (!hasMask || !maskCanvasRef.current) return null;
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) return null;

    const maskImageData = maskCtx.getImageData(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    );
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = maskCanvas.width;
    tempCanvas.height = maskCanvas.height;
    const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
    if (!tempCtx) return null;

    const tempImageData = tempCtx.getImageData(
      0,
      0,
      tempCanvas.width,
      tempCanvas.height
    );
    // Convert red overlay to binary white mask
    for (let i = 0; i < maskImageData.data.length; i += 4) {
      if (maskImageData.data[i] > 100 && maskImageData.data[i + 3] > 50) {
        tempImageData.data[i] = 255;
        tempImageData.data[i + 1] = 255;
        tempImageData.data[i + 2] = 255;
        tempImageData.data[i + 3] = 255;
      } else {
        tempImageData.data[i] = 0;
        tempImageData.data[i + 1] = 0;
        tempImageData.data[i + 2] = 0;
        tempImageData.data[i + 3] = 255;
      }
    }
    tempCtx.putImageData(tempImageData, 0, 0);
    return tempCanvas.toDataURL("image/png");
  }, [hasMask]);

  // Calculate actual brush size based on image dimensions (like useMasking)
  const getActualBrushSize = useCallback(() => {
    if (!imageDimensions) return brushSize;
    const baseImageSize = Math.min(
      imageDimensions.width,
      imageDimensions.height
    );
    // Scale from 0.5% to 10% of base image size
    return (brushSize / 100) * (baseImageSize / 5);
  }, [brushSize, imageDimensions]);

  // Load image first (only dimensions)
  useEffect(() => {
    if (!isOpen || !imageData) return;

    setImageDimensions(null);
    setLoadedImage(null);

    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
      setLoadedImage(img);
      setHasMask(false);
      setHasPendingBrush(false);
      setExtractedObject(null);
      confirmedMaskRef.current = null;
    };
    img.onerror = () => {
      console.error("Failed to load image");
    };
    img.src = imageData;
  }, [isOpen, imageData]);

  // Draw image to canvas after canvas is mounted
  useEffect(() => {
    if (!loadedImage || !imageDimensions) return;

    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    if (!canvas || !maskCanvas || !pendingCanvas) return;

    // Setup all canvas sizes
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;
    maskCanvas.width = imageDimensions.width;
    maskCanvas.height = imageDimensions.height;
    pendingCanvas.width = imageDimensions.width;
    pendingCanvas.height = imageDimensions.height;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (ctx) {
      ctx.drawImage(loadedImage, 0, 0);
    }

    // Clear mask canvas
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (maskCtx) {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }

    // Setup pending brush canvas context (this is where user draws)
    const pendingCtx = pendingCanvas.getContext("2d", {
      willReadFrequently: true,
    });
    if (pendingCtx) {
      pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
      ctxRef.current = pendingCtx;
    }

    // Reset confirmed mask
    confirmedMaskRef.current = null;
  }, [loadedImage, imageDimensions]);

  // Setup brush properties when brush size changes
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !imageDimensions) return;

    // Calculate brush size directly to avoid dependency on getActualBrushSize callback
    const baseImageSize = Math.min(
      imageDimensions.width,
      imageDimensions.height
    );
    const actualBrushSize = (brushSize / 100) * (baseImageSize / 5);

    ctx.lineWidth = actualBrushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
    ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  }, [brushSize, imageDimensions]);

  // Ensure focusable elements outside dialog don't retain focus when dialog opens
  useEffect(() => {
    if (!isOpen) return;

    // Use MutationObserver to watch for aria-hidden changes and blur focused elements
    const observer = new MutationObserver(() => {
      const activeElement = document.activeElement as HTMLElement;
      if (!activeElement) return;

      // Check if the active element is inside an aria-hidden container
      let parent = activeElement.parentElement;
      while (parent && parent !== document.body) {
        if (parent.getAttribute("aria-hidden") === "true") {
          // If focused element is inside aria-hidden container, blur it
          // The dialog will handle focus management
          activeElement.blur();
          break;
        }
        parent = parent.parentElement;
      }
    });

    // Observe the document body for aria-hidden attribute changes
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["aria-hidden"],
      subtree: true,
    });

    // Also check immediately after a short delay to catch initial state
    const timeoutId = setTimeout(() => {
      const activeElement = document.activeElement as HTMLElement;
      if (!activeElement) return;

      let parent = activeElement.parentElement;
      while (parent && parent !== document.body) {
        if (parent.getAttribute("aria-hidden") === "true") {
          activeElement.blur();
          break;
        }
        parent = parent.parentElement;
      }
    }, 100);

    return () => {
      observer.disconnect();
      clearTimeout(timeoutId);
    };
  }, [isOpen]);

  // Get canvas coordinates from mouse event
  const getCanvasCoordinates = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const canvas = maskCanvasRef.current;
      if (!canvas || !imageDimensions) return null;

      const rect = canvas.getBoundingClientRect();

      // Handle both mouse and touch events
      let clientX: number, clientY: number;
      if ("touches" in e) {
        if (e.touches.length === 0) return null;
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else {
        clientX = e.clientX;
        clientY = e.clientY;
      }

      // Calculate relative position and scale to canvas internal coordinates
      const relativeX = (clientX - rect.left) / rect.width;
      const relativeY = (clientY - rect.top) / rect.height;

      return {
        x: relativeX * imageDimensions.width,
        y: relativeY * imageDimensions.height,
      };
    },
    [imageDimensions]
  );

  // Helper: Update display canvas with combined mask (confirmed + pending)
  const updateDisplayCanvas = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    if (!maskCanvas || !pendingCanvas || !imageDimensions) return;

    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) return;

    // Clear display canvas
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

    // Draw confirmed mask first (if exists)
    if (confirmedMaskRef.current) {
      maskCtx.putImageData(confirmedMaskRef.current, 0, 0);
    }

    // Draw pending brush on top (with source-over to add)
    maskCtx.drawImage(pendingCanvas, 0, 0);
  }, [imageDimensions]);

  // RAF loop for display canvas updates - batches all updates to avoid lag
  useEffect(() => {
    if (!isDrawing && !hasPendingBrush) return;

    let rafId: number;
    const update = () => {
      updateDisplayCanvas();
      rafId = requestAnimationFrame(update);
    };
    rafId = requestAnimationFrame(update);

    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [isDrawing, hasPendingBrush, updateDisplayCanvas]);

  // RAF loop for cursor position updates - updates cursor DOM directly without React state
  useEffect(() => {
    if (!imageDimensions) return;

    let rafId: number;
    const updateCursor = () => {
      const cursor = cursorRef.current;
      const pos = mousePositionRef.current;

      if (cursor && pos) {
        cursor.style.display = "block";
        cursor.style.left = `${pos.x}px`;
        cursor.style.top = `${pos.y}px`;
      } else if (cursor) {
        cursor.style.display = "none";
      }

      rafId = requestAnimationFrame(updateCursor);
    };
    rafId = requestAnimationFrame(updateCursor);

    return () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [imageDimensions]);

  // Drawing handlers - similar to useMasking
  const startDrawing = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      const coords = getCanvasCoordinates(e);
      if (!coords) return;

      const ctx = ctxRef.current; // This is pending brush canvas context
      if (!ctx) return;

      setIsDrawing(true);

      if (toolType === "box") {
        // Box mode: store start position and capture initial state
        boxStartPosRef.current = coords;
        boxCurrentPosRef.current = coords;

        // Capture initial state for box preview
        const pendingCanvas = pendingBrushCanvasRef.current;
        if (pendingCanvas) {
          const tempCtx = pendingCanvas.getContext("2d", {
            willReadFrequently: true,
          });
          if (tempCtx) {
            initialBoxStateRef.current = tempCtx.getImageData(
              0,
              0,
              pendingCanvas.width,
              pendingCanvas.height
            );
          }
        }
      } else {
        // Brush mode: start path (same logic as main canvas)
        lastPointRef.current = coords;

        // Setup brush properties
        const actualBrushSize = getActualBrushSize();
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = actualBrushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";

        // Start path for continuous stroke (no initial dot, same as main canvas)
        ctx.beginPath();
        ctx.moveTo(coords.x, coords.y);
      }

      // Only set state once, not on every call
      setHasMask((prev) => (prev ? prev : true));
      setHasPendingBrush((prev) => (prev ? prev : true));

      // Display update will be handled by RAF loop
    },
    [getCanvasCoordinates, getActualBrushSize, toolType]
  );

  const draw = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (!isDrawing) return;
      e.preventDefault();

      const coords = getCanvasCoordinates(e);
      if (!coords) return;

      const ctx = ctxRef.current;
      const pendingCanvas = pendingBrushCanvasRef.current;
      if (!ctx || !pendingCanvas) return;

      if (toolType === "box") {
        // Box mode: draw preview box
        boxCurrentPosRef.current = coords;

        // Restore initial state and draw preview box
        if (initialBoxStateRef.current) {
          ctx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
          ctx.putImageData(initialBoxStateRef.current, 0, 0);
        } else {
          ctx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
        }

        // Draw preview box
        if (boxStartPosRef.current) {
          const startX = boxStartPosRef.current.x;
          const startY = boxStartPosRef.current.y;
          const width = coords.x - startX;
          const height = coords.y - startY;

          ctx.globalCompositeOperation = "source-over";
          ctx.globalAlpha = 1.0;
          ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
          ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
          ctx.lineWidth = 2;

          ctx.fillRect(startX, startY, width, height);
          ctx.strokeRect(startX, startY, width, height);
        }
      } else {
        // Brush mode: continue drawing path (same logic as main canvas)
        // Ensure brush properties are set for round, smooth brush
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        if (!ctx.strokeStyle || ctx.strokeStyle === "rgba(0, 0, 0, 0)") {
          ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
        }

        // Continue path (no beginPath/moveTo, just lineTo and stroke)
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        lastPointRef.current = coords;
      }

      // Mark that display needs update (RAF loop will handle it)
      // No direct call to updateDisplayCanvas here to avoid performance issues
    },
    [isDrawing, getCanvasCoordinates, toolType]
  );

  const stopDrawing = useCallback(() => {
    if (isDrawing) {
      setIsDrawing(false);

      if (toolType === "box") {
        // Box mode: finalize box by filling it
        if (boxStartPosRef.current && boxCurrentPosRef.current) {
          const ctx = ctxRef.current;
          const pendingCanvas = pendingBrushCanvasRef.current;
          if (ctx && pendingCanvas) {
            const startX = boxStartPosRef.current.x;
            const startY = boxStartPosRef.current.y;
            const width = boxCurrentPosRef.current.x - startX;
            const height = boxCurrentPosRef.current.y - startY;

            ctx.globalCompositeOperation = "source-over";
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
            ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
            ctx.lineWidth = 2;

            ctx.fillRect(startX, startY, width, height);
            ctx.strokeRect(startX, startY, width, height);
          }
        }
        boxStartPosRef.current = null;
        boxCurrentPosRef.current = null;
        initialBoxStateRef.current = null;
      } else {
        // Brush mode
        lastPointRef.current = null;
      }

      // Hide cursor when mouse is released (same behavior as main canvas)
      mousePositionRef.current = null;
    }
  }, [isDrawing, toolType]);

  // Handle mouse move for brush preview - only update ref, no state updates
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const newX = e.clientX - rect.left;
    const newY = e.clientY - rect.top;

    // Update ref only (no state update to avoid re-renders)
    mousePositionRef.current = { x: newX, y: newY };
  }, []);

  // Mouse move handler for drawing (only when drawing)
  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent) => {
      // Update mouse position ref
      handleMouseMove(e);
      // Draw only if currently drawing
      if (isDrawing) {
        draw(e);
      }
    },
    [handleMouseMove, isDrawing, draw]
  );

  // Mouse leave handler
  const handleCanvasMouseLeave = useCallback(() => {
    stopDrawing();
    mousePositionRef.current = null;
  }, [stopDrawing]);

  // Clear all masks (confirmed + pending)
  const clearMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;

    if (maskCanvas) {
      const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
      if (maskCtx) {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      }
    }

    if (pendingCanvas) {
      const pendingCtx = pendingCanvas.getContext("2d", {
        willReadFrequently: true,
      });
      if (pendingCtx) {
        pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
      }
    }

    confirmedMaskRef.current = null;
    setHasMask(false);
    setHasPendingBrush(false);
    setExtractedObject(null);

    // Clear box state
    boxStartPosRef.current = null;
    boxCurrentPosRef.current = null;
    initialBoxStateRef.current = null;
  }, []);

  // Segmentation detect
  const detectWithSAM = useCallback(async () => {
    if (!imageDimensions) return;

    setIsLoading(true);
    setLoadingMessage(
      hasPendingBrush
        ? toolType === "box"
          ? "Detecting object from box..."
          : "Detecting object from brush..."
        : "Auto-detecting object..."
    );

    try {
      // Use API Gateway instead of direct service call
      const API_GATEWAY_URL =
        process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
        process.env.NEXT_PUBLIC_API_URL ||
        "https://nxan2911--api-gateway.modal.run";

      // Get pending input (brush or box) as guidance
      let maskData: string | null = null;
      let bboxData: [number, number, number, number] | null = null;

      if (hasPendingBrush && pendingBrushCanvasRef.current) {
        if (
          toolType === "box" &&
          boxStartPosRef.current &&
          boxCurrentPosRef.current
        ) {
          // Box mode: calculate bbox from box coordinates
          const startX = Math.min(
            boxStartPosRef.current.x,
            boxCurrentPosRef.current.x
          );
          const startY = Math.min(
            boxStartPosRef.current.y,
            boxCurrentPosRef.current.y
          );
          const endX = Math.max(
            boxStartPosRef.current.x,
            boxCurrentPosRef.current.x
          );
          const endY = Math.max(
            boxStartPosRef.current.y,
            boxCurrentPosRef.current.y
          );

          // Ensure bbox is within canvas bounds
          const canvas = pendingBrushCanvasRef.current;
          const xMin = Math.max(0, Math.min(startX, canvas.width));
          const yMin = Math.max(0, Math.min(startY, canvas.height));
          const xMax = Math.max(xMin + 1, Math.min(endX, canvas.width));
          const yMax = Math.max(yMin + 1, Math.min(endY, canvas.height));

          bboxData = [xMin, yMin, xMax, yMax];
        } else if (toolType === "brush") {
          // Brush mode: convert pending brush to binary (white = brush area)
          const pendingCanvas = pendingBrushCanvasRef.current;
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = pendingCanvas.width;
          tempCanvas.height = pendingCanvas.height;
          const tempCtx = tempCanvas.getContext("2d", {
            willReadFrequently: true,
          });

          if (tempCtx) {
            tempCtx.fillStyle = "black";
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

            const pendingCtx = pendingCanvas.getContext("2d", {
              willReadFrequently: true,
            });
            if (pendingCtx) {
              const pendingImageData = pendingCtx.getImageData(
                0,
                0,
                pendingCanvas.width,
                pendingCanvas.height
              );
              const tempImageData = tempCtx.getImageData(
                0,
                0,
                tempCanvas.width,
                tempCanvas.height
              );

              for (let i = 0; i < pendingImageData.data.length; i += 4) {
                // If red channel has value (brush area), set to white
                if (
                  pendingImageData.data[i] > 100 &&
                  pendingImageData.data[i + 3] > 50
                ) {
                  tempImageData.data[i] = 255;
                  tempImageData.data[i + 1] = 255;
                  tempImageData.data[i + 2] = 255;
                  tempImageData.data[i + 3] = 255;
                }
              }

              tempCtx.putImageData(tempImageData, 0, 0);
            }

            maskData = tempCanvas.toDataURL("image/png").split(",")[1];
          }
        }
      }

      // Call segmentation API through API Gateway
      const endpoint = `${API_GATEWAY_URL}/api/smart-mask`;

      const requestBody: {
        image: string;
        border_adjustment: number;
        model_type: string;
        bbox?: [number, number, number, number] | null;
        mask?: string | null;
        auto_detect?: boolean;
      } = {
        image: imageData.startsWith("data:")
          ? imageData.split(",")[1]
          : imageData,
        border_adjustment: borderAdjustment,
        model_type: modelType,
      };

      if (bboxData) {
        // Box input: send bbox directly (works for both segmentation and BiRefNet)
        requestBody.bbox = bboxData;
        requestBody.auto_detect = false;
      } else if (maskData) {
        // Brush input: send mask as guidance
        // Backend will convert mask to points (centroid) for BiRefNet
        // BiRefNet will then run FastSAM first to get bbox, then refine with BiRefNet
        requestBody.mask = maskData;
        requestBody.auto_detect = false;
      } else {
        // No input: auto-detect mode
        requestBody.auto_detect = true;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error("Failed to detect object");
      }

      const data = await response.json();

      // Backend returns mask_base64 in SmartMaskResponse
      const maskBase64 = data.mask_base64 || data.mask;

      if (maskBase64 && data.success !== false) {
        // Apply detected mask and merge with existing confirmed mask
        const maskImg = new Image();
        maskImg.onload = () => {
          const maskCanvas = maskCanvasRef.current;
          const pendingCanvas = pendingBrushCanvasRef.current;
          if (!maskCanvas || !pendingCanvas) return;

          // Convert SAM result to red transparent
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = maskCanvas.width;
          tempCanvas.height = maskCanvas.height;
          const tempCtx = tempCanvas.getContext("2d", {
            willReadFrequently: true,
          });

          if (tempCtx) {
            tempCtx.drawImage(
              maskImg,
              0,
              0,
              maskCanvas.width,
              maskCanvas.height
            );
            const newMaskData = tempCtx.getImageData(
              0,
              0,
              tempCanvas.width,
              tempCanvas.height
            );

            // Convert white areas to red transparent
            for (let i = 0; i < newMaskData.data.length; i += 4) {
              if (newMaskData.data[i] > 100) {
                // White area
                newMaskData.data[i] = 255; // R
                newMaskData.data[i + 1] = 0; // G
                newMaskData.data[i + 2] = 0; // B
                newMaskData.data[i + 3] = 128; // A (semi-transparent)
              } else {
                newMaskData.data[i + 3] = 0; // Transparent
              }
            }

            // Merge with existing confirmed mask
            if (confirmedMaskRef.current) {
              const existingData = confirmedMaskRef.current.data;
              for (let i = 0; i < newMaskData.data.length; i += 4) {
                // If existing mask has content, keep it (OR operation)
                if (existingData[i + 3] > 0) {
                  newMaskData.data[i] = 255;
                  newMaskData.data[i + 1] = 0;
                  newMaskData.data[i + 2] = 0;
                  newMaskData.data[i + 3] = 128;
                }
              }
            }

            // Save as new confirmed mask
            confirmedMaskRef.current = newMaskData;

            // Clear pending brush (it's now part of confirmed mask)
            const pendingCtx = pendingCanvas.getContext("2d", {
              willReadFrequently: true,
            });
            if (pendingCtx) {
              pendingCtx.clearRect(
                0,
                0,
                pendingCanvas.width,
                pendingCanvas.height
              );
            }
            setHasPendingBrush(false);

            // Update display
            const maskCtx = maskCanvas.getContext("2d", {
              willReadFrequently: true,
            });
            if (maskCtx) {
              maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
              maskCtx.putImageData(newMaskData, 0, 0);
            }
          }
          setHasMask(true);
        };
        maskImg.src = `data:image/png;base64,${maskBase64}`;
      } else if (data.error) {
        console.error("Smart mask error:", data.error);
        alert(`Failed to detect object: ${data.error}`);
      }
    } catch (error) {
      console.error("Segmentation detection failed:", error);
      alert("Failed to detect object. Please try again.");
    } finally {
      setIsLoading(false);
      setLoadingMessage("");
    }
  }, [
    imageDimensions,
    hasPendingBrush,
    imageData,
    borderAdjustment,
    modelType,
    toolType,
  ]);

  // Extract object (remove background)
  const extractObject = useCallback(async () => {
    if (!hasMask || !imageDimensions) {
      // No mask, use original image
      onSubmit(imageData, null);
      onClose();
      return;
    }

    setIsLoading(true);
    setLoadingMessage("Extracting object...");

    try {
      // Use API Gateway instead of direct service call
      const API_GATEWAY_URL =
        process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
        process.env.NEXT_PUBLIC_API_URL ||
        "https://nxan2911--api-gateway.modal.run";
      const currentMaskDataUrl = getCurrentMaskDataUrl();

      // Get mask data
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) {
        onSubmit(imageData, currentMaskDataUrl);
        onClose();
        return;
      }

      // Convert mask to binary
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = maskCanvas.width;
      tempCanvas.height = maskCanvas.height;
      const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });

      if (!tempCtx) {
        onSubmit(imageData);
        onClose();
        return;
      }

      tempCtx.fillStyle = "black";
      tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

      const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
      if (maskCtx) {
        const maskImageData = maskCtx.getImageData(
          0,
          0,
          maskCanvas.width,
          maskCanvas.height
        );
        const tempImageData = tempCtx.getImageData(
          0,
          0,
          tempCanvas.width,
          tempCanvas.height
        );

        for (let i = 0; i < maskImageData.data.length; i += 4) {
          if (maskImageData.data[i] > 100 && maskImageData.data[i + 3] > 50) {
            tempImageData.data[i] = 255;
            tempImageData.data[i + 1] = 255;
            tempImageData.data[i + 2] = 255;
            tempImageData.data[i + 3] = 255;
          }
        }

        tempCtx.putImageData(tempImageData, 0, 0);
      }

      const maskData = tempCanvas.toDataURL("image/png").split(",")[1];

      // Call extract API through API Gateway
      const response = await fetch(
        `${API_GATEWAY_URL}/api/image-utils/extract-object`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: imageData.startsWith("data:")
              ? imageData.split(",")[1]
              : imageData,
            mask: maskData,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to extract object");
      }

      const data = await response.json();

      if (data.extracted_image) {
        const extractedDataUrl = `data:image/png;base64,${data.extracted_image}`;
        setExtractedObject(extractedDataUrl);
        onSubmit(extractedDataUrl, currentMaskDataUrl);
        onClose();
      } else {
        onSubmit(imageData, currentMaskDataUrl);
        onClose();
      }
    } catch (error) {
      console.error("Object extraction failed:", error);
      alert("Failed to extract object. Using original image.");
      onSubmit(imageData, getCurrentMaskDataUrl());
      onClose();
    } finally {
      setIsLoading(false);
      setLoadingMessage("");
    }
  }, [
    hasMask,
    imageDimensions,
    imageData,
    onSubmit,
    onClose,
    getCurrentMaskDataUrl,
  ]);

  // Handle submit
  const handleSubmit = useCallback(() => {
    if (!hasMask) {
      // No mask, use original image
      onSubmit(imageData, null);
      onClose();
    } else {
      // Has mask, extract object
      extractObject();
    }
  }, [hasMask, imageData, onSubmit, onClose, extractObject]);

  if (!isOpen) return null;

  const displayScale = imageDimensions
    ? Math.min(600 / imageDimensions.width, 500 / imageDimensions.height, 1)
    : 1;

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      fullScreen={false}
      disableEnforceFocus={false}
      disableAutoFocus={false}
      disableScrollLock={false}
      PaperProps={{
        sx: {
          bgcolor: "var(--secondary-bg)",
          color: "var(--text-primary)",
          maxHeight: { xs: "95vh", sm: "90vh" },
          borderRadius: { xs: 0, sm: 2 },
          m: { xs: 0, sm: 2 },
          width: { xs: "100%", sm: "auto" },
        },
      }}
    >
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          pb: 1,
          borderBottom: 1,
          borderColor: "var(--border-color)",
          color: "var(--text-primary)",
        }}
      >
        Edit Reference Image
        <IconButton
          onClick={onClose}
          size="small"
          sx={{ color: "var(--text-secondary)" }}
        >
          <X size={18} />
        </IconButton>
      </DialogTitle>

      <DialogContent
        sx={{
          p: 0,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Toolbar - Compact and responsive */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: { xs: 1, sm: 1.5 },
            px: { xs: 1, sm: 2 },
            py: { xs: 1, sm: 1.5 },
            borderBottom: 1,
            borderColor: "var(--border-color)",
            bgcolor: "var(--primary-bg)",
            flexWrap: "wrap",
          }}
        >
          {/* Tool selection */}
          <ToggleButtonGroup
            value={toolType}
            exclusive
            onChange={(_, value) => value && setToolType(value)}
            size="small"
            sx={{
              "& .MuiToggleButton-root": {
                borderColor: "var(--border-color)",
                color: "var(--text-secondary)",
                "&.Mui-selected": {
                  bgcolor: "var(--primary-accent)",
                  color: "white",
                  "&:hover": {
                    bgcolor: "var(--primary-accent)",
                  },
                },
              },
            }}
          >
            <Tooltip title="Brush tool">
              <span>
                <ToggleButton value="brush" disabled={isLoading}>
                  <Paintbrush size={16} />
                </ToggleButton>
              </span>
            </Tooltip>
            <Tooltip title="Box tool">
              <span>
                <ToggleButton value="box" disabled={isLoading}>
                  <Square size={16} />
                </ToggleButton>
              </span>
            </Tooltip>
          </ToggleButtonGroup>

          <Divider
            orientation="vertical"
            flexItem
            sx={{ borderColor: "var(--border-color)" }}
          />

          {/* Brush size (only show for brush tool) */}
          {toolType === "brush" && (
            <>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  minWidth: { xs: 100, sm: 120 },
                }}
              >
                <Paintbrush
                  size={14}
                  style={{ color: "var(--text-secondary)" }}
                />
                <Slider
                  value={brushSize}
                  onChange={(_, value) => setBrushSize(value as number)}
                  min={5}
                  max={100}
                  size="small"
                  sx={{
                    width: { xs: 60, sm: 80 },
                    color: "var(--primary-accent)",
                    "& .MuiSlider-thumb": {
                      backgroundColor: "var(--primary-accent)",
                    },
                    "& .MuiSlider-track": {
                      backgroundColor: "var(--primary-accent)",
                    },
                    "& .MuiSlider-rail": {
                      backgroundColor: "var(--border-color)",
                    },
                  }}
                />
                <Typography
                  variant="caption"
                  sx={{
                    color: "var(--text-secondary)",
                    minWidth: { xs: 28, sm: 32 },
                    fontSize: { xs: "0.65rem", sm: "0.75rem" },
                  }}
                >
                  {brushSize}px
                </Typography>
              </Box>
              <Divider
                orientation="vertical"
                flexItem
                sx={{
                  borderColor: "var(--border-color)",
                  display: { xs: "none", sm: "block" },
                }}
              />
            </>
          )}

          {/* Border Adjustment */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              minWidth: { xs: 120, sm: 140 },
            }}
          >
            <Typography
              variant="caption"
              sx={{
                color: "var(--text-secondary)",
                whiteSpace: "nowrap",
                fontSize: { xs: "0.65rem", sm: "0.75rem" },
                display: { xs: "none", sm: "block" },
              }}
            >
              Border:
            </Typography>
            <TextField
              type="number"
              value={borderAdjustment}
              onChange={(e) => {
                const value = parseInt(e.target.value);
                if (!isNaN(value) && value >= -10 && value <= 10) {
                  setBorderAdjustment(value);
                }
              }}
              onBlur={(e) => {
                const value = parseInt(e.target.value);
                if (isNaN(value) || value < -10) {
                  setBorderAdjustment(-10);
                } else if (value > 10) {
                  setBorderAdjustment(10);
                }
              }}
              size="small"
              inputProps={{
                min: -10,
                max: 10,
                style: { textAlign: "center", padding: "4px 8px" },
              }}
              sx={{
                width: { xs: 45, sm: 50 },
                "& .MuiOutlinedInput-root": {
                  bgcolor: "var(--primary-bg)",
                  borderColor: "var(--border-color)",
                  "& fieldset": {
                    borderColor: "var(--border-color)",
                  },
                  "&:hover fieldset": {
                    borderColor: "var(--primary-accent)",
                  },
                  "&.Mui-focused fieldset": {
                    borderColor: "var(--primary-accent)",
                  },
                },
                "& .MuiInputBase-input": {
                  color: "var(--text-primary)",
                  fontSize: { xs: "0.7rem", sm: "0.75rem" },
                },
              }}
            />
            <Box sx={{ width: { xs: 60, sm: 80 } }}>
              <Slider
                value={borderAdjustment}
                onChange={(_, value) => setBorderAdjustment(value as number)}
                min={-10}
                max={10}
                step={1}
                size="small"
                valueLabelDisplay="auto"
                sx={{
                  color: "var(--primary-accent)",
                  "& .MuiSlider-thumb": {
                    backgroundColor: "var(--primary-accent)",
                  },
                  "& .MuiSlider-track": {
                    backgroundColor: "var(--primary-accent)",
                  },
                  "& .MuiSlider-rail": {
                    backgroundColor: "var(--border-color)",
                  },
                }}
              />
            </Box>
          </Box>

          <Divider
            orientation="vertical"
            flexItem
            sx={{
              borderColor: "var(--border-color)",
              display: { xs: "none", sm: "block" },
            }}
          />

          {/* Model Type Selection */}
          {onModelTypeChange && (
            <>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  flexWrap: { xs: "wrap", sm: "nowrap" },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: "var(--text-secondary)",
                    whiteSpace: "nowrap",
                    fontSize: { xs: "0.65rem", sm: "0.75rem" },
                    display: { xs: "none", sm: "block" },
                  }}
                >
                  Model:
                </Typography>
                <ButtonGroup size="small" sx={{ height: 28 }}>
                  <Button
                    onClick={() => onModelTypeChange("segmentation")}
                    disabled={isLoading}
                    variant={
                      modelType === "segmentation" ? "contained" : "outlined"
                    }
                    sx={{
                      minWidth: { xs: 60, sm: 70 },
                      fontSize: { xs: "0.65rem", sm: "0.7rem" },
                      bgcolor:
                        modelType === "segmentation"
                          ? "var(--primary-accent)"
                          : "transparent",
                      color:
                        modelType === "segmentation"
                          ? "white"
                          : "var(--text-secondary)",
                      borderColor: "var(--border-color)",
                      "&:hover": {
                        bgcolor:
                          modelType === "segmentation"
                            ? "var(--primary-accent)"
                            : "var(--hover-bg)",
                        borderColor: "var(--primary-accent)",
                      },
                    }}
                  >
                    FastSAM
                  </Button>
                  <Button
                    onClick={() => onModelTypeChange("birefnet")}
                    disabled={isLoading}
                    variant={
                      modelType === "birefnet" ? "contained" : "outlined"
                    }
                    sx={{
                      minWidth: { xs: 60, sm: 70 },
                      fontSize: { xs: "0.65rem", sm: "0.7rem" },
                      bgcolor:
                        modelType === "birefnet"
                          ? "var(--primary-accent)"
                          : "transparent",
                      color:
                        modelType === "birefnet"
                          ? "white"
                          : "var(--text-secondary)",
                      borderColor: "var(--border-color)",
                      "&:hover": {
                        bgcolor:
                          modelType === "birefnet"
                            ? "var(--primary-accent)"
                            : "var(--hover-bg)",
                        borderColor: "var(--primary-accent)",
                      },
                    }}
                  >
                    BiRefNet
                  </Button>
                </ButtonGroup>
              </Box>
              <Divider
                orientation="vertical"
                flexItem
                sx={{
                  borderColor: "var(--border-color)",
                  display: { xs: "none", sm: "block" },
                }}
              />
            </>
          )}

          {/* Action buttons */}
          <Box
            sx={{
              display: "flex",
              gap: { xs: 0.5, sm: 1 },
              ml: { xs: "auto", sm: "auto" },
            }}
          >
            <Tooltip
              title={
                hasMask && !hasPendingBrush
                  ? "Add object"
                  : hasPendingBrush
                  ? toolType === "box"
                    ? "Detect from box"
                    : "Detect from brush"
                  : "Auto detect"
              }
            >
              <span>
                <Button
                  onClick={detectWithSAM}
                  disabled={isLoading}
                  size="small"
                  startIcon={
                    hasMask && !hasPendingBrush ? (
                      <Plus size={14} />
                    ) : (
                      <Wand2 size={14} />
                    )
                  }
                  sx={{
                    bgcolor: "var(--primary-accent)",
                    color: "white",
                    "&:hover": {
                      bgcolor: "var(--highlight-accent)",
                    },
                    fontSize: { xs: "0.7rem", sm: "0.75rem" },
                    px: { xs: 1, sm: 1.5 },
                    minWidth: { xs: "auto", sm: "auto" },
                  }}
                >
                  {hasMask && !hasPendingBrush
                    ? "Add"
                    : hasPendingBrush
                    ? "Detect"
                    : "Auto"}
                </Button>
              </span>
            </Tooltip>
            <Tooltip title="Clear mask">
              <span>
                <IconButton
                  onClick={clearMask}
                  disabled={isLoading || !hasMask}
                  size="small"
                  sx={{
                    color: "var(--text-secondary)",
                    "&:hover": {
                      bgcolor: "var(--hover-bg)",
                      color: "var(--text-primary)",
                    },
                  }}
                >
                  <Trash2 size={16} />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        </Box>

        {/* Canvas area */}
        <Box
          ref={containerRef}
          sx={{
            flex: 1,
            overflow: "auto",
            p: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "var(--primary-bg)",
            position: "relative",
            minHeight: 400,
          }}
        >
          {isLoading && (
            <Box
              sx={{
                position: "absolute",
                inset: 0,
                bgcolor: "rgba(0, 0, 0, 0.5)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: 10,
              }}
            >
              <Box sx={{ textAlign: "center" }}>
                <CircularProgress
                  size={32}
                  sx={{
                    color: "var(--primary-accent)",
                    mb: 1,
                  }}
                />
                <Typography
                  variant="body2"
                  sx={{ color: "var(--text-secondary)" }}
                >
                  {loadingMessage}
                </Typography>
              </Box>
            </Box>
          )}

          {!imageDimensions && (
            <Typography variant="body2" sx={{ color: "var(--text-secondary)" }}>
              Loading image...
            </Typography>
          )}

          {/* Hidden canvas for pending brush strokes */}
          <canvas ref={pendingBrushCanvasRef} className="hidden" />

          {imageDimensions && (
            <Box
              sx={{
                position: "relative",
                border: 1,
                borderColor: "var(--border-color)",
                borderRadius: 1,
                overflow: "hidden",
                width: Math.max(imageDimensions.width * displayScale, 200),
                height: Math.max(imageDimensions.height * displayScale, 200),
              }}
            >
              {/* Main image canvas */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 rounded-lg"
                style={{
                  width: "100%",
                  height: "100%",
                  imageRendering: "auto",
                }}
              />

              {/* Mask overlay canvas */}
              <canvas
                ref={maskCanvasRef}
                className="absolute inset-0 rounded-lg cursor-none"
                style={{
                  width: "100%",
                  height: "100%",
                  imageRendering: "auto",
                  touchAction: "none",
                }}
                onMouseDown={startDrawing}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={stopDrawing}
                onMouseLeave={handleCanvasMouseLeave}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />

              {/* Brush preview cursor - updated via RAF loop, no React state */}
              {imageDimensions && (
                <div
                  ref={cursorRef}
                  className="pointer-events-none absolute"
                  style={{
                    display: "none",
                    transform: "translate(-50%, -50%)",
                  }}
                >
                  {/* Outer ring - black border */}
                  <div
                    className="absolute rounded-full border-2 border-black/70"
                    style={{
                      width: getActualBrushSize() * displayScale,
                      height: getActualBrushSize() * displayScale,
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                  {/* Inner ring - white */}
                  <div
                    className="absolute rounded-full border border-white/90"
                    style={{
                      width: getActualBrushSize() * displayScale - 2,
                      height: getActualBrushSize() * displayScale - 2,
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                  {/* Center dot */}
                  <div
                    className="absolute w-1 h-1 bg-white rounded-full border border-black/50"
                    style={{
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                </div>
              )}
            </Box>
          )}
        </Box>
      </DialogContent>

      {/* Footer */}
      <DialogActions
        sx={{
          px: { xs: 1.5, sm: 2 },
          py: { xs: 1, sm: 1.5 },
          borderTop: 1,
          borderColor: "var(--border-color)",
          bgcolor: "var(--primary-bg)",
          justifyContent: "space-between",
          flexDirection: { xs: "column", sm: "row" },
          gap: { xs: 1.5, sm: 2 },
          alignItems: { xs: "stretch", sm: "center" },
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: "var(--text-secondary)",
            flex: { xs: 0, sm: 1 },
            fontSize: { xs: "0.65rem", sm: "0.75rem" },
            textAlign: { xs: "center", sm: "left" },
            lineHeight: 1.4,
            wordWrap: "break-word",
            overflowWrap: "break-word",
            maxWidth: "100%",
            minWidth: 0,
            px: { xs: 0.5, sm: 0 },
          }}
        >
          {hasMask
            ? "Object selected. Click Submit to extract."
            : "Draw on image to select object, or click Auto detect. Submit without selection to use original image."}
        </Typography>
        <Box
          sx={{
            display: "flex",
            gap: { xs: 1, sm: 1.5 },
            width: { xs: "100%", sm: "auto" },
            justifyContent: { xs: "stretch", sm: "flex-end" },
            flexShrink: 0,
            minWidth: 0,
          }}
        >
          <Button
            onClick={onClose}
            disabled={isLoading}
            size="small"
            sx={{
              color: "var(--text-secondary)",
              flex: { xs: 1, sm: 0 },
              minWidth: { xs: 0, sm: 64 },
              whiteSpace: "nowrap",
              "&:hover": {
                bgcolor: "var(--hover-bg)",
                color: "var(--text-primary)",
              },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isLoading}
            variant="contained"
            size="small"
            startIcon={<Check size={14} />}
            sx={{
              bgcolor: "var(--primary-accent)",
              color: "white",
              flex: { xs: 1, sm: 0 },
              minWidth: { xs: 0, sm: 80 },
              whiteSpace: "nowrap",
              "&:hover": {
                bgcolor: "var(--highlight-accent)",
              },
              "& .MuiButton-startIcon": {
                marginRight: { xs: 0.5, sm: 1 },
              },
            }}
          >
            Submit
          </Button>
        </Box>
      </DialogActions>
    </Dialog>
  );
}
