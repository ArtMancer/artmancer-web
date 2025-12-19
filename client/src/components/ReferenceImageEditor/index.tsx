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
  Eraser,
  Eye,
  EyeOff,
} from "lucide-react";
import { Dialog, Slider, ToggleGroup, Checkbox, Tooltip } from "radix-ui";

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
  enableSmartMasking?: boolean; // Whether smart masking is enabled
  onSmartMaskingChange?: (enabled: boolean) => void; // Callback for smart masking toggle
  onNotification?: (type: "success" | "error" | "info" | "warning", message: string) => void; // Notification callback
}

export default function ReferenceImageEditor({
  isOpen,
  imageData,
  onClose,
  onSubmit,
  initialMaskData: _initialMaskData,
  modelType = "segmentation",
  onModelTypeChange,
  borderAdjustment: propBorderAdjustment = 0,
  onBorderAdjustmentChange,
  enableSmartMasking: propEnableSmartMasking = true,
  onSmartMaskingChange,
  onNotification,
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
  const [toolType, setToolType] = useState<"brush" | "box" | "eraser">("brush");
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
  const [brushSize, setBrushSize] = useState(30); // UI value 1-100
  const [hasMask, setHasMask] = useState(false);
  const [hasPendingBrush, setHasPendingBrush] = useState(false); // Track if user drew new strokes (brush or box)
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [, setExtractedObject] = useState<string | null>(null);
  const [isMaskVisible, setIsMaskVisible] = useState(true);

  // Keep _initialMaskData for potential future use (e.g., loading existing mask)
  void _initialMaskData;
  // Use prop borderAdjustment if provided, otherwise use local state
  const [localBorderAdjustment, setLocalBorderAdjustment] = useState(0);
  const borderAdjustment =
    propBorderAdjustment !== undefined
      ? propBorderAdjustment
      : localBorderAdjustment;
  const setBorderAdjustment =
    onBorderAdjustmentChange || setLocalBorderAdjustment;
  // Use prop enableSmartMasking if provided, otherwise use local state (default: true)
  const [localEnableSmartMasking, setLocalEnableSmartMasking] = useState(true);
  const enableSmartMasking =
    propEnableSmartMasking !== undefined
      ? propEnableSmartMasking
      : localEnableSmartMasking;
  const setEnableSmartMasking =
    onSmartMaskingChange || setLocalEnableSmartMasking;

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

  // Cleanup state when dialog closes
  useEffect(() => {
    if (!isOpen) {
      // Reset all state when dialog closes
      setIsDrawing(false);
      setHasMask(false);
      setHasPendingBrush(false);
      setIsLoading(false);
      setLoadingMessage("");
      setIsMaskVisible(true);
      setBrushSize(30);
      setToolType("brush");
      
      // Clear refs
      lastPointRef.current = null;
      mousePositionRef.current = null;
      boxStartPosRef.current = null;
      boxCurrentPosRef.current = null;
      initialBoxStateRef.current = null;
      confirmedMaskRef.current = null;
      
      // Clear canvases
      const maskCanvas = maskCanvasRef.current;
      const pendingCanvas = pendingBrushCanvasRef.current;
      if (maskCanvas) {
        const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
        if (maskCtx) {
          maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        }
      }
      if (pendingCanvas) {
        const pendingCtx = pendingCanvas.getContext("2d", { willReadFrequently: true });
        if (pendingCtx) {
          pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
        }
      }
    }
  }, [isOpen]);

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
      if (onNotification) {
        onNotification("error", "Failed to load image");
      }
    };
    img.src = imageData;
  }, [isOpen, imageData, onNotification]);

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
    ctx.strokeStyle = "rgba(255, 0, 0, 0.3)";
    ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
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

  // Normalize mask opacity to ensure consistent rgba(255, 0, 0, 0.3) display (same as main canvas)
  const normalizeMaskOpacity = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;

    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) return;

    // Get current image data
    const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    // Normalize opacity: any pixel with alpha > 0 should be set to rgba(255, 0, 0, 0.3)
    // Opacity mờ hơn (0.3 thay vì 0.5) để dễ nhìn ảnh phía sau
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        // Pixel has been drawn - normalize to consistent opacity
        data[i] = 255; // R
        data[i + 1] = 0; // G
        data[i + 2] = 0; // B
        data[i + 3] = 76; // A (0.3 opacity = 76/255)
      }
    }

    // Put normalized data back
    maskCtx.putImageData(imageData, 0, 0);
  }, []);

  // Helper: Update display canvas with combined mask (confirmed + pending)
  const updateDisplayCanvas = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    if (!maskCanvas || !pendingCanvas || !imageDimensions) return;

    // If eraser is being used, don't update display canvas (eraser draws directly on maskCanvas)
    if (toolType === "eraser" && isDrawing) {
      return;
    }

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

    // Note: Don't normalize opacity here in RAF loop for performance
    // Normalize only when needed (after merge, after SAM detection, etc.)
  }, [imageDimensions, toolType, isDrawing]);

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
      } else if (toolType === "eraser") {
        // Eraser mode: draw directly on maskCanvas to erase confirmed mask
        const maskCanvas = maskCanvasRef.current;
        if (!maskCanvas) return;
        
        const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
        if (!maskCtx) return;
        
        lastPointRef.current = coords;

        // Setup eraser properties
        const actualBrushSize = getActualBrushSize();
        maskCtx.globalAlpha = 1.0;
        maskCtx.lineWidth = actualBrushSize;
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        maskCtx.imageSmoothingEnabled = true;
        maskCtx.imageSmoothingQuality = "high";
        maskCtx.globalCompositeOperation = "destination-out";

        // Start path for continuous stroke
        maskCtx.beginPath();
        maskCtx.moveTo(coords.x, coords.y);
        
        // Also erase from pending canvas if it has content
        const pendingCtx = ctxRef.current;
        if (pendingCtx) {
          pendingCtx.globalCompositeOperation = "destination-out";
          pendingCtx.globalAlpha = 1.0;
          pendingCtx.lineWidth = actualBrushSize;
          pendingCtx.lineCap = "round";
          pendingCtx.lineJoin = "round";
          pendingCtx.beginPath();
          pendingCtx.moveTo(coords.x, coords.y);
        }
      } else {
        // Brush mode: start path on pending canvas
        const ctx = ctxRef.current; // This is pending brush canvas context
        if (!ctx) return;
        
        lastPointRef.current = coords;

        // Setup brush properties
        const actualBrushSize = getActualBrushSize();
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = actualBrushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = "rgba(255, 0, 0, 0.3)";

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

      if (toolType === "box") {
        // Box mode: draw preview box
        const ctx = ctxRef.current;
        const pendingCanvas = pendingBrushCanvasRef.current;
        if (!ctx || !pendingCanvas) return;
        
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
          ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
          ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
          ctx.lineWidth = 2;

          ctx.fillRect(startX, startY, width, height);
          ctx.strokeRect(startX, startY, width, height);
        }
      } else if (toolType === "eraser") {
        // Eraser mode: draw directly on maskCanvas to erase confirmed mask
        const maskCanvas = maskCanvasRef.current;
        if (!maskCanvas) return;
        
        const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
        if (!maskCtx) return;
        
        // Continue eraser path on maskCanvas
        maskCtx.lineTo(coords.x, coords.y);
        maskCtx.stroke();
        
        // Also erase from pending canvas if it has content
        const pendingCtx = ctxRef.current;
        if (pendingCtx) {
          pendingCtx.lineTo(coords.x, coords.y);
          pendingCtx.stroke();
        }

        lastPointRef.current = coords;
      } else {
        // Brush mode: continue drawing path on pending canvas
        const ctx = ctxRef.current;
        if (!ctx) return;
        
        // Ensure brush properties are set for round, smooth brush
        ctx.globalAlpha = 1.0;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.globalCompositeOperation = "source-over";
        if (!ctx.strokeStyle || ctx.strokeStyle === "rgba(0, 0, 0, 0)") {
          ctx.strokeStyle = "rgba(255, 0, 0, 0.3)";
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
            ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
            ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
            ctx.lineWidth = 2;

            ctx.fillRect(startX, startY, width, height);
            ctx.strokeRect(startX, startY, width, height);
          }
        }
        boxStartPosRef.current = null;
        boxCurrentPosRef.current = null;
        initialBoxStateRef.current = null;
      } else if (toolType === "eraser") {
        // Eraser mode: eraser was drawn directly on maskCanvas, so confirmed mask is already updated
        // Just need to update confirmedMaskRef from current maskCanvas state
        const maskCanvas = maskCanvasRef.current;
        if (maskCanvas) {
          const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
          if (maskCtx) {
            // Update confirmed mask ref from current maskCanvas state
            confirmedMaskRef.current = maskCtx.getImageData(
              0,
              0,
              maskCanvas.width,
              maskCanvas.height
            );
          }
        }
        // Clear pending canvas (eraser strokes are already applied to maskCanvas)
        const pendingCtx = ctxRef.current;
        if (pendingCtx) {
          const pendingCanvas = pendingBrushCanvasRef.current;
          if (pendingCanvas) {
            pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
          }
        }
        lastPointRef.current = null;
      } else {
        // Brush mode
        lastPointRef.current = null;
      }

      // CRITICAL: Merge pending canvas to display canvas one final time before normalizing
      // This ensures all strokes are on display canvas
      const maskCanvas = maskCanvasRef.current;
      const pendingCanvas = pendingBrushCanvasRef.current;
      if (maskCanvas && pendingCanvas && hasPendingBrush) {
        const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
        if (maskCtx) {
          // Merge pending brush/box into display canvas
          maskCtx.globalCompositeOperation = "source-over";
          maskCtx.globalAlpha = 1.0;
          maskCtx.drawImage(pendingCanvas, 0, 0);
          
          // Update confirmed mask ref
          confirmedMaskRef.current = maskCtx.getImageData(
            0,
            0,
            maskCanvas.width,
            maskCanvas.height
          );
          
          // Clear pending canvas
          const pendingCtx = ctxRef.current;
          if (pendingCtx) {
            pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
          }
          
          setHasPendingBrush(false);
        }
      }

      // NOW normalize mask opacity on the merged display canvas
      normalizeMaskOpacity();

      // Hide cursor when mouse is released (same behavior as main canvas)
      mousePositionRef.current = null;
    }
  }, [isDrawing, toolType, normalizeMaskOpacity, hasPendingBrush]);

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
              
              // Normalize opacity to ensure consistent display (same as main canvas)
              normalizeMaskOpacity();
            }
          }
          setHasMask(true);
        };
        maskImg.src = `data:image/png;base64,${maskBase64}`;
      } else if (data.error) {
        console.error("Smart mask error:", data.error);
        if (onNotification) {
          onNotification("error", `Failed to detect object: ${data.error}`);
        } else {
          alert(`Failed to detect object: ${data.error}`);
        }
      }
    } catch (error) {
      console.error("Segmentation detection failed:", error);
      const errorMessage = error instanceof Error ? error.message : "Failed to detect object. Please try again.";
      if (onNotification) {
        onNotification("error", errorMessage);
      } else {
        alert(errorMessage);
      }
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
    normalizeMaskOpacity,
    onNotification,
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
      const errorMessage = "Failed to extract object. Using original image.";
      if (onNotification) {
        onNotification("warning", errorMessage);
      } else {
        alert(errorMessage);
      }
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
    onNotification,
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

  // Calculate display scale to fit within dialog (accounting for toolbar and padding)
  const displayScale = imageDimensions
    ? Math.min(
        (typeof window !== "undefined" ? window.innerWidth * 0.9 : 1200) / imageDimensions.width,
        (typeof window !== "undefined" ? window.innerHeight * 0.85 : 800) / imageDimensions.height,
        1
      )
    : 1;

  return (
    <Dialog.Root open={isOpen} onOpenChange={(open: boolean) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay fixed inset-0 bg-black/50 backdrop-blur z-[1300]" />
        <Dialog.Content className="dialog-content fixed inset-2 sm:inset-4 bg-secondary-bg text-text-primary max-h-[calc(100vh-1rem)] sm:max-h-[calc(100vh-2rem)] max-w-[calc(100vw-1rem)] sm:max-w-[calc(100vw-2rem)] rounded-lg flex flex-col overflow-hidden z-[1301]">
          <div className="flex items-center justify-between border-b border-border-color px-3 py-2">
            <Dialog.Title className="text-sm font-semibold text-text-primary">
              Edit Reference Image
            </Dialog.Title>
            <Dialog.Close asChild>
              <button
                onClick={onClose}
                className="p-1 text-text-secondary hover:text-text-primary transition-colors"
                aria-label="Close"
              >
                <X size={18} />
              </button>
            </Dialog.Close>
          </div>

          <div className="flex flex-col overflow-hidden">
            {/* Toolbar */}
            <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-border-color bg-primary-bg">
              <ToggleGroup.Root
                type="single"
                value={toolType}
                onValueChange={(value: string) => value && setToolType(value as "brush" | "box" | "eraser")}
                className="inline-flex rounded-md border border-border-color overflow-hidden"
              >
                <ToggleGroup.Item
                  value="brush"
                  disabled={isLoading}
                  title="Brush tool"
                  className={`tool-item px-2 py-1 text-sm flex items-center justify-center border-r border-border-color transition-all duration-150 ${
                    toolType === "brush"
                      ? "tool-item-active bg-primary-accent text-white"
                      : "bg-secondary-bg text-text-secondary hover:bg-[var(--hover-bg)]"
                  } ${isLoading ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <Paintbrush size={16} className={toolType === "brush" ? "tool-icon-rotate" : ""} data-active={toolType === "brush"} />
                </ToggleGroup.Item>
                <ToggleGroup.Item
                  value="box"
                  disabled={isLoading}
                  title="Box tool"
                  className={`tool-item px-2 py-1 text-sm flex items-center justify-center border-r border-border-color transition-all duration-150 ${
                    toolType === "box"
                      ? "tool-item-active bg-primary-accent text-white"
                      : "bg-secondary-bg text-text-secondary hover:bg-[var(--hover-bg)]"
                  } ${isLoading ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <Square size={16} />
                </ToggleGroup.Item>
                <ToggleGroup.Item
                  value="eraser"
                  disabled={isLoading}
                  title="Eraser tool"
                  className={`tool-item px-2 py-1 text-sm flex items-center justify-center transition-all duration-150 ${
                    toolType === "eraser"
                      ? "tool-item-active bg-primary-accent text-white"
                      : "bg-secondary-bg text-text-secondary hover:bg-[var(--hover-bg)]"
                  } ${isLoading ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <Eraser size={16} className={toolType === "eraser" ? "tool-icon-rotate" : ""} data-active={toolType === "eraser"} />
                </ToggleGroup.Item>
              </ToggleGroup.Root>

              <div className="hidden sm:block h-6 w-px bg-border-color" />

              {(toolType === "brush" || toolType === "eraser") && (
                <>
                  <div className="flex items-center gap-2 min-w-[120px]">
                    {toolType === "brush" ? (
                      <Paintbrush size={14} className="text-text-secondary" />
                    ) : (
                      <Eraser size={14} className="text-text-secondary" />
                    )}
                    <Slider.Root
                      min={1}
                      max={100}
                      step={1}
                      value={[brushSize]}
                      onValueChange={([value]: number[]) => setBrushSize(value)}
                      className="relative flex h-5 w-[80px] touch-none select-none items-center"
                    >
                      <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                        <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
                      </Slider.Track>
                      <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
                    </Slider.Root>
                    <span className="text-[0.75rem] text-text-secondary min-w-[32px]">
                      {brushSize}px
                    </span>
                  </div>
                  <div className="hidden sm:block h-6 w-px bg-border-color" />
                </>
              )}

              {/* Smart Masking Toggle */}
              <label className="flex items-center gap-2 text-text-primary text-sm cursor-pointer">
                <Checkbox.Root
                  checked={enableSmartMasking}
                  onCheckedChange={(checked: boolean | "indeterminate") =>
                    setEnableSmartMasking(!!checked)
                  }
                  disabled={isLoading}
                  className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
                    enableSmartMasking
                      ? "bg-primary-accent border-primary-accent"
                      : "bg-transparent border-border-color"
                  } ${isLoading ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
                >
                  <Checkbox.Indicator>
                    <Check size={12} strokeWidth={3} className="text-white" />
                  </Checkbox.Indicator>
                </Checkbox.Root>
                <span className="text-[0.75rem] text-text-secondary whitespace-nowrap">
                  Smart Masking
                </span>
              </label>

              {enableSmartMasking && onModelTypeChange && (
                <>
                  <div className="hidden sm:block h-6 w-px bg-border-color" />
                  <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
                    <span className="hidden sm:inline text-[0.75rem] text-text-secondary whitespace-nowrap">
                      Model:
                    </span>
                    <div className="inline-flex rounded border border-border-color overflow-hidden">
                      <button
                        onClick={() => {
                          onModelTypeChange("segmentation");
                        }}
                        disabled={isLoading}
                        className={`px-3 py-1 text-[0.7rem] ${
                          modelType === "segmentation"
                            ? "bg-primary-accent text-white"
                            : "bg-transparent text-text-secondary hover:bg-[var(--hover-bg)]"
                        } ${isLoading ? "opacity-60 cursor-not-allowed" : ""}`}
                      >
                        FastSAM
                      </button>
                      <button
                        onClick={() => {
                          onModelTypeChange("birefnet");
                        }}
                        disabled={isLoading}
                        className={`px-3 py-1 text-[0.7rem] border-l border-border-color ${
                          modelType === "birefnet"
                            ? "bg-primary-accent text-white"
                            : "bg-transparent text-text-secondary hover:bg-[var(--hover-bg)]"
                        } ${isLoading ? "opacity-60 cursor-not-allowed" : ""}`}
                      >
                        BiRefNet
                      </button>
                    </div>
                  </div>
                </>
              )}

              {enableSmartMasking && (
                <>
                  <div className="hidden sm:block h-6 w-px bg-border-color" />
                  <div className="flex items-center gap-2 min-w-[140px]">
                    <span className="hidden sm:inline text-[0.75rem] text-text-secondary whitespace-nowrap">
                      Border:
                    </span>
                    <input
                      type="number"
                      min={-10}
                      max={10}
                      value={borderAdjustment}
                      onChange={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (!Number.isNaN(value) && value >= -10 && value <= 10) {
                          setBorderAdjustment(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (Number.isNaN(value) || value < -10) {
                          setBorderAdjustment(-10);
                        } else if (value > 10) {
                          setBorderAdjustment(10);
                        }
                      }}
                      className="w-14 rounded border border-border-color bg-primary-bg text-text-primary text-[0.7rem] px-2 py-1 text-center"
                    />
                    <div className="w-[80px]">
                      <Slider.Root
                        min={-10}
                        max={10}
                        step={1}
                        value={[borderAdjustment]}
                        onValueChange={([value]: number[]) => setBorderAdjustment(value)}
                        className="relative flex h-5 w-full touch-none select-none items-center"
                      >
                        <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                          <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
                        </Slider.Track>
                        <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
                      </Slider.Root>
                    </div>
                  </div>
                </>
              )}

              {/* Mask Visibility Toggle */}
              {(hasMask || hasPendingBrush) && (
                <>
                  <div className="hidden sm:block h-6 w-px bg-border-color" />
                  <Tooltip.Root delayDuration={300}>
                    <Tooltip.Trigger asChild>
                      <button
                        onClick={() => setIsMaskVisible(!isMaskVisible)}
                        className={`btn-interactive p-2 rounded border transition-colors ${
                          isMaskVisible
                            ? "bg-secondary-bg hover:bg-primary-accent text-text-primary hover:text-white border-border-color"
                            : "bg-primary-accent hover:bg-[var(--highlight-accent)] text-white border-primary-accent"
                        }`}
                      >
                        {isMaskVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content className="bg-secondary-bg border border-border-color text-text-primary px-2 py-1 rounded text-xs shadow-lg" sideOffset={5}>
                        {isMaskVisible ? "Hide mask" : "Show mask"}
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                </>
              )}

              <div className="flex items-center gap-2 ml-auto">
                <button
                  onClick={detectWithSAM}
                  disabled={isLoading || !enableSmartMasking}
                  title={
                    !enableSmartMasking
                      ? "Enable Smart Masking to use auto detect"
                      : hasMask && !hasPendingBrush
                      ? "Add object"
                      : hasPendingBrush
                      ? toolType === "box"
                        ? "Detect from box"
                        : "Detect from brush"
                      : "Auto detect"
                  }
                  className={`inline-flex items-center gap-1 rounded px-3 py-2 text-xs font-medium transition-colors ${
                    isLoading || !enableSmartMasking
                      ? "opacity-60 cursor-not-allowed bg-primary-accent text-white"
                      : "bg-primary-accent text-white hover:bg-[var(--highlight-accent)]"
                  }`}
                >
                  {hasMask && !hasPendingBrush ? (
                    <Plus size={14} />
                  ) : (
                    <Wand2 size={14} />
                  )}
                  {hasMask && !hasPendingBrush
                    ? "Add"
                    : hasPendingBrush
                    ? "Detect"
                    : "Auto"}
                </button>
                <button
                  onClick={clearMask}
                  disabled={isLoading || !hasMask}
                  title="Clear mask"
                  className={`p-2 rounded border border-border-color text-text-secondary hover:bg-[var(--hover-bg)] transition-colors ${
                    isLoading || !hasMask ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>

            {/* Canvas area */}
            <div
              ref={containerRef}
              className="flex-1 overflow-auto p-4 flex items-center justify-center bg-primary-bg relative min-h-[400px]"
            >
              {isLoading && (
                <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center z-10">
                  <div className="h-8 w-8 rounded-full border-2 border-primary-accent border-t-transparent animate-spin mb-2" />
                  <p className="text-text-secondary text-sm">{loadingMessage}</p>
                </div>
              )}

              {!imageDimensions && (
                <p className="text-text-secondary text-sm">Loading image...</p>
              )}

              <canvas ref={pendingBrushCanvasRef} className="hidden" />

              {imageDimensions && (
                <div
                  className="relative border border-border-color rounded-md overflow-hidden"
                  style={{
                    width: Math.max(imageDimensions.width * displayScale, 200),
                    height: Math.max(imageDimensions.height * displayScale, 200),
                  }}
                >
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 rounded-lg"
                    style={{
                      width: "100%",
                      height: "100%",
                      imageRendering: "auto",
                    }}
                  />

                  <canvas
                    ref={maskCanvasRef}
                    className="absolute inset-0 rounded-lg cursor-none transition-opacity duration-200"
                    style={{
                      width: "100%",
                      height: "100%",
                      imageRendering: "auto",
                      touchAction: "none",
                      opacity: isMaskVisible ? 1 : 0,
                    }}
                    onMouseDown={startDrawing}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={stopDrawing}
                    onMouseLeave={handleCanvasMouseLeave}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                  />

                  {imageDimensions && (
                    <div
                      ref={cursorRef}
                      className="pointer-events-none absolute"
                      style={{
                        display: "none",
                        transform: "translate(-50%, -50%)",
                      }}
                    >
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
                </div>
              )}
            </div>
          </div>

          <div className="px-4 py-3 border-t border-border-color bg-primary-bg flex flex-col sm:flex-row gap-3 sm:gap-4 items-stretch sm:items-center justify-between">
            <p className="text-text-secondary text-[0.75rem] leading-relaxed sm:flex-1 text-center sm:text-left">
              {hasMask
                ? "Object selected. Click Submit to extract."
                : "Draw on image to select object, or click Auto detect. Submit without selection to use original image."}
            </p>
            <div className="flex gap-2 w-full sm:w-auto justify-stretch sm:justify-end">
              <button
                onClick={onClose}
                disabled={isLoading}
                className={`flex-1 sm:flex-none rounded border border-border-color px-3 py-2 text-sm font-medium text-text-secondary hover:bg-[var(--hover-bg)] transition-colors ${
                  isLoading ? "opacity-60 cursor-not-allowed" : ""
                }`}
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className={`flex-1 sm:flex-none inline-flex items-center justify-center gap-2 rounded bg-primary-accent text-white px-4 py-2 text-sm font-medium transition-colors ${
                  isLoading ? "opacity-60 cursor-not-allowed" : "hover:bg-[var(--highlight-accent)]"
                }`}
              >
                <Check size={14} />
                Submit
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
