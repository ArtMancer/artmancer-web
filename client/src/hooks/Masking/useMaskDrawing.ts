import { useState, useRef, useCallback, useEffect } from "react";
import { detectEdges, floodFill } from "../../utils/edgeDetection";

// Canvas state snapshot for commands
export interface CanvasState {
  imageData: ImageData | null;
  hasContent: boolean;
}

export interface UseMaskDrawingParams {
  imageDimensions: { width: number; height: number } | null;
  maskBrushSize: number;
  maskToolType: "brush" | "box" | "eraser";
  isMaskingMode: boolean;
  enableEdgeDetection: boolean;
  enableFloodFill: boolean;
  imageRef?: React.RefObject<HTMLImageElement | null>;
  edgeOverlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  skipMaskResetRef?: React.MutableRefObject<boolean>;
}

export interface UseMaskDrawingReturn {
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  ctxRef: React.MutableRefObject<CanvasRenderingContext2D | null>;
  isMaskDrawing: boolean;
  setIsMaskDrawing: (drawing: boolean) => void;
  hasCanvasContent: boolean;
  setHasCanvasContent: (hasContent: boolean) => void;
  boxStartPosRef: React.MutableRefObject<{ x: number; y: number } | null>;
  boxCurrentPosRef: React.MutableRefObject<{ x: number; y: number } | null>;
  initialDrawState: React.MutableRefObject<CanvasState | null>;
  getCanvasCoordinates: (e: React.MouseEvent) => { x: number; y: number };
  captureCanvasState: () => CanvasState;
  checkCanvasHasContent: () => boolean;
  normalizeMaskOpacity: () => void;
  startDrawing: (e: React.MouseEvent) => void;
  draw: (e: React.MouseEvent | MouseEvent) => void;
  stopDrawing: (
    onStrokeComplete?: (state: CanvasState) => void,
    onBoxComplete?: (bbox: [number, number, number, number]) => void
  ) => void;
  autoDetectAndFillAfterStroke: () => Promise<void>;
  getFinalMask: () => Promise<string | null>;
  saveMaskCanvasState: () => string | null;
  restoreMaskCanvasState: (savedState: string | null) => Promise<boolean>;
}

/**
 * Hook to manage mask canvas drawing operations
 * Handles brush and box drawing, canvas setup, coordinate conversion, and edge detection
 */
export function useMaskDrawing(
  params: UseMaskDrawingParams
): UseMaskDrawingReturn {
  const {
    imageDimensions,
    maskBrushSize,
    maskToolType,
    isMaskingMode,
    enableEdgeDetection,
    enableFloodFill,
    imageRef,
    edgeOverlayCanvasRef,
    skipMaskResetRef,
  } = params;

  const [isMaskDrawing, setIsMaskDrawing] = useState(false);
  const [hasCanvasContent, setHasCanvasContent] = useState(false);

  // Box drawing state
  const boxStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const boxCurrentPosRef = useRef<{ x: number; y: number } | null>(null);

  // Track last valid canvas position for brush stroke continuation
  const lastValidCanvasPosRef = useRef<{ x: number; y: number } | null>(null);
  // Track if we were outside canvas (to detect when returning)
  const wasOutsideCanvasRef = useRef(false);

  // Store initial state when starting to draw (for command creation)
  const initialDrawState = useRef<CanvasState | null>(null);

  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const globalListenersAttachedRef = useRef(false);

  // Helper function to check if canvas has content (for display purposes)
  const checkCanvasHasContent = useCallback((): boolean => {
    const canvas = maskCanvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      return false;
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return false;
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Check full buffer to avoid missing sparse masks
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        if (r > 0 || g > 0 || b > 0 || a > 0) {
          return true;
        }
      }
      return false;
    } catch (e) {
      console.warn("Could not check canvas content:", e);
      return false;
    }
  }, []);

  // Helper function to capture current canvas state
  const captureCanvasState = useCallback((): CanvasState => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      console.warn("No canvas available for state capture");
      return { imageData: null, hasContent: false };
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      console.warn("No canvas context available for state capture");
      return { imageData: null, hasContent: false };
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      let hasContent = false;

      // Check full buffer to avoid missing sparse masks
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        if (r > 0 || g > 0 || b > 0 || a > 0) {
          hasContent = true;
          break;
        }
      }

      return { imageData, hasContent };
    } catch (error) {
      console.error("Failed to capture canvas state:", error);
      return { imageData: null, hasContent: false };
    }
  }, []);

  // Get canvas coordinates from mouse event
  // This function allows coordinates outside canvas bounds (negative or > canvas size)
  // to support continuous strokes when mouse moves outside canvas
  const getCanvasCoordinates = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const canvas = maskCanvasRef.current;
      if (!canvas || !imageDimensions) return { x: 0, y: 0 };

      // Get the canvas bounding rect
      // Note: Canvas is inside TransformLayer which applies transform.scale
      // The bounding rect already accounts for the transform applied by CSS
      const canvasRect = canvas.getBoundingClientRect();

      // Calculate the mouse position relative to the canvas CSS size
      // The canvas CSS size = imageDimensions * displayScale
      // But the canvas internal size = imageDimensions (no displayScale)
      // Allow negative values and values > 1 to support coordinates outside canvas
      const relativeX = (e.clientX - canvasRect.left) / canvasRect.width;
      const relativeY = (e.clientY - canvasRect.top) / canvasRect.height;

      // Convert to canvas internal coordinates
      // Canvas internal resolution = imageDimensions (no displayScale)
      // CSS size = imageDimensions * displayScale
      // Transform is applied at TransformLayer level, bounding rect already accounts for it
      // This can result in negative values or values > canvas.width/height when outside canvas
      const x = relativeX * imageDimensions.width;
      const y = relativeY * imageDimensions.height;

      return { x, y };
    },
    [imageDimensions]
  );

  // Normalize mask opacity to ensure consistent visual appearance
  // This prevents opacity accumulation when strokes overlap
  const normalizeMaskOpacity = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    // Get current image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
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
    ctx.putImageData(imageData, 0, 0);
  }, []);

  // Auto detect edge and fill after stroke
  const autoDetectAndFillAfterStroke = useCallback(async () => {
    const canvas = maskCanvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx || !imageRef?.current || !imageDimensions) return;

    // Only proceed if edge detection or flood fill is enabled
    if (!enableEdgeDetection && !enableFloodFill) {
      // Clear edge overlay if disabled
      const edgeCanvas = edgeOverlayCanvasRef.current;
      if (edgeCanvas) {
        const edgeCtx = edgeCanvas.getContext("2d", { willReadFrequently: true });
        if (edgeCtx) {
          edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
        }
      }
      return;
    }

    try {
      // Get current mask data
      const maskData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Find all pixels that have mask (red pixels with alpha > 0)
      const maskPixels: Array<{ x: number; y: number }> = [];
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const idx = (y * canvas.width + x) * 4;
          // Check if pixel has mask (red channel > 0 and alpha > 0)
          if (maskData.data[idx] > 0 && maskData.data[idx + 3] > 0) {
            maskPixels.push({ x, y });
          }
        }
      }

      if (maskPixels.length === 0) return;

      // Create temporary canvas for edge detection in masked region
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = imageDimensions.width;
      tempCanvas.height = imageDimensions.height;
      const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
      if (!tempCtx) return;

      // Draw the image
      tempCtx.drawImage(
        imageRef.current,
        0,
        0,
        imageDimensions.width,
        imageDimensions.height
      );

      // Detect edges in the image
      const detectedEdges = detectEdges(tempCanvas, 50);
      if (!detectedEdges) return;

      // Initialize edge overlay canvas
      const edgeCanvas = edgeOverlayCanvasRef.current;
      if (edgeCanvas) {
        edgeCanvas.width = imageDimensions.width;
        edgeCanvas.height = imageDimensions.height;
        const edgeCtx = edgeCanvas.getContext("2d", { willReadFrequently: true });
        if (!edgeCtx) return;

        // Clear previous edge overlay
        edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);

        // Draw edges only in masked regions (yellow/cyan overlay)
        const edgeData = detectedEdges.data;
        const edgeImageData = edgeCtx.createImageData(
          edgeCanvas.width,
          edgeCanvas.height
        );

        for (const { x, y } of maskPixels) {
          const edgeIdx = (y * edgeCanvas.width + x) * 4;
          const edgeValue = edgeData[edgeIdx];

          if (edgeValue > 128) {
            // Draw edge in yellow/cyan color
            const pixelIdx = (y * edgeCanvas.width + x) * 4;
            edgeImageData.data[pixelIdx] = 255; // R - Yellow
            edgeImageData.data[pixelIdx + 1] = 255; // G
            edgeImageData.data[pixelIdx + 2] = 0; // B
            edgeImageData.data[pixelIdx + 3] = 150; // A - Semi-transparent
          }
        }

        edgeCtx.putImageData(edgeImageData, 0, 0);
      }

      // If flood fill is enabled, fill mask to edges
      if (enableFloodFill && detectedEdges) {
        // Get image data for flood fill
        const imageData = tempCtx.getImageData(
          0,
          0,
          imageDimensions.width,
          imageDimensions.height
        );

        // For each mask pixel, perform flood fill to edge
        // We'll use the center of the mask region as starting point
        if (maskPixels.length > 0) {
          const centerX = Math.floor(
            maskPixels.reduce((sum, p) => sum + p.x, 0) / maskPixels.length
          );
          const centerY = Math.floor(
            maskPixels.reduce((sum, p) => sum + p.y, 0) / maskPixels.length
          );

          const fillColor = { r: 255, g: 0, b: 0, a: 128 };
          const filledData = floodFill(
            imageData,
            centerX,
            centerY,
            fillColor,
            detectedEdges,
            10 // tolerance
          );

          // Merge filled data with existing mask
          const currentMaskData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let i = 0; i < filledData.data.length; i += 4) {
            // If filled pixel has mask, add to current mask
            if (filledData.data[i] > 0 || filledData.data[i + 3] > 0) {
              currentMaskData.data[i] = 255; // R
              currentMaskData.data[i + 1] = 0; // G
              currentMaskData.data[i + 2] = 0; // B
              currentMaskData.data[i + 3] = Math.max(
                currentMaskData.data[i + 3],
                filledData.data[i + 3]
              ); // A
            }
          }
          ctx.putImageData(currentMaskData, 0, 0);
        }
      }
    } catch (error) {
      console.error("Error in auto detect and fill:", error);
    }
  }, [imageRef, imageDimensions, enableEdgeDetection, enableFloodFill, edgeOverlayCanvasRef]);

  // Initialize canvas context - ONLY when imageDimensions change
  // DO NOT include maskBrushSize here to avoid clearing canvas when brush size changes
  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      return;
    }

    // Set canvas size if image dimensions are available
    // WARNING: Setting canvas.width or canvas.height clears the canvas!
    if (imageDimensions) {
      canvas.width = imageDimensions.width;
      canvas.height = imageDimensions.height;
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return;
    }

    // Set initial canvas properties
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    
    // Enable anti-aliasing for smooth brush edges
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctxRef.current = ctx;
    
    // Note: Brush size and colors will be set by the separate useEffect below
    // This prevents clearing the canvas when only brush size changes
  }, [imageDimensions]);

  // Update canvas drawing properties when brush settings change
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !imageDimensions) return;

    // Interpret maskBrushSize as brush RADIUS in pixels (UI shows "px")
    // Scale up for better visual feedback on high-res images
    const brushRadius = (maskBrushSize || 1) * 2;
    const brushSize = brushRadius * 2; // diameter
    const opacity = 0.3; // Fixed 30% opacity (mờ hơn để dễ nhìn)

    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha
    ctx.lineWidth = brushSize;
    ctx.lineCap = "round"; // Round brush tip like Krita/Photoshop
    ctx.lineJoin = "round"; // Round line joins
    ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
    ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;
    // Enable anti-aliasing for smooth brush edges
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  }, [maskBrushSize, imageDimensions]);

  // Start drawing handler
  const startDrawing = useCallback(
    (e: React.MouseEvent) => {
      if (!isMaskingMode || !imageDimensions) return;

      e.preventDefault();
      e.stopPropagation();

      // Capture initial state before drawing for command creation
      initialDrawState.current = captureCanvasState();

      const { x, y } = getCanvasCoordinates(e);
      let ctx = ctxRef.current;

      // Recreate context if it was reset (e.g., canvas resized)
      if (!ctx) {
        const canvas = maskCanvasRef.current;
        if (!canvas) return;
        // Ensure intrinsic size matches image
        canvas.width = imageDimensions.width;
        canvas.height = imageDimensions.height;
        ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;
        ctxRef.current = ctx;
        // Reset drawing props
        const brushRadius = (maskBrushSize || 1) * 2;
        const brushSize = brushRadius * 2;
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
        ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
      }

      // Ensure opacity and composite operation are set correctly before starting to draw
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha
      // Ensure strokeStyle with opacity is set (should already be set, but ensure it)
      if (!ctx.strokeStyle || ctx.strokeStyle === "rgba(0, 0, 0, 0)") {
        ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
      }

      if (maskToolType === "box") {
        // Box mode: store start position
        boxStartPosRef.current = { x, y };
        boxCurrentPosRef.current = { x, y };
        setIsMaskDrawing(true);
      } else {
        // Brush mode: start path
        // Check if starting position is within canvas bounds
        const isWithinBounds =
          x >= 0 &&
          y >= 0 &&
          x <= imageDimensions.width &&
          y <= imageDimensions.height;
        
        // Reset outside canvas flag when starting new stroke
        wasOutsideCanvasRef.current = false;
        
        if (isWithinBounds) {
          ctx.beginPath();
          ctx.moveTo(x, y);
          lastValidCanvasPosRef.current = { x, y };
        } else {
          // Start outside canvas - initialize path but don't draw yet
          ctx.beginPath();
          lastValidCanvasPosRef.current = null;
          wasOutsideCanvasRef.current = true;
        }
        setIsMaskDrawing(true);
      }
    },
    [
      isMaskingMode,
      imageDimensions,
      getCanvasCoordinates,
      captureCanvasState,
      maskToolType,
      maskBrushSize,
    ]
  );

  // Draw handler
  const draw = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      if (!isMaskDrawing || !isMaskingMode) return;

      e.preventDefault();

      const { x, y } = getCanvasCoordinates(e);
      const ctx = ctxRef.current;
      const canvas = maskCanvasRef.current;
      if (!ctx || !canvas) {
        return;
      }

      // Check if coordinates are within canvas bounds
      const isWithinBounds =
        x >= 0 && y >= 0 && x <= canvas.width && y <= canvas.height;

      if (maskToolType === "box") {
        // Box mode: draw preview box
        // Store current position (even if outside canvas for preview calculation)
        boxCurrentPosRef.current = { x, y };

        // Restore canvas state and draw preview box
        if (initialDrawState.current?.imageData) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.putImageData(initialDrawState.current.imageData, 0, 0);
        } else {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        // Draw preview box - clamp coordinates to canvas bounds for display
        if (boxStartPosRef.current) {
          const startX = boxStartPosRef.current.x;
          const startY = boxStartPosRef.current.y;
          
          // Clamp coordinates to canvas bounds for preview display
          const clampedX = Math.max(0, Math.min(x, canvas.width));
          const clampedY = Math.max(0, Math.min(y, canvas.height));
          const clampedStartX = Math.max(0, Math.min(startX, canvas.width));
          const clampedStartY = Math.max(0, Math.min(startY, canvas.height));
          
          const width = clampedX - clampedStartX;
          const height = clampedY - clampedStartY;

          ctx.save();
          ctx.globalCompositeOperation = "source-over";
          ctx.globalAlpha = 1.0;
          ctx.fillStyle = "rgba(255, 0, 0, 0.3)";
          ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
          ctx.lineWidth = 2;

          ctx.fillRect(clampedStartX, clampedStartY, width, height);
          ctx.strokeRect(clampedStartX, clampedStartY, width, height);
          ctx.restore();
        }
      } else {
        // Brush mode: continue drawing path
        // Only draw if within canvas bounds
        if (isWithinBounds) {
          // Ensure line width always reflects latest maskBrushSize
          const brushRadius = (maskBrushSize || 1) * 2;
          const brushSize = brushRadius * 2;
          ctx.lineWidth = brushSize;
          // If we were outside canvas and are now returning, moveTo last valid position
          if (wasOutsideCanvasRef.current && lastValidCanvasPosRef.current) {
            ctx.moveTo(lastValidCanvasPosRef.current.x, lastValidCanvasPosRef.current.y);
            wasOutsideCanvasRef.current = false;
          }

          // Track last valid position for stroke continuation
          lastValidCanvasPosRef.current = { x, y };

          // Mode: brush (paint) or eraser (destination-out)
          if (maskToolType === "eraser") {
            ctx.globalCompositeOperation = "destination-out";
            ctx.strokeStyle = "rgba(0,0,0,1)";
          } else {
            ctx.globalCompositeOperation = "source-over";
            ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
          }
          ctx.globalAlpha = 1.0;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          if (!ctx.strokeStyle || ctx.strokeStyle === "rgba(0, 0, 0, 0)") {
            ctx.strokeStyle =
              maskToolType === "eraser"
                ? "rgba(0,0,0,1)"
                : "rgba(255, 0, 0, 0.5)";
          }

          ctx.lineTo(x, y);
          ctx.stroke();
        } else {
          // Outside canvas - mark that we're outside
          wasOutsideCanvasRef.current = true;
          // Don't draw, but stroke continues (tracked by global listeners)
        }
      }
    },
    [isMaskDrawing, isMaskingMode, getCanvasCoordinates, maskToolType, maskBrushSize]
  );

  // Stop drawing handler
  const stopDrawing = useCallback(
    (
      onStrokeComplete?: (state: CanvasState) => void,
      onBoxComplete?: (bbox: [number, number, number, number]) => void
    ) => {
      const ctx = ctxRef.current;
      const canvas = maskCanvasRef.current;

      if (maskToolType === "box") {
        // Box mode: handle box drawing
        if (canvas && ctx && boxStartPosRef.current && boxCurrentPosRef.current) {
          const startX = boxStartPosRef.current.x;
          const startY = boxStartPosRef.current.y;
          const endX = boxCurrentPosRef.current.x;
          const endY = boxCurrentPosRef.current.y;

          // Calculate box dimensions
          const width = endX - startX;
          const height = endY - startY;

          // Only process if box has meaningful size
          if (Math.abs(width) > 2 && Math.abs(height) > 2) {
            // Calculate bbox [xMin, yMin, xMax, yMax] and clamp to canvas bounds
            const xMin = Math.max(0, Math.min(startX, endX));
            const yMin = Math.max(0, Math.min(startY, endY));
            const xMax = Math.min(canvas.width, Math.max(startX, endX));
            const yMax = Math.min(canvas.height, Math.max(startY, endY));
            
            // Only commit if clamped box has meaningful size
            if (xMax - xMin > 2 && yMax - yMin > 2) {
              const bbox: [number, number, number, number] = [xMin, yMin, xMax, yMax];

              // Call callback if provided
              if (onBoxComplete) {
                onBoxComplete(bbox);
              }
            }
          }
        }

        // Reset box state
        boxStartPosRef.current = null;
        boxCurrentPosRef.current = null;
        lastValidCanvasPosRef.current = null;
        wasOutsideCanvasRef.current = false;
        setIsMaskDrawing(false);
        return;
      }

      // Brush mode: just close path and mark canvas has content; avoid post-processing
      if (ctx) {
        ctx.closePath();
      }

      if (canvas) {
        const state = captureCanvasState();
        if (state.hasContent) {
          setHasCanvasContent(true);
        }

        if (onStrokeComplete) {
          onStrokeComplete(state);
        }
      }

      // Reset last valid position when stopping
      lastValidCanvasPosRef.current = null;
      wasOutsideCanvasRef.current = false;
      setIsMaskDrawing(false);
    },
    [maskToolType, captureCanvasState]
  );

  // Global mouse listeners để giữ stroke khi rời canvas
  // LƯU Ý: Listener thực tế được gắn ở hook useMasking (cấp cao hơn)
  // để có thể gọi đúng callback stopDrawing kèm smart masking & history.
  // Ở đây chỉ giữ ref trạng thái, không tự gắn listener nữa.
  useEffect(() => {
    // Chỉ đánh dấu flag, không attach listener tại đây
    if (!isMaskDrawing || !isMaskingMode) {
      globalListenersAttachedRef.current = false;
      return;
    }

    // Khi bắt đầu vẽ, đánh dấu để cấp trên biết có thể gắn global listener nếu cần
    globalListenersAttachedRef.current = true;

    return () => {
      globalListenersAttachedRef.current = false;
    };
  }, [isMaskDrawing, isMaskingMode]);

  // Hàm chuyển đổi Mask đỏ/trong suốt (UI) thành Mask Đen/Trắng chuẩn cho AI
  // Quy ước: TRẮNG = vùng cần sửa (mask area), ĐEN = vùng giữ nguyên (keep area)
  const getFinalMask = useCallback(async (): Promise<string | null> => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !imageDimensions) return null;

    // Ensure canvas only contains pixels within bounds before generating mask
    const originalCtx = canvas.getContext("2d", { willReadFrequently: true });
    if (!originalCtx) return null;

    // Get imageData (only contains pixels within bounds: 0, 0, width, height)
    // This ensures we only process pixels within canvas bounds
    let imageData = originalCtx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Double-check: Clear and redraw to ensure no out-of-bounds pixels
    // This is a safety measure to ensure canvas only contains pixels within bounds
    originalCtx.clearRect(0, 0, canvas.width, canvas.height);
    originalCtx.putImageData(imageData, 0, 0);
    
    // Get imageData again after cleanup
    imageData = originalCtx.getImageData(0, 0, canvas.width, canvas.height);

    // 1. Tạo canvas tạm để xử lý với kích thước đúng (bằng imageDimensions)
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = imageDimensions.width;
    tempCanvas.height = imageDimensions.height;
    const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;

    // 2. Tô nền ĐEN (Vùng giữ nguyên - không sửa)
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;

    // 4. Xử lý Binary Mask: Chuyển vùng có vẽ (alpha > 0) thành TRẮNG (vùng cần sửa)
    // Tạo ImageData mới cho canvas tạm
    const newImageData = ctx.createImageData(tempCanvas.width, tempCanvas.height);
    const newData = newImageData.data;

    // Canvas size đã match imageDimensions, nên copy trực tiếp
    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3]; // Lấy độ trong suốt của nét vẽ hiện tại

      if (alpha > 0) {
        // Có nét vẽ -> Set thành TRẮNG (255, 255, 255, 255) = Vùng cần sửa
        newData[i] = 255; // R
        newData[i + 1] = 255; // G
        newData[i + 2] = 255; // B
        newData[i + 3] = 255; // Alpha
      } else {
        // Không vẽ -> Giữ nguyên màu ĐEN (0, 0, 0, 255) = Vùng giữ nguyên
        newData[i] = 0;
        newData[i + 1] = 0;
        newData[i + 2] = 0;
        newData[i + 3] = 255; // Alpha nền luôn phải là 255
      }
    }

    // 5. Đưa dữ liệu đã chuẩn hóa vào canvas tạm
    ctx.putImageData(newImageData, 0, 0);

    // 6. Xuất ra base64 string (PNG format)
    return tempCanvas.toDataURL("image/png");
  }, [imageDimensions]);

  // Lưu mask canvas state (UI mask brush - đỏ trong suốt) để restore sau
  const saveMaskCanvasState = useCallback((): string | null => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;

    try {
      // Lưu toàn bộ canvas state (bao gồm cả mask brush mà user vẽ)
      const canvasData = canvas.toDataURL("image/png");
      console.log("💾 [useMaskDrawing] Saved mask canvas state");
      return canvasData;
    } catch (error) {
      console.error("❌ [useMaskDrawing] Failed to save mask canvas state:", error);
      return null;
    }
  }, []);

  // Restore mask canvas từ saved state - đơn giản, chỉ load một lần
  const restoreMaskCanvasState = useCallback(
    (savedState: string | null): Promise<boolean> => {
      return new Promise((resolve) => {
        const canvas = maskCanvasRef.current;
        if (!canvas || !savedState || !imageDimensions) {
          console.warn(
            "⚠️ [useMaskDrawing] Cannot restore: missing canvas, savedState, or imageDimensions"
          );
          resolve(false);
          return;
        }

        // Kiểm tra canvas dimensions đã đúng chưa
        if (
          canvas.width !== imageDimensions.width ||
          canvas.height !== imageDimensions.height
        ) {
          console.warn(
            `⚠️ [useMaskDrawing] Canvas dimensions mismatch: current=${canvas.width}x${canvas.height}, expected=${imageDimensions.width}x${imageDimensions.height}`
          );
          resolve(false);
          return;
        }

        try {
          const ctx = canvas.getContext("2d", { willReadFrequently: true });
          if (!ctx) {
            console.warn("⚠️ [useMaskDrawing] Cannot get canvas context");
            resolve(false);
            return;
          }

          // Re-initialize context settings (giống như trong resizeCanvas)
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          ctx.globalCompositeOperation = "source-over";
          ctx.globalAlpha = 1.0;

          // Apply current drawing properties
          const brushRadius = (maskBrushSize || 1) * 2;
          const brushSize = brushRadius * 2;
          const opacity = 0.5;
          ctx.lineWidth = brushSize;
          ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
          ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;

          // Clear canvas trước
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Load saved state
          const img = new Image();
          img.onload = () => {
            try {
              // Draw saved mask với dimensions hiện tại
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

              // Đợi một chút để canvas render xong trước khi check content
              setTimeout(() => {
                const hasContent = checkCanvasHasContent();
                console.log("✅ [useMaskDrawing] Restored mask canvas state", {
                  canvasSize: `${canvas.width}x${canvas.height}`,
                  imageSize: `${img.width}x${img.height}`,
                  hasContent,
                });
                // Update hasCanvasContent state
                setHasCanvasContent(hasContent);
                resolve(true); // Resolve sau khi img.onload và check content xong
              }, 10); // Đợi 10ms để canvas render xong
            } catch (error) {
              console.error("❌ [useMaskDrawing] Failed to draw restored mask canvas:", error);
              resolve(false);
            }
          };
          img.onerror = () => {
            console.error("❌ [useMaskDrawing] Failed to load saved mask canvas state");
            resolve(false);
          };
          img.src = savedState;
        } catch (error) {
          console.error("❌ [useMaskDrawing] Failed to restore mask canvas state:", error);
          resolve(false);
        }
      });
    },
    [imageDimensions, maskBrushSize, checkCanvasHasContent]
  );

  // Handle canvas resizing when image dimensions change
  useEffect(() => {
    if (!maskCanvasRef.current || !imageDimensions) return;

    const canvas = maskCanvasRef.current;

    const resizeCanvas = () => {
      // Use actual image dimensions, not container size
      // This ensures mask matches the original image size when exported
      const newWidth = imageDimensions.width;
      const newHeight = imageDimensions.height;

      // Check if canvas dimensions are changing
      const dimensionsChanged =
        canvas.width !== newWidth || canvas.height !== newHeight;

      // Check if we should preserve mask (when returning to original or after generation)
      const shouldPreserveMask = skipMaskResetRef?.current === true;

      // Store canvas content before resize if:
      // 1. Dimensions are changing, OR
      // 2. We should preserve mask (even if dimensions don't change, setting canvas.width/height clears it)
      let imageData = null;
      if (
        (dimensionsChanged || shouldPreserveMask) &&
        canvas.width > 0 &&
        canvas.height > 0
      ) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          try {
            imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          } catch (e) {
            console.warn("Could not save canvas content:", e);
          }
        }
      }

      // Set canvas internal resolution to match the actual image dimensions
      // NOTE: Setting canvas.width/height ALWAYS clears the canvas in HTML5, even if values are the same
      // This is why we need to preserve content even when dimensions don't change
      canvas.width = newWidth;
      canvas.height = newHeight;

      // Canvas positioning is handled by CSS (absolute inset-0 inside transform div)
      // Canvas now fills the entire parent div automatically

      // Always re-initialize context settings after canvas resize
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        // Re-initialize context settings for maximum brush hardness
        ctx.lineCap = "butt"; // Sharp line ends
        ctx.lineJoin = "miter"; // Sharp corners
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha

        // Enable anti-aliasing for smooth brush edges (like Krita/Photoshop)
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";

        // Apply current drawing properties
        const brushRadius = (maskBrushSize || 1) * 2;
        const brushSize = brushRadius * 2;
        const opacity = 0.5; // Fixed 50% opacity

        ctx.lineWidth = brushSize;
        ctx.lineCap = "round"; // Round brush tip
        ctx.lineJoin = "round"; // Round line joins
        ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
        ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;

        // Restore canvas content if it was saved
        // Restore if: dimensions changed OR we should preserve mask
        if (imageData && (dimensionsChanged || shouldPreserveMask)) {
          try {
            ctx.putImageData(imageData, 0, 0);
            console.log("💾 [resizeCanvas] Preserved mask content", {
              dimensionsChanged,
              shouldPreserveMask,
              canvasSize: `${canvas.width}x${canvas.height}`,
            });
            // After restoring, update hasCanvasContent state
            // Use setTimeout to ensure state update happens after canvas operation
            setTimeout(() => {
              const hasContent = checkCanvasHasContent();
              setHasCanvasContent(hasContent);
            }, 0);
          } catch (e) {
            console.warn("Could not restore canvas content:", e);
          }
        } else if (shouldPreserveMask) {
          // Nếu shouldPreserveMask nhưng không có imageData, log warning
          console.warn(
            "⚠️ [resizeCanvas] Should preserve mask but no imageData available",
            {
              canvasSize: `${canvas.width}x${canvas.height}`,
              dimensionsChanged,
              shouldPreserveMask,
            }
          );
        }

        // Reset the flag after preserving canvas content
        // NHƯNG chỉ reset nếu KHÔNG có savedMaskCanvasState đang chờ restore
        // (savedMaskCanvasState sẽ được restore sau, nên không reset flag ngay)
        // Flag sẽ được reset sau khi restoreMaskCanvasState hoàn thành
        if (shouldPreserveMask && skipMaskResetRef) {
          // Không reset ngay, để đảm bảo mask không bị clear bởi resizeCanvas lần sau
          // skipMaskResetRef.current = false;
          console.log("🔒 [resizeCanvas] Keeping skipMaskResetRef=true for mask restore");
        }

        ctxRef.current = ctx;
      }
    };

    // Initial resize - try immediately, then retry after a short delay
    resizeCanvas();
    const timer = setTimeout(resizeCanvas, 50);
    const timer2 = setTimeout(resizeCanvas, 200);

    // Listen for window resize
    window.addEventListener("resize", resizeCanvas);

    return () => {
      clearTimeout(timer);
      clearTimeout(timer2);
      window.removeEventListener("resize", resizeCanvas);
    };
    // We intentionally do NOT depend on maskBrushSize here because resizing the
    // canvas when only brush size changes would clear the mask. Brush settings
    // are applied via separate effects and draw handlers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageDimensions, skipMaskResetRef, checkCanvasHasContent]);

  // Check canvas content periodically to update hasCanvasContent state
  useEffect(() => {
    if (!maskCanvasRef.current || !imageDimensions) {
      setHasCanvasContent(false);
      return;
    }

    // Check canvas content after a short delay to allow canvas operations to complete
    const checkContent = () => {
      const hasContent = checkCanvasHasContent();
      setHasCanvasContent(hasContent);
    };

    // Check immediately and after a delay
    checkContent();
    const timer = setTimeout(checkContent, 100);
    const timer2 = setTimeout(checkContent, 300);

    return () => {
      clearTimeout(timer);
      clearTimeout(timer2);
    };
  }, [imageDimensions, checkCanvasHasContent, isMaskingMode]);

  return {
    maskCanvasRef,
    ctxRef,
    isMaskDrawing,
    setIsMaskDrawing,
    hasCanvasContent,
    setHasCanvasContent,
    boxStartPosRef,
    boxCurrentPosRef,
    initialDrawState,
    getCanvasCoordinates,
    captureCanvasState,
    checkCanvasHasContent,
    normalizeMaskOpacity,
    startDrawing,
    draw,
    stopDrawing,
    autoDetectAndFillAfterStroke,
    getFinalMask,
    saveMaskCanvasState,
    restoreMaskCanvasState,
  };
}

