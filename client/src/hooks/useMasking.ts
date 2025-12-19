import { useState, useRef, useCallback, useEffect } from "react";
import { detectEdges } from "../utils/edgeDetection";
import { useMaskDrawing } from "./Masking/useMaskDrawing";
import { useMaskHistory } from "./Masking/useMaskHistory";
import { useSmartMasking } from "./Masking/useSmartMasking";
import type { CanvasState } from "./Masking/useMaskDrawing";

/**
 * Main hook for mask management
 * Composes useMaskDrawing, useMaskHistory, and useSmartMasking hooks
 */
export function useMasking(
  uploadedImage: string | null,
  imageDimensions: { width: number; height: number } | null,
  imageContainerRef: React.RefObject<HTMLDivElement | null>,
  transform: { scale: number },
  viewportZoom: number,
  imageRef?: React.RefObject<HTMLImageElement | null>,
  onNotification?: (
    type: "success" | "error" | "info" | "warning",
    message: string
  ) => void,
  borderAdjustment: number = 0,
  skipMaskResetRef?: React.MutableRefObject<boolean>
) {
  // Masking state
  const [isMaskingMode, setIsMaskingMode] = useState(false);
  const [isMaskVisible, setIsMaskVisible] = useState(true);
  const [maskBrushSize, setMaskBrushSize] = useState(20); // Default 20% for better visibility
  const [maskToolType, setMaskToolType] = useState<"brush" | "box" | "eraser">(
    "brush"
  );
  const [enableEdgeDetection, setEnableEdgeDetection] = useState(false);
  const [enableFloodFill, setEnableFloodFill] = useState(false);
  // Note: edgeMask is generated but not currently used in the return value
  // It's kept for potential future use with edge detection visualization
  const [, setEdgeMask] = useState<ImageData | null>(null);

  // Edge overlay canvas ref
  const edgeOverlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Store state before box is drawn (for smart mask detection)
  const stateBeforeBoxRef = useRef<CanvasState | null>(null);

  // Track previous uploadedImage to detect actual changes
  const previousUploadedImageRef = useRef<string | null>(null);

  // Initialize useMaskDrawing hook
  const drawingHook = useMaskDrawing({
    imageDimensions,
    maskBrushSize,
    maskToolType,
    isMaskingMode,
    enableEdgeDetection,
    enableFloodFill,
    imageRef,
    edgeOverlayCanvasRef,
    skipMaskResetRef,
  });

  // Initialize useMaskHistory hook
  const historyHook = useMaskHistory({
    maskCanvasRef: drawingHook.maskCanvasRef,
    captureCanvasState: drawingHook.captureCanvasState,
    maxHistorySize: 20,
  });

  // Ensure edge overlay canvas has correct intrinsic size when image dims change
  // NOTE: maskCanvas size is already handled in useMaskDrawing's resizeCanvas effect
  // We only handle edgeOverlayCanvas here to avoid clearing mask on every state change
  useEffect(() => {
    const edgeCanvas = edgeOverlayCanvasRef?.current;
    if (imageDimensions && edgeCanvas) {
      edgeCanvas.width = imageDimensions.width;
      edgeCanvas.height = imageDimensions.height;
    }
  }, [imageDimensions, edgeOverlayCanvasRef]);
const maskCanvasRef = drawingHook.maskCanvasRef;

  // Initialize useSmartMasking hook
  const smartMaskingHook = useSmartMasking({
    uploadedImage,
    imageDimensions,
    imageRef,
    maskCanvasRef: drawingHook.maskCanvasRef,
    ctxRef: drawingHook.ctxRef,
    borderAdjustment,
    initialDrawState: drawingHook.initialDrawState,
    stateBeforeBoxRef,
    captureCanvasState: drawingHook.captureCanvasState,
    executeDrawCommand: historyHook.executeDrawCommand,
    onNotification,
  });

  // Generate edge mask from image
  const generateEdgeMask = useCallback(async (): Promise<ImageData | null> => {
    if (!imageRef?.current || !imageDimensions) return null;

    try {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = imageDimensions.width;
      tempCanvas.height = imageDimensions.height;
      const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
      if (!tempCtx) return null;

      tempCtx.drawImage(
        imageRef.current,
        0,
        0,
        imageDimensions.width,
        imageDimensions.height
      );

      const edgeData = detectEdges(tempCanvas, 50);
      return edgeData;
    } catch (error) {
      console.error("Error generating edge mask:", error);
      return null;
    }
  }, [imageRef, imageDimensions]);

  // Generate edge mask when enableEdgeDetection is true
  useEffect(() => {
    if (
      !enableEdgeDetection ||
      !uploadedImage ||
      !imageDimensions ||
      !imageRef?.current
    ) {
      setEdgeMask(null);
      return;
    }

    let isCancelled = false;

    generateEdgeMask()
      .then((mask) => {
      if (!isCancelled) {
        setEdgeMask(mask);
      }
      })
      .catch((error) => {
      if (!isCancelled) {
          console.error("Error generating edge mask:", error);
        setEdgeMask(null);
      }
    });

    return () => {
      isCancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    enableEdgeDetection,
    uploadedImage,
    imageDimensions?.width,
    imageDimensions?.height,
    imageRef?.current,
  ]);

// Reset mask history helper (needs to be declared before effects below)
const resetMaskHistory = useCallback(() => {
  historyHook.resetMaskHistory();
      setIsMaskingMode(false);
  drawingHook.setIsMaskDrawing(false);
  drawingHook.setHasCanvasContent(false);

      const canvas = maskCanvasRef.current;
      if (canvas) {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }

  // Clear edge overlay if any
    const edgeCanvas = edgeOverlayCanvasRef.current;
      if (edgeCanvas) {
    const edgeCtx = edgeCanvas.getContext("2d", { willReadFrequently: true });
        if (edgeCtx) {
          edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
        }
      }

  // Force ctx re-init on next draw
  drawingHook.ctxRef.current = null;
  drawingHook.initialDrawState.current = null;
  drawingHook.boxStartPosRef.current = null;
  drawingHook.boxCurrentPosRef.current = null;
}, [historyHook, drawingHook, maskCanvasRef, edgeOverlayCanvasRef]);

  // Reset mask history when uploaded image changes
useEffect(() => {
  if (skipMaskResetRef?.current === true) {
    previousUploadedImageRef.current = uploadedImage;
      return;
    }

  if (uploadedImage === previousUploadedImageRef.current) {
        return;
      }

  previousUploadedImageRef.current = uploadedImage;

  if (uploadedImage) {
    resetMaskHistory();
    setIsMaskingMode(false);
    drawingHook.setIsMaskDrawing(false);
    setEdgeMask(null);
    smartMaskingHook.clearSmartMaskState();
  }
  }, [uploadedImage, skipMaskResetRef, resetMaskHistory, drawingHook, smartMaskingHook]);

// Clear command history when exiting masking mode
useEffect(() => {
  if (skipMaskResetRef?.current === true) {
      return;
    }

  if (!isMaskingMode) {
    resetMaskHistory();
  }
}, [isMaskingMode, resetMaskHistory, skipMaskResetRef]);

  // Wrapper for stopDrawing that integrates smart masking and history
  const stopDrawing = useCallback(() => {
    drawingHook.stopDrawing(
      // onStrokeComplete callback
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      (_currentState: CanvasState) => {
        // Normalize opacity after brush stroke to make it more transparent
        drawingHook.normalizeMaskOpacity();
        
        // Smart masking is now MANUAL - user must click "Detect" button
        // No auto-trigger on brush stroke complete
      },
      // onBoxComplete callback
      (bbox: [number, number, number, number]) => {
        // Handle normal box drawing (smart masking is now manual)
        const canvas = drawingHook.maskCanvasRef.current;
        const ctx = drawingHook.ctxRef.current;
        
        // CRITICAL: Save state before box is drawn (for smart mask detection)
        // This will be used to restore canvas before merging smart mask
        if (canvas && ctx && drawingHook.initialDrawState.current?.imageData) {
          // Save state before box
          stateBeforeBoxRef.current = {
            imageData: drawingHook.initialDrawState.current.imageData,
            hasContent: drawingHook.initialDrawState.current.hasContent,
          };
          
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.putImageData(drawingHook.initialDrawState.current.imageData, 0, 0);
        }

        const [xMin, yMin, xMax, yMax] = bbox;
        if (ctx) {
          ctx.globalCompositeOperation = "source-over";
          ctx.globalAlpha = 1.0;
          ctx.fillStyle = "rgba(255, 0, 0, 0.3)"; // Opacity 0.3 (mờ hơn)
          ctx.strokeStyle = "rgba(255, 0, 0, 0.5)"; // Viền opacity 0.5
          ctx.lineWidth = 2;
          const width = xMax - xMin;
          const height = yMax - yMin;
          ctx.fillRect(xMin, yMin, width, height);
          ctx.strokeRect(xMin, yMin, width, height);
        }

        // Normalize opacity after drawing box to ensure consistent visual appearance
        drawingHook.normalizeMaskOpacity();

        // Save to command history
        if (canvas && drawingHook.initialDrawState.current) {
          const currentState = drawingHook.captureCanvasState();
          historyHook.executeDrawCommand(
            drawingHook.initialDrawState.current,
            currentState
          );
          drawingHook.initialDrawState.current = null;
        }
        
        // Smart masking is now MANUAL - user must click "Detect" button
        // Store bbox for later use when user clicks Detect
      }
    );
  }, [
    drawingHook,
    historyHook,
  ]);

  // Global mouse listeners để giữ stroke khi chuột rời khỏi canvas
  // và vẫn đảm bảo stopDrawing của useMasking được gọi (có smart masking + history).
  useEffect(() => {
    if (!drawingHook.isMaskDrawing || !isMaskingMode) {
      return;
    }

    const handleGlobalMove = (event: MouseEvent) => {
      if (drawingHook.isMaskDrawing && isMaskingMode) {
        drawingHook.draw(event);
      }
    };

    const handleGlobalUp = () => {
      if (drawingHook.isMaskDrawing && isMaskingMode) {
        stopDrawing();
      }
    };

    document.addEventListener("mousemove", handleGlobalMove, { passive: false });
    document.addEventListener("mouseup", handleGlobalUp, { passive: false });

    return () => {
      document.removeEventListener("mousemove", handleGlobalMove);
      document.removeEventListener("mouseup", handleGlobalUp);
    };
  }, [drawingHook.isMaskDrawing, isMaskingMode, drawingHook.draw, stopDrawing, drawingHook]);

  // Update stroke tracking for smart masking in startDrawing
  const handleStartDrawing = useCallback(
    (e: React.MouseEvent) => {
      drawingHook.startDrawing(e);

      // Track stroke points for smart masking
      if (
        smartMaskingHook.enableSmartMasking &&
        maskToolType === "brush"
      ) {
        const { x, y } = drawingHook.getCanvasCoordinates(e);
        smartMaskingHook.strokePointsRef.current = [{ x, y }];
        smartMaskingHook.strokeStartTimeRef.current = Date.now();
        smartMaskingHook.strokeStartPosRef.current = { x, y };
        smartMaskingHook.strokeMovementRef.current = 0;
      }
    },
    [drawingHook, smartMaskingHook, maskToolType]
  );

  // Update stroke tracking in draw
  const handleDraw = useCallback(
    (e: React.MouseEvent) => {
      drawingHook.draw(e);

      // Track stroke points and movement for smart masking
      if (
        smartMaskingHook.enableSmartMasking &&
        maskToolType === "brush" &&
        drawingHook.isMaskDrawing
      ) {
        const { x, y } = drawingHook.getCanvasCoordinates(e);
        smartMaskingHook.strokePointsRef.current.push({ x, y });

        if (smartMaskingHook.strokeStartPosRef.current) {
          const dx = x - smartMaskingHook.strokeStartPosRef.current.x;
          const dy = y - smartMaskingHook.strokeStartPosRef.current.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          smartMaskingHook.strokeMovementRef.current = Math.max(
            smartMaskingHook.strokeMovementRef.current,
            distance
          );
        }
    }
    },
    [drawingHook, smartMaskingHook, maskToolType]
  );

  // Clear mask
  const clearMask = useCallback(() => {
    const canvas = drawingHook.maskCanvasRef.current;
    if (!canvas) {
      return;
    }

    historyHook.ensureBaseState();

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    const previousState = drawingHook.captureCanvasState();
    if (!previousState.hasContent) {
      return;
    }

    // Create an empty ImageData as the new state and record it
    const emptyImageData = ctx.createImageData(canvas.width, canvas.height);
    ctx.putImageData(emptyImageData, 0, 0);
    historyHook.executeDrawCommand(previousState, {
      imageData: emptyImageData,
      hasContent: false,
    });
    drawingHook.setHasCanvasContent(false);
  }, [drawingHook, historyHook]);

  // Toggle masking mode
  const toggleMaskingMode = useCallback(() => {
    const wasInMaskingMode = isMaskingMode;
    setIsMaskingMode(!isMaskingMode);

    if (wasInMaskingMode) {
      drawingHook.setIsMaskDrawing(false);
    // Xóa toàn bộ mask cũ để lần bật lại là canvas trống
    historyHook.clearMask();
    historyHook.resetMaskHistory();
    drawingHook.setHasCanvasContent(false);
    setIsMaskVisible(true);

    const maskCanvas = drawingHook.maskCanvasRef.current;
    if (maskCanvas) {
      const ctx = maskCanvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        }
      }

    // Dọn edge overlay nếu có
    const edgeCanvas = edgeOverlayCanvasRef.current;
    if (edgeCanvas) {
      const edgeCtx = edgeCanvas.getContext("2d", { willReadFrequently: true });
      if (edgeCtx) {
        edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
      }
    }

    // Force ctx re-init and clear draw state
    drawingHook.ctxRef.current = null;
    drawingHook.initialDrawState.current = null;
    drawingHook.boxStartPosRef.current = null;
    drawingHook.boxCurrentPosRef.current = null;

    // Cancel smart masking timers/queues if any
    smartMaskingHook.strokePointsRef.current = [];
    smartMaskingHook.strokeMovementRef.current = 0;
    if (smartMaskingHook.smartMaskDebounceTimerRef.current) {
      clearTimeout(smartMaskingHook.smartMaskDebounceTimerRef.current);
      smartMaskingHook.smartMaskDebounceTimerRef.current = null;
    }
    smartMaskingHook.cancelSmartMask();
    } else {
      drawingHook.setIsMaskDrawing(false);
      historyHook.ensureBaseState();
    }
}, [isMaskingMode, drawingHook, historyHook, smartMaskingHook]);

  // Undo mask
  const undoMask = useCallback(() => {
    // If hidden, auto-show so user sees effect
    if (!isMaskVisible) {
      setIsMaskVisible(true);
    }
    historyHook.undoMask();
  }, [historyHook, isMaskVisible]);

  // Redo mask
  const redoMask = useCallback(() => {
    if (!isMaskVisible) {
      setIsMaskVisible(true);
    }
    historyHook.redoMask();
  }, [historyHook, isMaskVisible]);

  return {
    isMaskingMode,
    isMaskVisible,
    isMaskDrawing: drawingHook.isMaskDrawing,
    maskBrushSize,
    maskToolType,
    maskCanvasRef: drawingHook.maskCanvasRef,
    edgeOverlayCanvasRef,
    setMaskBrushSize,
    setMaskToolType,
    handleMaskMouseDown: handleStartDrawing,
    handleMaskMouseMove: handleDraw,
    handleMaskMouseUp: stopDrawing,
    clearMask,
    setIsMaskVisible,
    resetMaskHistory,
    toggleMaskingMode,
    // Mask history with Command Pattern
    maskHistoryIndex: historyHook.maskHistoryIndex,
    maskHistoryLength: historyHook.maskHistoryLength,
    undoMask,
    redoMask,
    hasMaskContent:
      historyHook.historyState.hasContent || drawingHook.hasCanvasContent,
    canUndo: historyHook.canUndo,
    canRedo: historyHook.canRedo,
    // Export final mask (Black/White binary mask for AI)
    getFinalMask: drawingHook.getFinalMask,
    // Save and restore mask canvas state
    saveMaskCanvasState: drawingHook.saveMaskCanvasState,
    restoreMaskCanvasState: useCallback(
      async (savedState: string | null): Promise<boolean> => {
        const result = await drawingHook.restoreMaskCanvasState(savedState);
        if (result) {
          // Sync history với canvas đã restore sau khi img.onload hoàn thành
          // Đợi đủ thời gian để canvas render xong (10ms từ restoreMaskCanvasState)
          await new Promise(resolve => setTimeout(resolve, 10)); // Thêm buffer
          // Sync history với canvas state hiện tại (sau khi restore)
          historyHook.syncHistoryWithCanvas();
        }
        return result;
      },
      [drawingHook, historyHook]
    ),
    // Edge detection and flood fill
    enableEdgeDetection,
    enableFloodFill,
    setEnableEdgeDetection,
    setEnableFloodFill,
    // Smart masking
    enableSmartMasking: smartMaskingHook.enableSmartMasking,
    setEnableSmartMasking: smartMaskingHook.setEnableSmartMasking,
    smartMaskModelType: smartMaskingHook.smartMaskModelType,
    setSmartMaskModelType: smartMaskingHook.setSmartMaskModelType,
    isSmartMaskLoading: smartMaskingHook.isSmartMaskLoading,
    generateSmartMaskFromBox: smartMaskingHook.generateSmartMaskFromBox,
    // Cancel smart mask generation
    cancelSmartMask: smartMaskingHook.cancelSmartMask,
  };
}
