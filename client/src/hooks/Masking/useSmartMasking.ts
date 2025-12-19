import { useState, useRef, useCallback } from "react";
import { apiService } from "../../services/api";
import type { CanvasState } from "./useMaskDrawing";

export interface UseSmartMaskingParams {
  uploadedImage: string | null;
  imageDimensions: { width: number; height: number } | null;
  imageRef?: React.RefObject<HTMLImageElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  ctxRef: React.MutableRefObject<CanvasRenderingContext2D | null>;
  borderAdjustment: number;
  initialDrawState: React.MutableRefObject<CanvasState | null>;
  stateBeforeBoxRef: React.MutableRefObject<CanvasState | null>;
  captureCanvasState: () => CanvasState;
  executeDrawCommand: (previousState: CanvasState, currentState: CanvasState) => void;
  onNotification?: (type: "success" | "error" | "info", message: string) => void;
}

export interface UseSmartMaskingReturn {
  enableSmartMasking: boolean;
  setEnableSmartMasking: (enabled: boolean) => void;
  smartMaskModelType: "segmentation" | "birefnet";
  setSmartMaskModelType: (type: "segmentation" | "birefnet") => void;
  isSmartMaskLoading: boolean;
  strokePointsRef: React.MutableRefObject<Array<{ x: number; y: number }>>;
  strokeStartTimeRef: React.MutableRefObject<number>;
  strokeStartPosRef: React.MutableRefObject<{ x: number; y: number } | null>;
  strokeMovementRef: React.MutableRefObject<number>;
  smartMaskDebounceTimerRef: React.MutableRefObject<NodeJS.Timeout | null>;
  generateSmartMaskFromBox: (bbox: [number, number, number, number]) => Promise<void>;
  generateSmartMaskFromStroke: () => Promise<void>;
  cancelSmartMask: () => Promise<void>;
  clearSmartMaskState: () => void;
}

/**
 * Hook to manage smart masking functionality (FastSAM/BiRefNet)
 * Handles AI-assisted mask generation from strokes and boxes
 */
export function useSmartMasking(
  params: UseSmartMaskingParams
): UseSmartMaskingReturn {
  const {
    uploadedImage,
    imageDimensions,
    imageRef,
    maskCanvasRef,
    ctxRef,
    borderAdjustment,
    initialDrawState,
    stateBeforeBoxRef,
    captureCanvasState,
    executeDrawCommand,
    onNotification,
  } = params;

  const [enableSmartMasking, setEnableSmartMasking] = useState(false);
  const [smartMaskModelType, setSmartMaskModelType] = useState<
    "segmentation" | "birefnet"
  >("segmentation");
  // Note: smartMaskImageId is kept for future use but currently not used
  const [, setSmartMaskImageId] = useState<string | null>(null);
  const [isSmartMaskLoading, setIsSmartMaskLoading] = useState(false);
  const [currentSmartMaskRequestId, setCurrentSmartMaskRequestId] =
    useState<string | null>(null);

  // Store stroke points for smart masking
  const strokePointsRef = useRef<Array<{ x: number; y: number }>>([]);
  const smartMaskDebounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Track click-only mode detection
  const strokeStartTimeRef = useRef<number>(0);
  const strokeStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const strokeMovementRef = useRef<number>(0); // Total movement distance in pixels

  // Helper: Calculate bounding box from points
  const calculateBoundingBox = useCallback(
    (
      points: Array<{ x: number; y: number }>
    ): [number, number, number, number] | null => {
      if (points.length === 0) return null;

      let xMin = points[0].x;
      let yMin = points[0].y;
      let xMax = points[0].x;
      let yMax = points[0].y;

      for (const point of points) {
        xMin = Math.min(xMin, point.x);
        yMin = Math.min(yMin, point.y);
        xMax = Math.max(xMax, point.x);
        yMax = Math.max(yMax, point.y);
      }

      return [xMin, yMin, xMax, yMax];
    },
    []
  );

  // Helper: Sample points from stroke (for bbox mode)
  const samplePointsFromStroke = useCallback(
    (
      points: Array<{ x: number; y: number }>,
      maxPoints: number
    ): Array<[number, number]> => {
      if (points.length === 0) return [];
      if (points.length <= maxPoints) {
        return points.map((p) => [p.x, p.y]);
      }

      // Sample evenly distributed points
      const step = Math.floor(points.length / maxPoints);
      const sampled: Array<[number, number]> = [];

      for (let i = 0; i < points.length; i += step) {
        sampled.push([points[i].x, points[i].y]);
        if (sampled.length >= maxPoints) break;
      }

      return sampled;
    },
    []
  );

  // Helper: Scale coordinates from canvas to original image dimensions
  const scaleCoordinatesToOriginal = useCallback(
    (coords: {
      bbox?: [number, number, number, number];
      points?: Array<[number, number]>;
    }): {
      bbox?: [number, number, number, number];
      points?: Array<[number, number]>;
    } => {
      const canvas = maskCanvasRef.current;
      const img = imageRef?.current;
      if (!canvas || !img || !imageDimensions) {
        return {};
      }

      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      const naturalWidth = img.naturalWidth;
      const naturalHeight = img.naturalHeight;

      // Scale factor: canvas size -> natural image size
      const scaleX = naturalWidth / canvasWidth;
      const scaleY = naturalHeight / canvasHeight;

      // Helper to scale and clamp coordinates
      const scaleCoord = (x: number, y: number): [number, number] => {
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;

        // Clamp to image bounds
        const clampedX = Math.max(0, Math.min(scaledX, naturalWidth - 1));
        const clampedY = Math.max(0, Math.min(scaledY, naturalHeight - 1));

        // Log warning if coordinates were clamped
        if (scaledX !== clampedX || scaledY !== clampedY) {
          console.warn("Coordinates clamped to image bounds:", {
            original: [x, y],
            scaled: [scaledX, scaledY],
            clamped: [clampedX, clampedY],
            naturalSize: [naturalWidth, naturalHeight],
          });
        }

        return [clampedX, clampedY];
      };

      const result: {
        bbox?: [number, number, number, number];
        points?: Array<[number, number]>;
      } = {};

      if (coords.bbox) {
        const [xMin, yMin, xMax, yMax] = coords.bbox;
        const [scaledXMin, scaledYMin] = scaleCoord(xMin, yMin);
        const [scaledXMax, scaledYMax] = scaleCoord(xMax, yMax);

        // Ensure bbox is valid (min < max)
        const finalXMin = Math.min(scaledXMin, scaledXMax);
        const finalYMin = Math.min(scaledYMin, scaledYMax);
        const finalXMax = Math.max(scaledXMin, scaledXMax);
        const finalYMax = Math.max(scaledYMin, scaledYMax);

        result.bbox = [finalXMin, finalYMin, finalXMax, finalYMax];
      }

      if (coords.points) {
        result.points = coords.points.map(([x, y]) => scaleCoord(x, y));
      }

      return result;
    },
    [maskCanvasRef, imageRef, imageDimensions]
  );

  // Helper: Convert binary mask (white/black) to red transparent mask
  const convertBinaryMaskToRedTransparent = useCallback(
    (
      binaryMaskImage: HTMLImageElement,
      width: number,
      height: number
    ): ImageData => {
      // Create temporary canvas to process the mask
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
      if (!tempCtx) throw new Error("Failed to get canvas context");

      // Draw the binary mask onto temp canvas
      tempCtx.drawImage(binaryMaskImage, 0, 0, width, height);

      // Get image data
      const imageData = tempCtx.getImageData(0, 0, width, height);
      const data = imageData.data;

      // Convert binary mask to red transparent
      // White pixels (255, 255, 255) → rgba(255, 0, 0, 0.5)
      // Black pixels (0, 0, 0) → transparent (rgba(0, 0, 0, 0))
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];

        // Convert grayscale to determine if pixel is white or black
        const grayscale = (r + g + b) / 3;

        if (grayscale > 128 && a > 0) {
          // White pixel (mask area) → red transparent
          data[i] = 255; // R
          data[i + 1] = 0; // G
          data[i + 2] = 0; // B
          data[i + 3] = 76; // A (0.3 opacity = 76/255, mờ hơn)
        } else {
          // Black pixel or transparent → fully transparent
          data[i] = 0; // R
          data[i + 1] = 0; // G
          data[i + 2] = 0; // B
          data[i + 3] = 0; // A (fully transparent)
        }
      }

      return imageData;
    },
    []
  );

  // Function to merge smart mask into canvas
  const mergeSmartMask = useCallback(
    async (maskBase64: string) => {
      const canvas = maskCanvasRef.current;
      const ctx = ctxRef.current;
      if (!canvas || !ctx) return;

      try {
        // Create image from base64 mask
        const img = new Image();
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = reject;
          img.src = `data:image/png;base64,${maskBase64}`;
        });

        // Get current composite operation (preserve user's mode)
        const currentComposite = ctx.globalCompositeOperation;

        // Convert binary mask to red transparent mask
        const redTransparentMask = convertBinaryMaskToRedTransparent(
          img,
          canvas.width,
          canvas.height
        );

        // Create temporary canvas to convert ImageData to Image
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });
        if (!tempCtx) throw new Error("Failed to get temp canvas context");

        // Put the converted mask data onto temp canvas
        tempCtx.putImageData(redTransparentMask, 0, 0);

        // Use source-over to add mask (giống như stroke)
        ctx.globalCompositeOperation = "source-over";

        // Draw converted mask onto canvas using drawImage
        ctx.drawImage(tempCanvas, 0, 0);

        // Restore original composite operation
        ctx.globalCompositeOperation = currentComposite;
      } catch (error) {
        console.error("Failed to merge smart mask:", error);
        throw error;
      }
    },
    [maskCanvasRef, ctxRef, convertBinaryMaskToRedTransparent]
  );

  // Function to generate smart mask from box
  const generateSmartMaskFromBox = useCallback(
    async (bbox: [number, number, number, number]) => {
      if (
        !enableSmartMasking ||
        !uploadedImage ||
        !imageDimensions ||
        !imageRef?.current
      )
        return;

      setIsSmartMaskLoading(true);

      // Save current mask state BEFORE attempting detection
      // This will be used to restore if detection fails
      const savedMaskState = captureCanvasState();

      try {
        // Scale bbox to original image dimensions
        const scaledCoords = scaleCoordinatesToOriginal({ bbox });

        if (!scaledCoords.bbox) {
          console.error("Failed to scale bbox coordinates");
          return;
        }

        // Call API with bbox
        const result = await apiService.generateSmartMask(
          uploadedImage, // Always send image for independent segmentation
          null, // Don't use image_id - each mask is independent
          scaledCoords.bbox,
          undefined, // No points for box mode
          borderAdjustment,
          false, // use_blur
          smartMaskModelType
        );

        if (result.success && result.mask_base64) {
          const canvas = maskCanvasRef.current;
          const ctx = ctxRef.current;

          // Track request_id for cancellation
          if (result.request_id) {
            setCurrentSmartMaskRequestId(result.request_id);
          }

          // CRITICAL: Restore to state before box was drawn, then merge smart mask
          // This removes the box and replaces it with the detected mask
          if (canvas && ctx && stateBeforeBoxRef.current?.imageData) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.putImageData(stateBeforeBoxRef.current.imageData, 0, 0);
          }

          // Merge mask into canvas (on top of existing strokes, but replacing box)
          await mergeSmartMask(result.mask_base64);

          // Clear initialDrawState and stateBeforeBoxRef (no history integration anymore)
          if (initialDrawState.current) {
            initialDrawState.current = null;
          }
          if (stateBeforeBoxRef.current) {
            stateBeforeBoxRef.current = null;
          }

          // Show success notification
          if (onNotification) {
            onNotification("success", "Smart mask generated successfully");
          }
        } else {
          const errorMsg = result.error || "Smart mask generation failed";
          console.error("Smart mask generation failed:", errorMsg);
          if (onNotification) {
            onNotification("error", errorMsg);
          }
        }
      } catch (error: unknown) {
        console.error("Error generating smart mask from box:", error);
        let errorMessage = "Failed to generate smart mask";

        // Parse error message
        let parsedError: { error?: string } | null = null;
        const errorObj = error as { message?: string };
        if (errorObj?.message) {
          try {
            parsedError = JSON.parse(errorObj.message) as { error?: string };
          } catch {
            parsedError = { error: errorObj.message };
          }
        }

        const errorText = parsedError?.error || errorObj?.message || "";

        // If image_id expired, clear it
        if (errorText.includes("Image ID not found") || errorText.includes("expired")) {
          console.log("🔄 Image ID expired, clearing cached image_id");
          setSmartMaskImageId(null);
        }

        // Check if detection failed (no masks found)
        const isDetectionFailed = 
          errorText.includes("No masks found") ||
          errorText.includes("did not generate") ||
          errorText.includes("not found") ||
          errorText.includes("404");

        // CRITICAL: Restore mask to state before detection attempt
        // This preserves the user's drawn stroke even if detection fails
        const canvas = maskCanvasRef.current;
        const ctx = ctxRef.current;
        if (canvas && ctx && savedMaskState?.imageData) {
          console.log("🔄 Restoring mask to state before detection (keeping original stroke)");
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.putImageData(savedMaskState.imageData, 0, 0);
        }

        if (isDetectionFailed) {
          // Detection failed - keep original stroke
          if (onNotification) {
            onNotification(
              "warning",
              "Could not detect object. Keeping your original stroke."
            );
          }
          return;
        }

        // Provide user-friendly error messages for other errors
        if (errorText.includes("Network") || errorText.includes("fetch")) {
          errorMessage = "Network error. Keeping your original stroke.";
        } else if (errorText.includes("timeout")) {
          errorMessage = "Request timed out. Keeping your original stroke.";
        } else if (parsedError?.error) {
          errorMessage = parsedError.error + ". Keeping your original stroke.";
        } else if (errorObj?.message) {
          errorMessage = errorObj.message + ". Keeping your original stroke.";
        }

        if (onNotification) {
          onNotification("warning", errorMessage);
        }
      } finally {
        setIsSmartMaskLoading(false);
      }
    },
    [
      enableSmartMasking,
      smartMaskModelType,
      uploadedImage,
      imageDimensions,
      imageRef,
      scaleCoordinatesToOriginal,
      mergeSmartMask,
      captureCanvasState,
      executeDrawCommand,
      onNotification,
      borderAdjustment,
      initialDrawState,
      stateBeforeBoxRef,
      ctxRef,
      maskCanvasRef,
    ]
  );

  // Function to generate smart mask from stroke
  const generateSmartMaskFromStroke = useCallback(async () => {
    if (
      !enableSmartMasking ||
      !uploadedImage ||
      !imageDimensions ||
      !imageRef?.current
    ) {
      return;
    }

    const points = strokePointsRef.current;
    if (points.length === 0) return;

    setIsSmartMaskLoading(true);

    try {
      // For single point (click-only), use point mode directly
      // For multiple points (stroke), calculate bbox and sample points
      const isSinglePoint = points.length === 1;
      const bbox = isSinglePoint ? null : calculateBoundingBox(points);
      const sampledPoints: Array<[number, number]> = isSinglePoint
        ? [[points[0].x, points[0].y]]
        : samplePointsFromStroke(points, 10);

      // Determine whether to use bbox or points mode
      let usePoints = isSinglePoint;
      let scaledCoords: {
        bbox?: [number, number, number, number];
        points?: Array<[number, number]>;
      } = {};

      if (!isSinglePoint && bbox) {
        const [xMin, yMin, xMax, yMax] = bbox;
        const bboxWidth = xMax - xMin;
        const bboxHeight = yMax - yMin;

        // If bbox is too small (< 10x10px), use point mode instead
        if (bboxWidth < 10 || bboxHeight < 10) {
          usePoints = true;
        } else {
          // Use bbox mode - scale coordinates to original image dimensions
          scaledCoords = scaleCoordinatesToOriginal({ bbox, points: sampledPoints });
        }
      }

      if (usePoints && sampledPoints.length > 0) {
        // Use point mode - scale points to original image dimensions
        scaledCoords = scaleCoordinatesToOriginal({ points: sampledPoints });
      }

      // Call API
      const result = await apiService.generateSmartMask(
        uploadedImage, // Always send image for independent segmentation
        null, // Don't use image_id - each mask is independent
        scaledCoords.bbox,
        scaledCoords.points,
        borderAdjustment,
        false, // use_blur
        smartMaskModelType
      );

      if (result.success && result.mask_base64) {
        // Track request_id for cancellation
        if (result.request_id) {
          setCurrentSmartMaskRequestId(result.request_id);
        }

        const canvas = maskCanvasRef.current;

        // Merge mask into canvas on top of existing strokes
        await mergeSmartMask(result.mask_base64);

        // Clear initialDrawState (no history integration anymore)
        if (canvas && initialDrawState.current) {
          initialDrawState.current = null;
        }

        // Show success notification
        if (onNotification) {
          onNotification("success", "Smart mask generated successfully");
        }
      } else {
        const errorMsg = result.error || "Smart mask generation failed";
        console.error("Smart mask generation failed:", errorMsg);
        if (onNotification) {
          onNotification("error", errorMsg);
        }
      }
    } catch (error: unknown) {
      console.error("Error generating smart mask:", error);
      let errorMessage = "Failed to generate smart mask";

      // Parse error message
      let parsedError: { error?: string } | null = null;
      const errorObj = error as { message?: string };
      if (errorObj?.message) {
        try {
          parsedError = JSON.parse(errorObj.message) as { error?: string };
        } catch {
          parsedError = { error: errorObj.message };
        }
      }

      const errorText = parsedError?.error || errorObj?.message || "";

      // If image_id expired, clear it
      if (errorText.includes("Image ID not found") || errorText.includes("expired")) {
        console.log("🔄 Image ID expired, clearing cached image_id");
        setSmartMaskImageId(null);
      }

      if (
        errorText.includes("No masks found") ||
        errorText.includes("did not generate")
      ) {
        // Normal case, show info message
        if (onNotification) {
          onNotification(
            "info",
            "FastSAM không detect được gì với vị trí mask, hãy chọn vị trí khác hay vật thể khác"
          );
        }
        return;
      }

      // Provide user-friendly error messages
      if (errorText.includes("Network") || errorText.includes("fetch")) {
        errorMessage = "Network error. Please check your connection and try again.";
      } else if (errorText.includes("timeout")) {
        errorMessage = "Request timed out. Please try again.";
      } else if (parsedError?.error) {
        errorMessage = parsedError.error;
      } else if (errorObj?.message) {
        errorMessage = errorObj.message;
      }

      if (onNotification) {
        onNotification("error", errorMessage);
      }
    } finally {
      setIsSmartMaskLoading(false);
      strokePointsRef.current = [];
    }
  }, [
    enableSmartMasking,
    smartMaskModelType,
    uploadedImage,
    imageDimensions,
    imageRef,
    calculateBoundingBox,
    samplePointsFromStroke,
    scaleCoordinatesToOriginal,
    mergeSmartMask,
    captureCanvasState,
    executeDrawCommand,
    onNotification,
    borderAdjustment,
    initialDrawState,
    ctxRef,
    maskCanvasRef,
  ]);

  // Cancel smart mask request
  const cancelSmartMask = useCallback(async () => {
    if (currentSmartMaskRequestId) {
      try {
        await apiService.cancelSmartMask(currentSmartMaskRequestId);
        console.log(`✅ Smart mask request ${currentSmartMaskRequestId} cancelled`);
        setCurrentSmartMaskRequestId(null);
        setIsSmartMaskLoading(false);
      } catch (error) {
        console.error("Failed to cancel smart mask:", error);
      }
    }
  }, [currentSmartMaskRequestId]);

  // Clear smart mask state
  const clearSmartMaskState = useCallback(() => {
    setSmartMaskImageId(null);
    strokePointsRef.current = [];
    if (smartMaskDebounceTimerRef.current) {
      clearTimeout(smartMaskDebounceTimerRef.current);
      smartMaskDebounceTimerRef.current = null;
    }
  }, []);

  return {
    enableSmartMasking,
    setEnableSmartMasking,
    smartMaskModelType,
    setSmartMaskModelType,
    isSmartMaskLoading,
    strokePointsRef,
    strokeStartTimeRef,
    strokeStartPosRef,
    strokeMovementRef,
    smartMaskDebounceTimerRef,
    generateSmartMaskFromBox,
    generateSmartMaskFromStroke,
    cancelSmartMask,
    clearSmartMaskState,
  };
}

