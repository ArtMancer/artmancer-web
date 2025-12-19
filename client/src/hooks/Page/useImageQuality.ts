import { useState, useRef, useCallback } from "react";
import type { InputQualityPreset } from "@/services/api";

/**
 * Helper function to scale an image data URL to target dimensions
 * @param dataUrl - Source image data URL
 * @param targetWidth - Target width
 * @param targetHeight - Target height
 * @param preserveAspectRatio - Whether to preserve aspect ratio (pad with black)
 * @returns Promise with scaled image data URL and dimensions
 */
const scaleImageDataUrl = (
  dataUrl: string,
  targetWidth: number,
  targetHeight: number,
  preserveAspectRatio: boolean = false
): Promise<{ dataUrl: string; width: number; height: number }> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        reject(new Error("Cannot get canvas context"));
        return;
      }
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";

      if (preserveAspectRatio) {
        // Fill canvas with black background
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, targetWidth, targetHeight);

        // Calculate scale to fit image within target size (preserve aspect ratio)
        const scale = Math.min(
          targetWidth / img.width,
          targetHeight / img.height
        );
        const newWidth = img.width * scale;
        const newHeight = img.height * scale;

        // Center the image
        const x = (targetWidth - newWidth) / 2;
        const y = (targetHeight - newHeight) / 2;

        ctx.drawImage(img, x, y, newWidth, newHeight);
      } else {
        // Force resize (may distort)
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
      }

      resolve({
        dataUrl: canvas.toDataURL("image/png"),
        width: targetWidth,
        height: targetHeight,
      });
    };
    img.onerror = (err) => reject(err);
    img.src = dataUrl;
  });
};

export interface UseImageQualityParams {
  // Source image data
  baseImageData: string | null;
  baseImageDimensions: { width: number; height: number } | null;
  originalUploadData: string | null;
  originalUploadDimensions: { width: number; height: number } | null;
  
  // Image setters
  setUploadedImage: (image: string | null) => void;
  setModifiedImage: (image: string | null) => void;
  setImageDimensions: (dims: { width: number; height: number } | null) => void;
  
  // Comparison state setters
  setOriginalImage: (image: string | null) => void;
  setModifiedImageForComparison: (image: string | null) => void;
  setComparisonSlider: (value: number) => void;
  
  // Mask state setters
  setSavedMaskData: (data: string | null) => void;
  setSavedMaskCanvasState: (state: string | null) => void;
  resetMaskHistory: () => void;
  clearMask: () => void;
  
  // History and request ID setters
  initializeHistory: (image: string) => void;
  setLastRequestId: (id: string | null) => void;
  
  // Notification
  setNotificationWithTimeout: (
    type: "success" | "error" | "info",
    message: string,
    timeoutMs?: number
  ) => void;
}

export interface UseImageQualityReturn {
  inputQuality: InputQualityPreset;
  setInputQuality: (quality: InputQualityPreset) => void;
  customSquareSize: number;
  setCustomSquareSize: (size: number) => void;
  isApplyingQuality: boolean;
  applyInputQualityPreset: (
    quality: InputQualityPreset,
    options?: {
      sourceData?: string;
      sourceDimensions?: { width: number; height: number };
      silent?: boolean;
      squareSize?: number;
    }
  ) => Promise<boolean>;
}

/**
 * Hook to manage image quality settings and apply quality presets
 * Handles resizing images to square format or keeping original dimensions
 */
export function useImageQuality(
  params: UseImageQualityParams
): UseImageQualityReturn {
  const {
    baseImageData,
    baseImageDimensions,
    originalUploadData,
    originalUploadDimensions,
    setUploadedImage,
    setModifiedImage,
    setImageDimensions,
    setOriginalImage,
    setModifiedImageForComparison,
    setComparisonSlider,
    setSavedMaskData,
    setSavedMaskCanvasState,
    resetMaskHistory,
    clearMask,
    initializeHistory,
    setLastRequestId,
    setNotificationWithTimeout,
  } = params;

  const [inputQuality, setInputQuality] = useState<InputQualityPreset>("resized");
  const [customSquareSize, setCustomSquareSize] = useState<number>(1024);
  const [isApplyingQuality, setIsApplyingQuality] = useState(false);
  const qualityChangeRequestRef = useRef(0);

  const applyInputQualityPreset = useCallback(
    async (
      quality: InputQualityPreset,
      options?: {
        sourceData?: string;
        sourceDimensions?: { width: number; height: number };
        silent?: boolean;
        squareSize?: number;
      }
    ): Promise<boolean> => {
      // For "original" quality, always use the original upload data (never resized)
      // For "resized" quality, use current base image or source data
      const useOriginalUpload =
        quality === "original" &&
        originalUploadData &&
        originalUploadDimensions;
      const sourceData = useOriginalUpload
        ? originalUploadData
        : options?.sourceData ?? baseImageData;
      const sourceDimensions = useOriginalUpload
        ? originalUploadDimensions
        : options?.sourceDimensions ?? baseImageDimensions;

      if (!sourceData || !sourceDimensions) {
        return false;
      }

      const requestId = ++qualityChangeRequestRef.current;
      setIsApplyingQuality(true);

      try {
        let targetWidth = sourceDimensions.width;
        let targetHeight = sourceDimensions.height;
        let resultDataUrl = sourceData;

        // If quality is "resized", resize to 1:1 aspect ratio (square) with padding
        if (quality === "resized") {
          // Use custom size if provided, otherwise use state value
          const squareSize = options?.squareSize ?? customSquareSize;

          targetWidth = squareSize;
          targetHeight = squareSize;

          const scaled = await scaleImageDataUrl(
            sourceData,
            targetWidth,
            targetHeight,
            true // preserveAspectRatio = true (pad with black)
          );
          resultDataUrl = scaled.dataUrl;
          targetWidth = scaled.width;
          targetHeight = scaled.height;
        }
        // If quality is "original", use the original upload dimensions (already set above)

        if (qualityChangeRequestRef.current !== requestId) {
          return false;
        }

        setUploadedImage(resultDataUrl);
        setModifiedImage(resultDataUrl);
        setImageDimensions({ width: targetWidth, height: targetHeight });
        setOriginalImage(resultDataUrl);
        setModifiedImageForComparison(null);
        setComparisonSlider(50);
        initializeHistory(resultDataUrl);
        setSavedMaskData(null); // Clear saved mask when applying quality preset
        setSavedMaskCanvasState(null); // Clear saved mask canvas state
        resetMaskHistory();
        clearMask();
        setLastRequestId(null);

        if (!options?.silent) {
          console.log("✅ Applied input quality preset:", {
            quality,
            mode: quality === "resized" ? "1:1 square" : "original",
            width: targetWidth,
            height: targetHeight,
          });
        }

        return true;
      } catch (error) {
        console.error("Failed to apply input quality:", error);
        if (!options?.silent) {
          setNotificationWithTimeout(
            "error",
            "Không thể thay đổi chất lượng ảnh. Vui lòng thử lại."
          );
        }
        return false;
      } finally {
        if (qualityChangeRequestRef.current === requestId) {
          setIsApplyingQuality(false);
        }
      }
    },
    [
      customSquareSize,
      baseImageData,
      baseImageDimensions,
      originalUploadData,
      originalUploadDimensions,
      clearMask,
      initializeHistory,
      resetMaskHistory,
      setImageDimensions,
      setModifiedImage,
      setModifiedImageForComparison,
      setOriginalImage,
      setComparisonSlider,
      setLastRequestId,
      setNotificationWithTimeout,
      setUploadedImage,
      setSavedMaskData,
      setSavedMaskCanvasState,
    ]
  );

  return {
    inputQuality,
    setInputQuality,
    customSquareSize,
    setCustomSquareSize,
    isApplyingQuality,
    applyInputQualityPreset,
  };
}

