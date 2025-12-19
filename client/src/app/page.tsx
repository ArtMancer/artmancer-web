"use client";

import { useState, useRef, useCallback, useEffect, use } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import HelpBox from "@/components/HelpBox";
import Canvas from "@/components/MainCanvas";
import NotificationComponent from "@/components/Notification";
import type { NotificationType } from "@/components/Notification";
import {
  useImageUpload,
  useViewportControls,
  useMasking,
  useImageHistory,
  useImageTransform,
  useImageGeneration,
} from "@/hooks";
import { useNotification } from "@/hooks/Page/useNotification";
import { useImageQuality } from "@/hooks/Page/useImageQuality";
import { useReferenceImage } from "@/hooks/Page/useReferenceImage";
import { useSidebarResize } from "@/hooks/Page/useSidebarResize";
import type { InputQualityPreset, DebugInfo } from "@/services/api";
import DebugPanel from "@/components/DebugPanel";
import ReferenceImageEditor from "@/components/ReferenceImageEditor";

type PageProps = {
  params: Promise<Record<string, string>>;
  searchParams: Promise<Record<string, string>>;
};

export default function Home({ params, searchParams }: PageProps) {
  // Next.js 16: unwrap params/searchParams (even if unused) to avoid sync access warnings
  use(params);
  use(searchParams);
  // Basic UI state
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  // Notification management hook
  const {
    notificationType,
    notificationMessage,
    isNotificationVisible,
    showNotification,
    hideNotification,
    clearNotification,
  } = useNotification();

  // Legacy error/success state - kept for backward compatibility (unused but kept for potential future use)
  const [, setError] = useState<string | null>(null);
  const [, setSuccess] = useState<string | null>(null);

  // AI Task state
  const [aiTask, setAiTask] = useState<
    "white-balance" | "object-insert" | "object-removal"
  >("white-balance");
  
  // Saved mask states - lưu mask khi generation bắt đầu để không bị mất
  // savedMaskData: mask binary (đen trắng) để gửi API - được lưu nhưng không cần restore
  // savedMaskCanvasState: mask canvas UI (đỏ trong suốt) để restore mask brush sau generation
   
  const [, setSavedMaskData] = useState<string | null>(null);
  const [savedMaskCanvasState, setSavedMaskCanvasState] = useState<string | null>(null);
  const [maskVisible, setMaskVisible] = useState(true);
  
  // Track previous uploadedImage to prevent mask visibility reset when generating multiple times
  const prevUploadedImageRef = useRef<string | null>(null);

  // Reference image management hook
  const {
    referenceImage,
    referenceMaskR,
    isRefEditorOpen,
    pendingRefImage,
    handleReferenceImageUpload,
    handleRefEditorSubmit,
    handleRefEditorClose,
    handleRemoveReferenceImage,
    handleEditReferenceImage,
    setReferenceImage,
    setReferenceMaskR,
  } = useReferenceImage();

  // Advanced options state
  const [negativePrompt, setNegativePrompt] = useState<string>("");
  const [seed, setSeed] = useState<number>(42); // Default seed: 42 (famous default)
  const [enableMaeRefinement, setEnableMaeRefinement] = useState<boolean>(true); // Default: enabled
  // Task-specific default parameters
  // All tasks use the same defaults: inferenceSteps=15, guidanceScale=2.0
  const getDefaultParametersForTask = useCallback((task: string) => {
    switch (task) {
      case "object-removal":
        return {
          guidanceScale: 2.0, // Default 2.0 for all tasks
          inferenceSteps: 15, // Default 15 steps for all tasks
        };
      case "object-insert":
        return {
          guidanceScale: 2.0, // Default 2.0 for all tasks
          inferenceSteps: 15, // Default 15 steps for all tasks
        };
      case "white-balance":
        return {
          guidanceScale: 2.0, // Default 2.0 for all tasks
          inferenceSteps: 15, // Default 15 steps for all tasks
        };
      default:
        return {
          guidanceScale: 2.0, // Default 2.0 for all tasks
          inferenceSteps: 15, // Default 15 steps for all tasks
        };
    }
  }, []);

  const [guidanceScale, setGuidanceScale] = useState<number>(2.0); // Default 2.0 (used for both guidance_scale and true_cfg_scale)
  const [inferenceSteps, setInferenceSteps] = useState<number>(15); // Default 15 steps
  const [borderAdjustment, setBorderAdjustment] = useState<number>(0); // Border adjustment for smart masks

  const [baseImageData, setBaseImageData] = useState<string | null>(null);
  const [baseImageDimensions, setBaseImageDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  // Store original upload data separately (not overwritten after generation)
  const [originalUploadData, setOriginalUploadData] = useState<string | null>(
    null
  );
  const [originalUploadDimensions, setOriginalUploadDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);

  // White balance state

  // Visualization request ID state
  const [lastRequestId, setLastRequestId] = useState<string | null>(null);

  // Debug panel state
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [isDebugPanelVisible, setIsDebugPanelVisible] = useState(false); // Default: OFF, enable in settings when needed

  // Auto-refine prompt state

  // Sidebar resize management hook
  const {
    sidebarWidth,
    isResizing,
    handleResizeStart,
    setSidebarWidth,
  } = useSidebarResize(320);

  // Legacy notification wrapper for backward compatibility
  // Maps old setNotificationWithTimeout to new showNotification
  const setNotificationWithTimeout = useCallback(
    (type: NotificationType, message: string, timeoutMs: number = 5000) => {
      // Legacy state for backward compatibility
      if (type === "success") {
        setSuccess(message);
        setError(null);
      } else if (type === "error") {
        setError(message);
        setSuccess(null);
      }

      // Use new notification hook
      showNotification(type, message, timeoutMs);
    },
    [showNotification]
  );

  // Legacy close handler wrapper
  const handleCloseNotification = useCallback(() => {
    hideNotification();
    setSuccess(null);
    setError(null);
  }, [hideNotification]);

  // Legacy clear all wrapper
  const clearAllNotifications = useCallback(() => {
    clearNotification();
    setSuccess(null);
    setError(null);
  }, [clearNotification]);


  // Listen for server notifications from ServerContext
  useEffect(() => {
    const handleServerNotification = (
      event: CustomEvent<{
        type: "success" | "error" | "info";
        message: string;
      }>
    ) => {
      const { type, message } = event.detail;
      setNotificationWithTimeout(type, message);
    };

    window.addEventListener(
      "server-notification",
      handleServerNotification as EventListener
    );
    return () => {
      window.removeEventListener(
        "server-notification",
        handleServerNotification as EventListener
      );
    };
  }, [setNotificationWithTimeout]);

  // Custom hooks
  const {
    uploadedImage,
    imageDimensions,
    displayScale,
    removeImage,
    handleImageClick,
    setUploadedImage,
    setModifiedImage,
    setImageDimensions,
  } = useImageUpload();

  // Generation progress state
  const [generationProgress, setGenerationProgress] = useState<{
    current: number;
    total: number;
    status?: string;
    loadingMessage?: string;
    estimatedTimeRemaining?: number; // in seconds
    progressPercent?: number; // 0.0-1.0 for loading phase
  } | null>(null);

  // Track step timing for ETR calculation
  const stepTimingRef = useRef<{ step: number; timestamp: number }[]>([]);
  // Track if generation has completed to prevent progress updates after completion
  const generationCompletedRef = useRef(false);
  // Track when processing starts to show different messages during initialization
  const processingStartTimeRef = useRef<number | null>(null);

  // API integration
  const {
    generateImage,
    isGenerating,
    error: generationError,
    clearError,
    cancelGeneration,
  } = useImageGeneration();

  const { transform, imageRef, setTransform } = useImageTransform();

  const {
    viewportZoom,
    imageContainerRef,
    containerRef,
    handleWheel,
    zoomViewportIn,
    zoomViewportOut,
    resetViewportZoom,
  } = useViewportControls({
    imageDimensions,
    setTransform,
  });


  const handleAiTaskChange = (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => {
    const previousTask = aiTask;
    setAiTask(task);

    // Debug logging
    console.log("🔍 [Frontend Debug] Task changed:", {
      from: previousTask,
      to: task,
      currentReferenceImage: referenceImage ? "exists" : "null",
    });

    // Clear reference image and mask when switching away from object-insert
    if (task !== "object-insert") {
      console.log(
        "🔍 [Frontend Debug] Clearing reference image and mask (switched from object-insert)"
      );
      setReferenceImage(null);
      setReferenceMaskR(null);
    } else {
      console.log(
        "🔍 [Frontend Debug] Keeping reference image and mask (object-insert task)"
      );
    }
  };

  // Ref to skip mask reset when returning to original
  const skipMaskResetRef = useRef(false);

  const {
    isMaskingMode,
    maskBrushSize,
    maskToolType,
    maskCanvasRef,
    setMaskBrushSize,
    setMaskToolType,
    toggleMaskingMode,
    clearMask,
    resetMaskHistory,
    handleMaskMouseDown,
    handleMaskMouseMove,
    handleMaskMouseUp,
    hasMaskContent,
    getFinalMask,
    saveMaskCanvasState,
    restoreMaskCanvasState,
    enableSmartMasking,
    setEnableSmartMasking,
    smartMaskModelType,
    setSmartMaskModelType,
    isSmartMaskLoading,
    generateSmartMaskFromBox,
    edgeOverlayCanvasRef,
    setIsMaskVisible,
  } = useMasking(
    uploadedImage,
    imageDimensions,
    imageContainerRef,
    transform,
    viewportZoom,
    imageRef,
    setNotificationWithTimeout,
    borderAdjustment,
    skipMaskResetRef
  );

  // Track last uploaded image just for potential future use (no auto mask reset here)
  useEffect(() => {
    if (uploadedImage) {
      prevUploadedImageRef.current = uploadedImage;
    } else {
      prevUploadedImageRef.current = null;
    }
  }, [uploadedImage]);

  const {
    historyIndex,
    historyStack,
    handleUndo,
    handleRedo,
    addToHistory,
    initializeHistory,
  } = useImageHistory();

  const [comparisonSlider, setComparisonSlider] = useState(50);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [modifiedImageForComparison, setModifiedImageForComparison] = useState<
    string | null
  >(null);

  const handleToggleMaskVisible = () => {
    setMaskVisible((prev) => !prev);
    setIsMaskVisible((prev) => !prev);
  };

  // Smart mask detect from current mask
  const handleDetectSmartMask = useCallback(async () => {
    if (!hasMaskContent) {
      setNotificationWithTimeout("warning", "Please draw a mask first");
      return;
    }

    try {
      // Calculate bbox from current mask canvas
      const canvas = maskCanvasRef.current;
      if (!canvas) {
        setNotificationWithTimeout("error", "Mask canvas not found");
        return;
      }

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Find bounding box of drawn mask
      let minX = canvas.width;
      let minY = canvas.height;
      let maxX = 0;
      let maxY = 0;
      let hasPixels = false;

      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const idx = (y * canvas.width + x) * 4;
          const alpha = data[idx + 3];
          if (alpha > 0) {
            hasPixels = true;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }

      if (!hasPixels) {
        setNotificationWithTimeout("warning", "No mask content found");
        return;
      }

      // Call smart mask generation with calculated bbox
      const bbox: [number, number, number, number] = [minX, minY, maxX, maxY];
      await generateSmartMaskFromBox(bbox);
    } catch (error) {
      console.error("Error detecting smart mask:", error);
      setNotificationWithTimeout("error", "Failed to detect smart mask");
    }
  }, [hasMaskContent, maskCanvasRef, generateSmartMaskFromBox, setNotificationWithTimeout]);

  // Auto-unhide mask before undo/redo for better UX
  // Undo/Redo removed per requirement

  const setBaseImage = useCallback(
    (dataUrl: string | null, dims?: { width: number; height: number }) => {
      setBaseImageData(dataUrl);
      if (!dataUrl) {
        setBaseImageDimensions(null);
        return;
      }
      if (dims) {
        setBaseImageDimensions(dims);
        return;
      }
      const img = new Image();
      img.onload = () => {
        setBaseImageDimensions({ width: img.width, height: img.height });
      };
      img.src = dataUrl;
    },
    []
  );

  // Image quality management hook
  const {
    inputQuality,
    setInputQuality,
    customSquareSize,
    setCustomSquareSize,
    isApplyingQuality,
    applyInputQualityPreset,
  } = useImageQuality({
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
  });

  // Track if this is a new upload to properly set originalImage
  const isNewUploadRef = useRef(false);

  // Track previous task to detect task changes
  const previousTaskRef = useRef<string>(aiTask);

  // Auto-update parameters when task changes
  useEffect(() => {
    // Only update if task actually changed (not on initial mount)
    if (previousTaskRef.current !== aiTask && previousTaskRef.current !== "") {
      const defaults = getDefaultParametersForTask(aiTask);
      queueMicrotask(() => {
        setGuidanceScale(defaults.guidanceScale);
        setInferenceSteps(defaults.inferenceSteps);
      });
    }
    previousTaskRef.current = aiTask;
  }, [aiTask, getDefaultParametersForTask]);

  // Initialize history and set original image when a new image is uploaded
  // Note: originalImage is now set directly in handleImageUploadWrapper to avoid timing issues
  useEffect(() => {
    if (uploadedImage && isNewUploadRef.current) {
      // Only set originalImage if flag is true (new upload)
      // This is a backup in case handleImageUploadWrapper didn't set it
      if (!originalImage) {
        queueMicrotask(() => {
          setOriginalImage(uploadedImage);
          initializeHistory(uploadedImage);
        });
      }
      isNewUploadRef.current = false; // Reset flag
    }
    // If uploadedImage changed but flag is false,
    // it means user returned to original or loaded from history
    // Don't overwrite originalImage in this case
  }, [uploadedImage, originalImage, imageDimensions, initializeHistory]);

  // Download handler - handles both Object URL (blob:) and base64 data URLs
  const handleDownload = async () => {
    if (!uploadedImage) return;

    try {
      let blob: Blob;
      let downloadUrl: string;

      // Check if it's a blob URL (Object URL)
      if (uploadedImage.startsWith("blob:")) {
        // Fetch the blob from the Object URL
        const response = await fetch(uploadedImage);
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.statusText}`);
        }
        blob = await response.blob();
        downloadUrl = URL.createObjectURL(blob);
      } 
      // Check if it's a base64 data URL
      else if (uploadedImage.startsWith("data:")) {
        // Convert base64 to blob
        const response = await fetch(uploadedImage);
        blob = await response.blob();
        downloadUrl = URL.createObjectURL(blob);
      } 
      // Fallback: try to use directly (shouldn't happen but just in case)
      else {
        downloadUrl = uploadedImage;
      }

      // Create download link
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = "artmancer-edited-image.png";
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link);
        // Only revoke if we created a new Object URL
        if (downloadUrl.startsWith("blob:") && downloadUrl !== uploadedImage) {
          URL.revokeObjectURL(downloadUrl);
        }
      }, 100);
    } catch (error) {
      console.error("❌ [handleDownload] Failed to download image:", error);
      setNotificationWithTimeout(
        "error",
        `Failed to download image: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  };

  // Create wrapped image upload handler that initializes history
  const handleImageUploadWrapper = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // First clear any existing state including mask history
    setOriginalImage(null);
    setModifiedImageForComparison(null);
    setComparisonSlider(50);
    setSavedMaskData(null); // Clear saved mask when uploading new image
    setSavedMaskCanvasState(null); // Clear saved mask canvas state
    resetMaskHistory();

    // Set flag to indicate this is a new upload
    isNewUploadRef.current = true;

    // Read the file and set states immediately
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;

      // Create an image element to get dimensions
      const img = document.createElement("img");
      img.onload = () => {
        const dims = { width: img.width, height: img.height };
        // Store original upload data (never overwritten)
        setOriginalUploadData(imageData);
        setOriginalUploadDimensions(dims);
        setBaseImage(imageData, dims);

        // Immediately update uploadedImage with the new image to avoid display lag
        // This ensures the canvas shows the new image right away
        // applyInputQualityPreset will update it again with the processed version
        setUploadedImage(imageData);
        // For a brand new upload, show mask by default so user can start masking
        setMaskVisible(true);
        setIsMaskVisible(true);
        setImageDimensions(dims);

        void applyInputQualityPreset(inputQuality, {
          sourceData: imageData,
          sourceDimensions: dims,
          silent: true,
        });
        isNewUploadRef.current = false; // Reset flag
      };
      img.src = imageData;
    };
    reader.readAsDataURL(file);
  };

  // Handle image removal
  const handleRemoveImage = () => {
    // Clear saved masks when removing image
    setSavedMaskData(null);
    setSavedMaskCanvasState(null);
    removeImage();
    setOriginalImage(null);
    setModifiedImageForComparison(null);
    setComparisonSlider(50);
    resetMaskHistory();
    setBaseImage(null);
    // Clear original upload data
    setOriginalUploadData(null);
    setOriginalUploadDimensions(null);
    setInputQuality("resized");
    setLastRequestId(null);
  };


  // Handle return to original image
  const handleReturnToOriginal = () => {
    if (originalImage) {
      // Set flag to skip mask reset before changing uploadedImage
      skipMaskResetRef.current = true;
      // Set uploadedImage and modifiedImage back to originalImage
      setUploadedImage(originalImage);
      setModifiedImage(originalImage);
      // Clear the comparison image so slider shows original on both sides
      setModifiedImageForComparison(null);
      setComparisonSlider(50);
      // Keep mask when returning to original image (don't reset mask history or clear mask)
      // resetMaskHistory();
      // clearMask();
      setBaseImage(originalImage, imageDimensions ?? undefined);
      // Reset mask visibility to show (because after generation, mask is hidden)
      setMaskVisible(true);
      setIsMaskVisible(true);
      
      // CRITICAL: Restore mask canvas state khi quay lại original image
      // Đảm bảo mask hiển thị đúng (mask của lần generate hiện tại, không phải lần trước)
      if (savedMaskCanvasState) {
        console.log("🔄 [handleReturnToOriginal] Restoring mask canvas state", {
          savedStateLength: savedMaskCanvasState.length,
        });
        // Đợi sau khi resizeCanvas hoàn thành (tương tự như sau generation)
        setTimeout(async () => {
          const restored = await restoreMaskCanvasState(savedMaskCanvasState);
          if (restored) {
            console.log("✅ [handleReturnToOriginal] Mask canvas restored successfully");
          } else {
            console.warn("⚠️ [handleReturnToOriginal] Failed to restore mask canvas");
          }
        }, 250);
      }
      
      setNotificationWithTimeout("success", "Returned to original image");
    }
  };

  // Advanced options handlers
  const handleNegativePromptChange = (value: string) => {
    setNegativePrompt(value);
  };

  const handleGuidanceScaleChange = (value: number) => {
    setGuidanceScale(value);
  };

  const handleInferenceStepsChange = (value: number) => {
    setInferenceSteps(value);
  };


  const handleSeedChange = (value: number) => {
    setSeed(value);
  };

  const handleBorderAdjustmentChange = (value: number) => {
    setBorderAdjustment(value);
  };

  const handleInputQualityChange = useCallback(
    async (quality: InputQualityPreset) => {
      if (quality === inputQuality && baseImageData) {
        return;
      }
      setInputQuality(quality);
      if (!baseImageData) {
        return;
      }
      await applyInputQualityPreset(quality);
    },
    [applyInputQualityPreset, baseImageData, inputQuality, setInputQuality]
  );

  const handleCustomSquareSizeChange = useCallback(
    async (size: number) => {
      setCustomSquareSize(size);
      // Re-apply quality preset with new size if in resized mode
      if (inputQuality === "resized" && baseImageData) {
        await applyInputQualityPreset("resized", { squareSize: size });
      }
    },
    [applyInputQualityPreset, baseImageData, inputQuality, setCustomSquareSize]
  );

  // Handle image generation
  const handleEdit = async (prompt: string): Promise<void> => {
    // 🔍 DEBUG: Log when handleEdit is called
    console.log("🎯 [page.tsx handleEdit] Called", {
      timestamp: new Date().toISOString(),
      prompt,
      stackTrace: new Error().stack?.split('\n').slice(2, 5).join('\n'),
    });

    // Declare intervals outside try block so they can be cleared in catch
    let pipelineInterval: NodeJS.Timeout | null = null;
    let generationInterval: NodeJS.Timeout | null = null;

    try {
      clearAllNotifications();
      clearError();

      // Clear debug info at the start of each generation to ensure fresh data
      setDebugInfo(null);
      setIsDebugPanelVisible(false);

      if (!uploadedImage) {
        setNotificationWithTimeout(
          "error",
          "Please upload an image first to edit it."
        );
        return;
      }

      // White balance doesn't require mask
      if (aiTask !== "white-balance" && !hasMaskContent) {
        setNotificationWithTimeout(
          "error",
          "Please create a mask to specify the area to edit."
        );
        return;
      }

      // Validate reference image for object insertion
      if (aiTask === "object-insert" && !referenceImage) {
        setNotificationWithTimeout(
          "error",
          "Please upload a reference image for object insertion."
        );
        return;
      }

      // Get final mask (Black/White binary mask for AI) - only for non-white-balance tasks
      let maskImageData: string | null = null;
      if (aiTask !== "white-balance") {
        // Lưu mask canvas state (UI mask brush) TRƯỚC khi export
        const canvasState = saveMaskCanvasState();
        if (canvasState) {
          console.log("💾 [handleEdit] Saving mask canvas state (UI mask brush)", {
            stateLength: canvasState.length,
            previousStateLength: savedMaskCanvasState?.length || 0,
          });
          // CRITICAL: Clear state cũ trước khi set state mới để tránh conflict
          setSavedMaskCanvasState(null);
          // Set state mới
          setSavedMaskCanvasState(canvasState);
        } else {
          console.warn("⚠️ [handleEdit] No mask canvas state to save");
          // Clear state nếu không có mask
          setSavedMaskCanvasState(null);
        }
        
        maskImageData = await getFinalMask();
        if (!maskImageData) {
          setNotificationWithTimeout(
            "error",
            "Failed to export mask. Please try again."
          );
          return;
        }
        
        // Lưu mask binary data (đen trắng) để gửi API
        console.log("💾 [handleEdit] Saving mask binary data to state");
        setSavedMaskData(maskImageData);
      } else {
        // Clear saved masks for white-balance task
        setSavedMaskData(null);
        setSavedMaskCanvasState(null);
      }

      // Log mask export for debugging
      console.log("🎭 Final mask exported (Black/White binary):", {
        imageDimensions: imageDimensions
          ? `${imageDimensions.width}x${imageDimensions.height}`
          : "N/A",
        format: "PNG (Binary: Black background, White mask area)",
      });

      // CRITICAL: Before generating, ensure originalImage is set to the current uploadedImage
      // This is the image that will be used as input for generation
      // After generation, originalImage will remain as the input, and modifiedImageForComparison will be the output
      if (!originalImage) {
        // If originalImage is null, set it to current uploadedImage
        setOriginalImage(uploadedImage);
      } else if (originalImage !== uploadedImage) {
        // If originalImage exists but is different from uploadedImage,
        // it means user uploaded a new image after returning to original
        // In this case, update originalImage to the new uploaded image
        setOriginalImage(uploadedImage);
      }

      // Initialize progress tracking
      // Will be updated from SSE events
      stepTimingRef.current = []; // Reset timing tracking
      setGenerationProgress({
        current: 0,
        total: inferenceSteps, // Initial total, will be updated from SSE
        status: "loading_pipeline",
        progressPercent: undefined,
      });

      // Get settings from state
      // Note: width and height are not set - backend will use original image size automatically
      const settings = {
        num_inference_steps: inferenceSteps,
        guidance_scale: guidanceScale,
        // Use same value for true_cfg_scale (Qwen uses this, SD uses guidance_scale)
        true_cfg_scale: guidanceScale,
        negative_prompt: negativePrompt || undefined,
        generator_seed: seed, // Use seed from state (default: 42)
        input_quality: inputQuality,
      };

      // Use uploadedImage (current image) for API call
      // originalImage is preserved for comparison view
      // Include reference image if task is object-insert
      const referenceImageForApi =
        aiTask === "object-insert" ? referenceImage : null;

      // Include reference mask R if task is object-insert (for two-source mask workflow)
      const referenceMaskRForApi =
        aiTask === "object-insert" ? referenceMaskR : null;

      // Debug logging
      console.log("🔍 [Frontend Debug] Generation Request:", {
        aiTask,
        hasReferenceImage: !!referenceImage,
        referenceImageForApi: referenceImageForApi ? "provided" : "null",
        hasReferenceMaskR: !!referenceMaskR,
        referenceMaskRForApi: referenceMaskRForApi ? "provided" : "null",
        usingTwoSourceMasks: !!(referenceImageForApi && referenceMaskRForApi),
        taskType: aiTask === "object-insert" ? "insertion" : "removal",
      });

      // Map aiTask to task_type for backend
      const taskType =
        aiTask === "white-balance"
          ? "white-balance"
          : aiTask === "object-insert"
          ? "object-insert"
          : "object-removal";

      // Use prompt directly (no temperature/tint conversion)
      const finalPrompt = prompt;

      // Reset generation completed flag at start of new generation
      generationCompletedRef.current = false;

      // Progress callback to update from SSE events
      const progressCallback = (progress: {
        current_step?: number;
        total_steps?: number;
        status: string;
        progress: number;
        loading_message?: string;
      }) => {
        // Don't update progress if generation has already completed
        if (generationCompletedRef.current) {
          console.log("🚫 [Progress Callback] Ignoring update - generation already completed");
          return;
        }

        console.log("🔄 [Progress Callback] Received update:", {
          status: progress.status,
          current_step: progress.current_step,
          total_steps: progress.total_steps,
          progress: progress.progress,
        });

        setGenerationProgress((prev) => {
          if (!prev) return null;

          // Update total_steps if provided
          const total = progress.total_steps || prev.total;

          // Map status from backend to frontend
          let status = prev.status;
          if (
            progress.status === "queued" ||
            progress.status === "initializing_h100" ||
            progress.status === "loading_pipeline"
          ) {
            status = "loading_pipeline";
          } else if (progress.status === "processing") {
            status = "processing";
          }

          console.log("📊 [Progress Callback] Status mapping:", {
            backend_status: progress.status,
            frontend_status: status,
            prev_status: prev.status,
          });

          // Calculate current step from progress percentage or use current_step
          let current = prev.current;
          let estimatedTimeRemaining: number | undefined = undefined;

          // Reset current to 0 when transitioning from loading to processing
          if (
            status === "processing" &&
            prev.status === "loading_pipeline"
          ) {
            current = 0;
            // Clear step timing when starting processing
            stepTimingRef.current = [];
            // Track when processing starts
            processingStartTimeRef.current = Date.now();
          }

          // Reset current to 0 when processing starts (if no step info yet)
          if (
            status === "processing" &&
            (progress.current_step === null ||
              progress.current_step === undefined)
          ) {
            current = 0;
            // Track when processing starts if not already tracked
            if (processingStartTimeRef.current === null) {
              processingStartTimeRef.current = Date.now();
            }
          }

          // Only use step count when status is "processing"
          if (
            status === "processing" &&
            progress.current_step !== undefined &&
            progress.current_step !== null &&
            progress.total_steps !== undefined
          ) {
            // Use actual step count from backend
            current = progress.current_step;

            // Calculate estimated time remaining for inference steps
            const now = Date.now();
            stepTimingRef.current.push({ step: current, timestamp: now });

            // Keep only last 5 steps for average calculation
            if (stepTimingRef.current.length > 5) {
              stepTimingRef.current.shift();
            }

            // Calculate average time per step
            if (stepTimingRef.current.length >= 2) {
              const steps = stepTimingRef.current;
              const timeDiffs = [];
              for (let i = 1; i < steps.length; i++) {
                timeDiffs.push(steps[i].timestamp - steps[i - 1].timestamp);
              }
              const avgTimePerStep =
                timeDiffs.reduce((a, b) => a + b, 0) / timeDiffs.length;
              const remainingSteps = total - current;
              estimatedTimeRemaining = Math.ceil(
                (remainingSteps * avgTimePerStep) / 1000
              ); // Convert to seconds
            }
          } else if (status === "loading_pipeline") {
            // For loading phase, keep current at 0 (don't map progress to steps)
            // Progress bar will use progress.progress (0.0-1.0) directly
            current = 0;
          }

          // Ensure current is always a number (default to 0 if null/undefined)
          if (current === null || current === undefined) {
            current = 0;
          }

          return {
            current,
            total,
            status,
            loadingMessage: progress.loading_message,
            estimatedTimeRemaining,
            progressPercent: progress.progress !== undefined ? progress.progress : undefined,
          };
        });
      };

      const result = await generateImage(
        finalPrompt,
        uploadedImage,
        maskImageData || "", // Empty string for white balance (mask not required)
        settings,
        referenceImageForApi,
        referenceMaskRForApi,
        taskType,
        progressCallback,
        maskToolType === "eraser" ? "brush" : maskToolType, // Convert "eraser" to "brush" for API
        enableMaeRefinement, // Pass MAE refinement setting
        isDebugPanelVisible // Pass debug panel visibility as enable_debug flag
      );

      // Clear intervals and progress when generation completes
      if (pipelineInterval) {
        clearInterval(pipelineInterval);
        pipelineInterval = null;
      }
      if (generationInterval) {
        clearInterval(generationInterval);
        generationInterval = null;
      }
      // Mark generation as completed to prevent any late SSE events from updating progress
      generationCompletedRef.current = true;
      setGenerationProgress(null);

      if (result && result.image) {
        // Backend giờ trả về Object URL (blob:...) thay vì base64
        const imageData = result.image;

        // Update image dimensions from the generated image to ensure correct display
        const img = new Image();
        img.onload = () => {
          // Update dimensions if they changed (should match original, but verify)
          if (
            imageDimensions &&
            (imageDimensions.width !== img.width ||
              imageDimensions.height !== img.height)
          ) {
            console.log(
              `📐 Image size changed: ${imageDimensions.width}x${imageDimensions.height} -> ${img.width}x${img.height}`
            );
            // Keep original dimensions for display consistency
            // The generated image should match, but if not, we keep the original dimensions
          }
          setBaseImage(imageData, { width: img.width, height: img.height });
        };
        img.src = imageData;

        // Preserve mask when updating to generated image
        // Set flag để preserve mask trong resizeCanvas (tránh bị clear khi canvas resize)
        // Sau đó sẽ restore từ savedMaskCanvasState để đảm bảo mask đúng
        skipMaskResetRef.current = true;
        
        // Update displayed image
        setUploadedImage(imageData);
        setModifiedImage(imageData);

        // Set the AI-generated image for comparison (this is the new AI result)
        setModifiedImageForComparison(imageData);

        // Ẩn mask mặc định sau khi có kết quả mới (giữ dữ liệu mask để toggle lại)
        setMaskVisible(false);
        setIsMaskVisible(false);

        // Restore mask canvas state (UI mask brush) sau khi generation hoàn thành
        // Đợi tất cả resizeCanvas hoàn thành (resizeCanvas chạy ở 0ms, 50ms, 200ms)
        if (savedMaskCanvasState) {
          console.log("🔄 [handleEdit] Restoring mask canvas state after generation", {
            savedStateLength: savedMaskCanvasState.length,
            imageDimensions: imageDimensions ? `${imageDimensions.width}x${imageDimensions.height}` : 'null'
          });
          // Đợi sau khi tất cả resizeCanvas hoàn thành (sau 250ms)
          // Sau đó restore một lần, nếu không được thì xóa savedMaskCanvasState
          setTimeout(async () => {
            const currentSavedState = savedMaskCanvasState; // Capture current state
            const restored = await restoreMaskCanvasState(currentSavedState);
            if (restored) {
              console.log("✅ [handleEdit] Mask canvas restored successfully");
              // CRITICAL: Clear saved state sau khi restore thành công
              // Để đảm bảo lần generate tiếp theo sẽ dùng mask mới nhất
              setSavedMaskCanvasState(null);
              // Reset flag sau khi restore xong
              skipMaskResetRef.current = false;
            } else {
              console.warn("⚠️ [handleEdit] Failed to restore mask canvas, clearing saved state");
              setSavedMaskCanvasState(null);
              // Reset flag nếu restore failed
              skipMaskResetRef.current = false;
            }
          }, 250); // Đợi sau khi tất cả resizeCanvas hoàn thành (200ms + buffer)
        } else {
          console.warn("⚠️ [handleEdit] No saved mask canvas state to restore");
          // Reset flag nếu không có saved state
          skipMaskResetRef.current = false;
        }

        // Add to history
        addToHistory(imageData);

        // Save request_id for visualization download
        if (result.request_id) {
          setLastRequestId(result.request_id);
        }

        // Always update debug_info (even if null) to ensure fresh data for each generation
        // This ensures debug panel shows correct info for current generation, not previous one
        // Merge backend debug_info with prompt info from frontend
        const debugInfoWithPrompt = {
          ...(result.debug_info || {}),
          original_prompt: prompt,
          debug_path: result.debug_path, // Include debug_path for download
        };
        setDebugInfo(debugInfoWithPrompt);
        // Auto-show debug panel when debug info is available
        if (result.debug_info) {
          setIsDebugPanelVisible(true);
        }

        setNotificationWithTimeout(
          "success",
          `Image edited successfully! (${result.generation_time.toFixed(1)}s)`
        );

        console.log("Generation successful:", {
          model: result.model_used,
          time: result.generation_time,
          parameters: result.parameters_used,
          request_id: result.request_id,
          debug_info: result.debug_info,
        });
      }
    } catch (err) {
      console.error("Generation failed:", err);
      // Clear any intervals and progress on error
      if (pipelineInterval) {
        clearInterval(pipelineInterval);
        pipelineInterval = null;
      }
      if (generationInterval) {
        clearInterval(generationInterval);
        generationInterval = null;
      }
      // Mark generation as completed to prevent any late SSE events from updating progress
      generationCompletedRef.current = true;
      setGenerationProgress(null);
      processingStartTimeRef.current = null;
      setNotificationWithTimeout(
        "error",
        "Failed to edit image. Please try again."
      );
    }
  };

  // Effect to handle generation errors
  useEffect(() => {
    if (generationError) {
      queueMicrotask(() =>
        setNotificationWithTimeout("error", generationError)
      );
    }
  }, [generationError, setNotificationWithTimeout]);

  // Check API connectivity on mount with retry logic
  // Removed automatic backend health check notification
  // Backend status is now managed by ServerContext and displayed in Header
  // This reduces redundant API calls and improves cost optimization

  // Cleanup effect to clear notification timeouts
  useEffect(() => {
    return () => {
      clearNotification();
    };
  }, [clearNotification]);


  return (
    <div className="min-h-screen max-h-screen bg-primary-bg text-text-primary flex flex-col dots-pattern-small overflow-hidden">
      {/* Header */}
      <Header
        onSummon={handleEdit}
        isCustomizeOpen={isCustomizeOpen}
        onToggleCustomize={() => setIsCustomizeOpen(!isCustomizeOpen)}
        isGenerating={isGenerating}
        aiTask={aiTask}
        isDebugPanelVisible={isDebugPanelVisible}
        onDebugPanelVisibilityChange={setIsDebugPanelVisible}
      />

      {/* Main Content - Optimized transitions */}
      <main
        className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden relative"
        style={{
          paddingRight: isCustomizeOpen ? `${sidebarWidth}px` : "0px",
          transition: isResizing
            ? "none"
            : "padding-right 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          transform: "translateZ(0)", // Force hardware acceleration
          willChange: isResizing ? "padding-right" : "auto",
        }}
      >
        {/* Loading Overlay */}
        {isGenerating && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-40">
            <div className="bg-secondary-bg border border-primary-accent rounded-lg p-6 text-center min-w-[300px]">
              <div className="animate-spin w-8 h-8 border-4 border-primary-accent border-t-transparent rounded-full mx-auto mb-3"></div>
              <p className="text-text-primary font-medium">
                Generating image...
              </p>
              {/* Generation Progress */}
              {isGenerating &&
                generationProgress &&
                !(
                  generationProgress.status === "processing" &&
                  generationProgress.total > 0 &&
                  (generationProgress.current ?? 0) >= generationProgress.total
                ) && (() => {
                  // Determine current status message
                  const statusMessage = generationProgress.status === "loading_pipeline"
                    ? generationProgress.loadingMessage || "Loading model..."
                    : generationProgress.status === "processing"
                    ? (() => {
                        const current = generationProgress.current ?? 0;
                        const total = generationProgress.total;
                        
                        // If no steps have started yet, show initialization messages
                        if (current === 0 && processingStartTimeRef.current !== null) {
                          const elapsedSeconds = Math.floor(
                            (Date.now() - processingStartTimeRef.current) / 1000
                          );
                          
                          // Show different messages based on elapsed time
                          if (elapsedSeconds < 3) {
                            return "Initializing...";
                          } else if (elapsedSeconds < 6) {
                            return "Preparing model...";
                          } else if (elapsedSeconds < 10) {
                            return "Setting up pipeline...";
                          } else {
                            return "Almost ready...";
                          }
                        }
                        
                        // Once steps start, show normal progress
                        return `Processing: ${current} / ${total} steps`;
                      })()
                    : `Step ${generationProgress.current ?? 0} / ${
                        generationProgress.total
                      }`;
                  
                  // Check if we should hide progress bar and percentage
                  const shouldHideProgress = statusMessage === "Almost ready...";
                  
                  return (
                    <div className="mt-4">
                      {/* Only show progress bar if not "Almost ready..." */}
                      {!shouldHideProgress && (
                        <div className="w-full bg-border-color rounded-full h-2 mb-2 overflow-hidden">
                          <div
                            className="progress-bar-smooth bg-primary-accent h-2 rounded-full"
                            style={{
                              width: `${
                                generationProgress.status === "loading_pipeline"
                                  ? // For loading phase, use progressPercent (0.0-1.0) if available
                                    generationProgress.progressPercent !== undefined
                                    ? generationProgress.progressPercent * 100
                                    : 0
                                  : // For processing phase, use step-based progress
                                    generationProgress.total > 0
                                  ? ((generationProgress.current ?? 0) /
                                      generationProgress.total) *
                                    100
                                  : 0
                              }%`,
                            }}
                          ></div>
                        </div>
                      )}
                      <p className="text-text-secondary text-sm">
                        {statusMessage}
                      </p>
                      {/* Only show percentage for processing phase and not "Almost ready..." */}
                      {!shouldHideProgress &&
                        generationProgress.status === "processing" &&
                        generationProgress.total > 0 && (
                          <span className="block text-xs mt-1 opacity-75">
                            {Math.round(
                              ((generationProgress.current ?? 0) /
                                generationProgress.total) *
                                100
                            )}
                            %
                            {generationProgress.estimatedTimeRemaining !==
                              undefined &&
                              generationProgress.status === "processing" && (
                                <span className="ml-2">
                                  • ~{generationProgress.estimatedTimeRemaining}s
                                  remaining
                                </span>
                              )}
                          </span>
                        )}
                    </div>
                  );
                })()}
              {/* Fallback message when no progress available */}
              {!generationProgress && (
                <p className="text-text-secondary text-sm mt-1">
                  {isGenerating
                    ? "This may take a few seconds"
                    : "Please wait..."}
                </p>
              )}
              {isGenerating && (
                <button
                  onClick={cancelGeneration}
                  className="btn-interactive mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium transition-colors"
                >
                  Cancel
                </button>
              )}
            </div>
          </div>
        )}

        {/* Refined Notification Component */}
        <NotificationComponent
          type={notificationType}
          message={notificationMessage}
          isVisible={isNotificationVisible}
          onClose={handleCloseNotification}
          duration={5000}
          position="top"
        />

        {/* Left Side - Canvas */}
        <Canvas
          uploadedImage={uploadedImage}
          imageDimensions={imageDimensions}
          displayScale={displayScale}
          transform={transform}
          viewportZoom={viewportZoom}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          maskToolType={maskToolType}
          maskVisible={maskVisible}
          isSmartMaskLoading={isSmartMaskLoading}
          hasMaskContent={hasMaskContent}
          originalImage={originalImage}
          modifiedImage={modifiedImageForComparison}
          comparisonSlider={comparisonSlider}
          historyIndex={historyIndex}
          historyStackLength={historyStack.length}
          isHelpOpen={isHelpOpen}
          imageContainerRef={imageContainerRef}
          containerRef={containerRef}
          maskCanvasRef={maskCanvasRef}
          edgeOverlayCanvasRef={edgeOverlayCanvasRef}
          imageRef={imageRef}
          onImageUpload={handleImageUploadWrapper}
          onRemoveImage={handleRemoveImage}
          onImageClick={handleImageClick}
          onWheel={handleWheel}
          onMaskMouseDown={handleMaskMouseDown}
          onMaskMouseMove={handleMaskMouseMove}
          onMaskMouseUp={handleMaskMouseUp}
          onComparisonSliderChange={setComparisonSlider}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onDownload={handleDownload}
          onZoomViewportIn={zoomViewportIn}
          onZoomViewportOut={zoomViewportOut}
          onResetViewportZoom={resetViewportZoom}
          onToggleHelp={() => setIsHelpOpen(!isHelpOpen)}
        />

        {/* Help Box Component */}
        <HelpBox isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />

        {/* Right Side - Customize Panel */}
        <Sidebar
          isOpen={isCustomizeOpen}
          width={sidebarWidth}
          isResizing={isResizing}
          uploadedImage={uploadedImage}
          imageDimensions={imageDimensions}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          maskToolType={maskToolType}
          referenceImage={referenceImage}
          aiTask={aiTask}
          inputQuality={inputQuality}
          customSquareSize={customSquareSize}
          isApplyingQuality={isApplyingQuality}
          onRemoveImage={handleRemoveImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
          onMaskToolTypeChange={setMaskToolType}
          onToggleMaskVisible={handleToggleMaskVisible}
          hasMaskContent={hasMaskContent}
          isMaskVisible={maskVisible}
          enableSmartMasking={enableSmartMasking}
          isSmartMaskLoading={isSmartMaskLoading}
          onSmartMaskingChange={setEnableSmartMasking}
          smartMaskModelType={smartMaskModelType}
          onSmartMaskModelTypeChange={setSmartMaskModelType}
          borderAdjustment={borderAdjustment}
          onBorderAdjustmentChange={handleBorderAdjustmentChange}
          onDetectSmartMask={handleDetectSmartMask}
          onReferenceImageUpload={handleReferenceImageUpload}
          onRemoveReferenceImage={handleRemoveReferenceImage}
          onEditReferenceImage={handleEditReferenceImage}
          onAiTaskChange={handleAiTaskChange}
          onResizeStart={handleResizeStart}
          onWidthChange={setSidebarWidth}
          originalImage={originalImage}
          modifiedImage={modifiedImageForComparison}
          onReturnToOriginal={handleReturnToOriginal}
          negativePrompt={negativePrompt}
          guidanceScale={guidanceScale}
          inferenceSteps={inferenceSteps}
          onNegativePromptChange={handleNegativePromptChange}
          onGuidanceScaleChange={handleGuidanceScaleChange}
          onInferenceStepsChange={handleInferenceStepsChange}
          seed={seed}
          onSeedChange={handleSeedChange}
          enableMaeRefinement={enableMaeRefinement}
          onEnableMaeRefinementChange={setEnableMaeRefinement}
          onInputQualityChange={handleInputQualityChange}
          onCustomSquareSizeChange={handleCustomSquareSizeChange}
        />
      </main>

      {/* Debug Panel - Hidden by default, shows conditional images */}
      <DebugPanel
        debugInfo={debugInfo}
        isVisible={isDebugPanelVisible}
        onClose={() => setIsDebugPanelVisible(false)}
        onOpen={() => setIsDebugPanelVisible(true)}
      />

      {/* Reference Image Editor Modal */}
      {pendingRefImage && (
        <ReferenceImageEditor
          isOpen={isRefEditorOpen}
          imageData={pendingRefImage}
          onClose={handleRefEditorClose}
          onSubmit={handleRefEditorSubmit}
          initialMaskData={referenceMaskR}
          modelType={smartMaskModelType}
          onModelTypeChange={setSmartMaskModelType}
          borderAdjustment={borderAdjustment}
          onBorderAdjustmentChange={handleBorderAdjustmentChange}
          enableSmartMasking={enableSmartMasking}
          onSmartMaskingChange={setEnableSmartMasking}
          onNotification={setNotificationWithTimeout}
        />
      )}
    </div>
  );
}
