"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
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

export default function Home() {
  // Basic UI state
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Enhanced notification state
  const [notificationType, setNotificationType] =
    useState<NotificationType>("success");
  const [notificationMessage, setNotificationMessage] = useState<string>("");
  const [isNotificationVisible, setIsNotificationVisible] = useState(false);

  // App Mode state (Inference or Evaluation)
  const [appMode, setAppMode] = useState<"inference" | "evaluation">(
    "inference"
  );

  // AI Task state
  const [aiTask, setAiTask] = useState<
    "white-balance" | "object-insert" | "object-removal" | "evaluation"
  >("white-balance");
  const [referenceImage, setReferenceImage] = useState<string | null>(null);

  // Advanced options state
  const [negativePrompt, setNegativePrompt] = useState<string>("");
  // Task-specific default parameters
  const getDefaultParametersForTask = useCallback((task: string) => {
    switch (task) {
      case "object-removal":
        return {
          guidanceScale: 7.0, // Strict for removal (higher in 1-10 range)
          inferenceSteps: 50, // High quality
          cfgScale: 4.0, // Keep default
        };
      case "object-insert":
        return {
          guidanceScale: 4.0, // Balanced for insertion (middle of 1-10 range)
          inferenceSteps: 50, // High quality
          cfgScale: 4.0, // Keep default
        };
      case "white-balance":
        return {
          guidanceScale: 1.0, // Not used for white-balance
          inferenceSteps: 20, // Pix2Pix default
          cfgScale: 4.0, // Keep default
        };
      default:
        return {
          guidanceScale: 1.0,
          inferenceSteps: 40,
          cfgScale: 4.0,
        };
    }
  }, []);

  const [guidanceScale, setGuidanceScale] = useState<number>(1.0);
  const [inferenceSteps, setInferenceSteps] = useState<number>(20); // Start with white-balance default
  const [cfgScale, setCfgScale] = useState<number>(4.0);

  // White balance state
  const [whiteBalanceTemperature, setWhiteBalanceTemperature] =
    useState<number>(0);
  const [whiteBalanceTint, setWhiteBalanceTint] = useState<number>(0);

  // Visualization request ID state
  const [lastRequestId, setLastRequestId] = useState<string | null>(null);

  // Evaluation mode state
  const [evaluationTask, setEvaluationTask] = useState<
    "white-balance" | "object-insert" | "object-removal"
  >("white-balance");
  const [evaluationMode, setEvaluationMode] = useState<"single" | "multiple">(
    "single"
  );
  const [evaluationSingleOriginal, setEvaluationSingleOriginal] = useState<
    string | null
  >(null);
  const [evaluationSingleTarget, setEvaluationSingleTarget] = useState<
    string | null
  >(null);
  const [evaluationImagePairs, setEvaluationImagePairs] = useState<
    Array<{
      original: string | null;
      target: string | null;
      filename: string;
    }>
  >([]);
  const [evaluationConditionalImages, setEvaluationConditionalImages] =
    useState<string[]>([]);
  const [evaluationReferenceImage, setEvaluationReferenceImage] = useState<
    string | null
  >(null);
  const [evaluationResults, setEvaluationResults] = useState<any[]>([]);
  const [evaluationResponse, setEvaluationResponse] = useState<any>(null);
  const [evaluationDisplayLimit, setEvaluationDisplayLimit] =
    useState<number>(10);
  const [allowMultipleFolders, setAllowMultipleFolders] =
    useState<boolean>(false);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evaluationProgress, setEvaluationProgress] = useState<{
    current: number;
    total: number;
    currentPair?: string;
  } | null>(null);

  // Notification timeout refs
  const notificationTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Resizable panel state - start with default, then load from localStorage
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Load sidebar width from localStorage after hydration
  useEffect(() => {
    const saved = localStorage.getItem("artmancer-sidebar-width");
    if (saved) {
      setSidebarWidth(parseInt(saved, 10));
    }
  }, []);

  // Notification helpers with auto-hide functionality
  const clearNotificationTimeout = useCallback(() => {
    if (notificationTimeoutRef.current) {
      clearTimeout(notificationTimeoutRef.current);
      notificationTimeoutRef.current = null;
    }
  }, []);

  const handleCloseNotification = useCallback(() => {
    setIsNotificationVisible(false);
    clearNotificationTimeout();
    setSuccess(null);
    setError(null);
  }, [clearNotificationTimeout]);

  const setNotificationWithTimeout = useCallback(
    (type: NotificationType, message: string, timeoutMs: number = 5000) => {
      clearNotificationTimeout();

      // Set new notification
      setNotificationType(type);
      setNotificationMessage(message);
      setIsNotificationVisible(true);

      // Legacy state for backward compatibility
      if (type === "success") {
        setSuccess(message);
        setError(null);
      } else if (type === "error") {
        setError(message);
        setSuccess(null);
      }

      // Auto-hide timer
      notificationTimeoutRef.current = setTimeout(() => {
        setIsNotificationVisible(false);
        if (type === "success") {
          setSuccess(null);
        } else {
          setError(null);
        }
      }, timeoutMs);
    },
    [clearNotificationTimeout]
  );

  const clearAllNotifications = useCallback(() => {
    clearNotificationTimeout();
    setSuccess(null);
    setError(null);
    setIsNotificationVisible(false);
  }, [clearNotificationTimeout]);

  // Custom hooks
  const {
    uploadedImage,
    imageDimensions,
    displayScale,
    handleImageUpload,
    removeImage,
    handleImageClick,
    setUploadedImage,
    setModifiedImage,
    setImageDimensions,
  } = useImageUpload();

  // API integration
  const {
    generateImage,
    applyWhiteBalance,
    isGenerating,
    error: generationError,
    lastGeneration,
    clearError,
    cancelGeneration,
  } = useImageGeneration();

  const {
    viewportZoom,
    isDragging,
    imageContainerRef,
    containerRef,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    zoomViewportIn,
    zoomViewportOut,
    resetViewportZoom,
  } = useViewportControls(imageDimensions, displayScale);

  const { transform, imageRef } = useImageTransform();

  // Reference image handling for AI tasks
  const handleReferenceImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target && typeof e.target.result === "string") {
          setReferenceImage(e.target.result);
          setError(null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveReferenceImage = () => {
    setReferenceImage(null);
  };

  // Evaluation mode handlers
  const handleEvaluationModeChange = (mode: "single" | "multiple") => {
    setEvaluationMode(mode);
    // Clear data when switching modes
    if (mode === "single") {
      setEvaluationImagePairs([]);
    } else {
      setEvaluationSingleOriginal(null);
      setEvaluationSingleTarget(null);
    }
  };

  const handleEvaluationTaskChange = (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => {
    setEvaluationTask(task);
    // Clear reference image when switching away from object-insert
    if (task !== "object-insert") {
      setEvaluationReferenceImage(null);
    }
  };

  const handleEvaluationSingleOriginalUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      setEvaluationSingleOriginal(base64);
      // Set to canvas for preview
      if (appMode === "evaluation") {
        setUploadedImage(base64);
      }
    } catch (error) {
      console.error("Error reading file:", error);
      setNotificationWithTimeout("error", "Failed to load original image");
    }
  };

  const handleEvaluationSingleTargetUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      setEvaluationSingleTarget(base64);
    } catch (error) {
      console.error("Error reading file:", error);
      setNotificationWithTimeout("error", "Failed to load target image");
    }
  };

  // Handle folder upload for original images
  const handleEvaluationOriginalFolderUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    try {
      // Sort files by numeric name (1.png, 2.png, ...)
      const sortedFiles = files
        .filter((file) => {
          const match = file.name.match(/^(\d+)\.(png|jpg|jpeg|webp)$/i);
          return match !== null;
        })
        .sort((a, b) => {
          const numA = parseInt(a.name.match(/^(\d+)/)?.[1] || "0");
          const numB = parseInt(b.name.match(/^(\d+)/)?.[1] || "0");
          return numA - numB;
        });

      if (sortedFiles.length === 0) {
        setNotificationWithTimeout(
          "error",
          "No valid image files found. Files should be named as 1.png, 2.png, etc."
        );
        return;
      }

      // Store original images temporarily
      const originalImages: string[] = [];
      for (const file of sortedFiles) {
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        originalImages.push(base64);
      }

      // Update pairs with original images
      const newPairs = originalImages.map((original, index) => {
        const existingPair = evaluationImagePairs[index];
        return {
          original: original,
          target: existingPair?.target || null, // Use null instead of empty string
          filename: `${index + 1}`,
        };
      });

      // If we have fewer originals than existing pairs, keep existing pairs
      if (originalImages.length < evaluationImagePairs.length) {
        const remainingPairs = evaluationImagePairs.slice(originalImages.length);
        setEvaluationImagePairs([...newPairs, ...remainingPairs]);
      } else {
        setEvaluationImagePairs(newPairs);
      }

      setNotificationWithTimeout(
        "success",
        `Loaded ${originalImages.length} original image(s)`
      );
    } catch (error) {
      console.error("Error processing original folder:", error);
      setNotificationWithTimeout("error", "Failed to process original images");
    }
  };

  // Handle folder upload for target images
  const handleEvaluationTargetFolderUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    try {
      // Sort files by numeric name (1.png, 2.png, ...)
      const sortedFiles = files
        .filter((file) => {
          const match = file.name.match(/^(\d+)\.(png|jpg|jpeg|webp)$/i);
          return match !== null;
        })
        .sort((a, b) => {
          const numA = parseInt(a.name.match(/^(\d+)/)?.[1] || "0");
          const numB = parseInt(b.name.match(/^(\d+)/)?.[1] || "0");
          return numA - numB;
        });

      if (sortedFiles.length === 0) {
        setNotificationWithTimeout(
          "error",
          "No valid image files found. Files should be named as 1.png, 2.png, etc."
        );
        return;
      }

      // Store target images temporarily
      const targetImages: string[] = [];
      for (const file of sortedFiles) {
        const base64 = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        targetImages.push(base64);
      }

      // Update pairs with target images
      const newPairs = targetImages.map((target, index) => {
        const existingPair = evaluationImagePairs[index];
        return {
          original: existingPair?.original || null, // Use null instead of empty string
          target: target,
          filename: `${index + 1}`,
        };
      });

      // If we have fewer targets than existing pairs, keep existing pairs
      if (targetImages.length < evaluationImagePairs.length) {
        const remainingPairs = evaluationImagePairs.slice(targetImages.length);
        setEvaluationImagePairs([...newPairs, ...remainingPairs]);
      } else {
        setEvaluationImagePairs(newPairs);
      }

      setNotificationWithTimeout(
        "success",
        `Loaded ${targetImages.length} target image(s)`
      );
    } catch (error) {
      console.error("Error processing target folder:", error);
      setNotificationWithTimeout("error", "Failed to process target images");
    }
  };

  // Legacy function for backward compatibility (kept but deprecated)
  const handleEvaluationMultipleUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    // Redirect to folder upload
    handleEvaluationOriginalFolderUpload(event);
  };

  const handleEvaluationConditionalImagesUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    try {
      // Sort files by numeric name (1.png, 2.png, ...)
      const sortedFiles = files
        .filter((file) => {
          const match = file.name.match(/^(\d+)\.(png|jpg|jpeg|webp)$/i);
          return match !== null;
        })
        .sort((a, b) => {
          const numA = parseInt(a.name.match(/^(\d+)/)?.[1] || "0");
          const numB = parseInt(b.name.match(/^(\d+)/)?.[1] || "0");
          return numA - numB;
        });

      if (sortedFiles.length === 0) {
        setNotificationWithTimeout(
          "error",
          "No valid image files found. Files should be named as 1.png, 2.png, etc."
        );
        return;
      }

      const base64Images = await Promise.all(
        sortedFiles.map(
          (file) =>
            new Promise<string>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result as string);
              reader.onerror = reject;
              reader.readAsDataURL(file);
            })
        )
      );
      setEvaluationConditionalImages(base64Images);
      setNotificationWithTimeout(
        "success",
        `Loaded ${base64Images.length} conditional image(s) from folder`
      );
    } catch (error) {
      console.error("Error reading files:", error);
      setNotificationWithTimeout("error", "Failed to load conditional images");
    }
  };

  const handleEvaluationReferenceImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      setEvaluationReferenceImage(base64);
      setNotificationWithTimeout("success", "Reference image loaded");
    } catch (error) {
      console.error("Error reading file:", error);
      setNotificationWithTimeout("error", "Failed to load reference image");
    }
  };

  const handleEvaluationInputImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      setEvaluationInputImage(base64);
    } catch (error) {
      console.error("Error reading file:", error);
      setNotificationWithTimeout("error", "Failed to load input image");
    }
  };

  const handleEvaluate = async (prompt: string) => {
    await handleEvaluateImages(prompt);
  };

  const handleEvaluateImages = async (prompt?: string) => {
    // Validate based on task type
    if (evaluationTask === "object-insert" && !evaluationReferenceImage) {
      setNotificationWithTimeout(
        "error",
        "Please upload a reference image for object insertion evaluation"
      );
      return;
    }
    // White-balance and object-removal don't require reference image

    if (evaluationMode === "single") {
      if (!evaluationSingleOriginal || !evaluationSingleTarget) {
        setNotificationWithTimeout(
          "error",
          "Please upload both original and target images"
        );
        return;
      }
    } else {
      if (evaluationImagePairs.length === 0) {
        setNotificationWithTimeout(
          "error",
          "Please upload at least one image pair"
        );
        return;
      }
    }

    try {
      setIsEvaluating(true);
      setEvaluationProgress(null); // Reset progress
      
      // Calculate total pairs for progress tracking
      const totalPairs = evaluationMode === "single" 
        ? 1 
        : evaluationImagePairs.filter((pair) => pair.original && pair.target).length;
      
      setEvaluationProgress({ current: 0, total: totalPairs });
      
      const { apiService } = await import("@/services/api");

      const request: any = {
        task_type: evaluationTask, // "white-balance" | "object-insert" | "object-removal"
        conditional_images:
          evaluationConditionalImages.length > 0
            ? evaluationConditionalImages
            : undefined,
      };

      // Add reference image for object-insert task only
      if (evaluationTask === "object-insert" && evaluationReferenceImage) {
        // Extract base64 from data URL if needed
        const base64Ref = evaluationReferenceImage.startsWith("data:")
          ? evaluationReferenceImage.split(",")[1]
          : evaluationReferenceImage;
        request.reference_image = base64Ref;
      }

      if (evaluationMode === "single") {
        request.original_image = evaluationSingleOriginal;
        request.target_image = evaluationSingleTarget;
        if (prompt) {
          request.prompt = prompt;
        }
      } else {
        // Filter out pairs that don't have both original and target images
        const validPairs = evaluationImagePairs.filter(
          (pair) => pair.original && pair.target
        );

        if (validPairs.length === 0) {
          setNotificationWithTimeout(
            "error",
            "Please upload both original and target images for all pairs"
          );
          setIsEvaluating(false);
          return;
        }

        // For multiple pairs, assign prompt to each pair
        request.image_pairs = validPairs.map((pair, index) => {
          const pairData: any = {
            original_image: pair.original!,
            target_image: pair.target!,
            filename: pair.filename,
          };
          // If prompt provided, assign to this pair
          // If prompt contains newlines or is formatted for multiple pairs, split it
          if (prompt) {
            const prompts = prompt.split('\n').filter(p => p.trim());
            if (prompts.length > index) {
              pairData.prompt = prompts[index].trim();
            } else if (prompts.length === 1) {
              // Single prompt for all pairs
              pairData.prompt = prompts[0].trim();
            } else {
              // Use the prompt as-is for all pairs
              pairData.prompt = prompt.trim();
            }
          }
          return pairData;
        });
      }

      const response = await apiService.evaluateImages(request);

      if (response.success) {
        setEvaluationResults(response.results);
        setEvaluationResponse(response);
        setNotificationWithTimeout(
          "success",
          `Evaluation completed: ${response.successful_evaluations}/${response.total_pairs} successful`
        );
      } else {
        setNotificationWithTimeout("error", "Evaluation failed");
        setEvaluationResponse(null);
      }
    } catch (error: any) {
      console.error("Evaluation error:", error);
      setNotificationWithTimeout(
        "error",
        error.message || "Failed to evaluate images"
      );
    } finally {
      setIsEvaluating(false);
    }
  };

  // Export evaluation results as JSON
  const handleExportEvaluationJSON = () => {
    if (!evaluationResponse) return;
    
    const dataStr = JSON.stringify(evaluationResponse, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `evaluation-results-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export evaluation results as CSV
  const handleExportEvaluationCSV = () => {
    if (!evaluationResults || evaluationResults.length === 0) return;
    
    const headers = ["Filename", "PSNR (dB)", "SSIM", "LPIPS", "ΔE00", "Time (s)", "Status"];
    const rows = evaluationResults.map((result) => {
      const metrics = result.metrics || {};
      return [
        result.filename || "N/A",
        metrics.psnr?.toFixed(3) || "N/A",
        metrics.ssim?.toFixed(4) || "N/A",
        metrics.lpips?.toFixed(4) || "N/A",
        metrics.de00?.toFixed(3) || "N/A",
        metrics.evaluation_time?.toFixed(2) || "N/A",
        result.success ? "Success" : "Failed",
      ];
    });
    
    // Add summary row
    const avgPsnr = evaluationResults
      .filter((r) => r.success && r.metrics?.psnr)
      .reduce((sum, r) => sum + (r.metrics.psnr || 0), 0) / 
      evaluationResults.filter((r) => r.success && r.metrics?.psnr).length || 0;
    const avgSsim = evaluationResults
      .filter((r) => r.success && r.metrics?.ssim)
      .reduce((sum, r) => sum + (r.metrics.ssim || 0), 0) / 
      evaluationResults.filter((r) => r.success && r.metrics?.ssim).length || 0;
    
    rows.push([]);
    rows.push(["Summary", "", "", "", "", "", ""]);
    rows.push([
      "Average",
      isNaN(avgPsnr) ? "N/A" : avgPsnr.toFixed(3),
      isNaN(avgSsim) ? "N/A" : avgSsim.toFixed(4),
      "",
      "",
      evaluationResponse?.total_evaluation_time?.toFixed(2) || "N/A",
      `${evaluationResponse?.successful_evaluations || 0}/${evaluationResponse?.total_pairs || 0}`,
    ]);
    
    const csvContent = [headers, ...rows]
      .map((row) => row.map((cell) => `"${cell}"`).join(","))
      .join("\n");
    
    const dataBlob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `evaluation-results-${new Date().toISOString().split("T")[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleAppModeChange = (mode: "inference" | "evaluation") => {
    setAppMode(mode);
    // Auto-switch to evaluation task when in evaluation mode
    if (mode === "evaluation") {
      setAiTask("evaluation");
    } else {
      // Reset to white-balance when switching back to inference
      setAiTask("white-balance");
    }
  };

  const handleAiTaskChange = (
    task: "white-balance" | "object-insert" | "object-removal" | "evaluation"
  ) => {
    const previousTask = aiTask;
    setAiTask(task);

    // Debug logging
    console.log("🔍 [Frontend Debug] Task changed:", {
      from: previousTask,
      to: task,
      currentReferenceImage: referenceImage ? "exists" : "null",
    });

    // Auto-switch to evaluation mode if evaluation task is selected
    if (task === "evaluation") {
      setAppMode("evaluation");
    } else if (appMode === "evaluation") {
      // Switch back to inference mode if selecting other tasks
      setAppMode("inference");
    }
    // Clear reference image when switching away from object-insert
    if (task !== "object-insert") {
      console.log(
        "🔍 [Frontend Debug] Clearing reference image (switched from object-insert)"
      );
      setReferenceImage(null);
    } else {
      console.log(
        "🔍 [Frontend Debug] Keeping reference image (object-insert task)"
      );
    }
  };

  const {
    isMaskingMode,
    isMaskDrawing,
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
    maskHistoryIndex,
    maskHistoryLength,
    undoMask,
    redoMask,
    hasMaskContent,
    getFinalMask,
    enableSmartMasking,
    setEnableSmartMasking,
    isSmartMaskLoading,
    edgeOverlayCanvasRef,
  } = useMasking(
    uploadedImage,
    imageDimensions,
    imageContainerRef,
    transform,
    viewportZoom,
    imageRef,
    setNotificationWithTimeout
  );

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

  // Track if this is a new upload to properly set originalImage
  const isNewUploadRef = useRef(false);
  
  // Track previous task to detect task changes
  const previousTaskRef = useRef<string>(aiTask);
  
  // Auto-update parameters when task changes
  useEffect(() => {
    // Only update if task actually changed (not on initial mount)
    if (previousTaskRef.current !== aiTask && previousTaskRef.current !== "") {
      const defaults = getDefaultParametersForTask(aiTask);
      setGuidanceScale(defaults.guidanceScale);
      setInferenceSteps(defaults.inferenceSteps);
      setCfgScale(defaults.cfgScale);
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
        setOriginalImage(uploadedImage);
        initializeHistory(uploadedImage);
      }
      isNewUploadRef.current = false; // Reset flag

    }
    // If uploadedImage changed but flag is false,
    // it means user returned to original or loaded from history
    // Don't overwrite originalImage in this case
  }, [uploadedImage, originalImage, imageDimensions, initializeHistory]);

  // Simple download handler
  const handleDownload = () => {
    if (!uploadedImage) return;

    const link = document.createElement("a");
    link.href = uploadedImage;
    link.download = "artmancer-edited-image.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
    resetMaskHistory();

    // Set flag to indicate this is a new upload
    isNewUploadRef.current = true;

    // Read the file and set states immediately
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;
      
      // Create an image element to get dimensions
      const img = document.createElement('img');
      img.onload = () => {
        // IMPORTANT: Set imageDimensions here to ensure it's available immediately
        // This is critical for the new layer architecture to work correctly
        setImageDimensions({ width: img.width, height: img.height });
      };
      img.src = imageData;
      
      // Set uploaded image (this will trigger useEffect)
      setUploadedImage(imageData);
      setModifiedImage(imageData);
      
      // IMPORTANT: Set originalImage immediately to avoid timing issues
      // This ensures originalImage is always the latest uploaded image
      setOriginalImage(imageData);
      initializeHistory(imageData);
      isNewUploadRef.current = false; // Reset flag
    };
    reader.readAsDataURL(file);
  };

  // Handle image removal
  const handleRemoveImage = () => {
    removeImage();
    setOriginalImage(null);
    setModifiedImageForComparison(null);
    setComparisonSlider(50);
    resetMaskHistory();
  };

  // Handle remove evaluation pair
  const handleRemoveEvaluationPair = (index: number) => {
    setEvaluationImagePairs((prevPairs) => {
      const newPairs = [...prevPairs];
      newPairs.splice(index, 1);
      return newPairs;
    });
    setNotificationWithTimeout("success", "Image pair removed");
  };

  // Handle return to original image
  const handleReturnToOriginal = () => {
    if (originalImage) {
      // Set uploadedImage and modifiedImage back to originalImage
      setUploadedImage(originalImage);
      setModifiedImage(originalImage);
      // Clear the comparison image so slider shows original on both sides
      setModifiedImageForComparison(null);
      setComparisonSlider(50);
      resetMaskHistory();
      clearMask();
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

  const handleCfgScaleChange = (value: number) => {
    setCfgScale(value);
  };

  const handleWhiteBalanceTemperatureChange = (value: number) => {
    setWhiteBalanceTemperature(value);
  };

  const handleWhiteBalanceTintChange = (value: number) => {
    setWhiteBalanceTint(value);
  };

  // Handle white balance
  const handleWhiteBalance = async (
    method: "auto" | "manual" | "ai" = "auto",
    temperature?: number,
    tint?: number
  ) => {
    try {
      clearAllNotifications();
      clearError();

      if (!uploadedImage) {
        setNotificationWithTimeout(
          "error",
          "Please upload an image first to apply white balance."
        );
        return;
      }

      // Convert base64 to File object
      const response = await fetch(uploadedImage);
      const blob = await response.blob();
      const file = new File([blob], "image.jpg", { type: "image/jpeg" });

      // Store the original image before editing
      setOriginalImage(uploadedImage);

      const result = await applyWhiteBalance(file, method, temperature, tint);

      if (result && result.corrected_image) {
        const imageData = `data:image/png;base64,${result.corrected_image}`;
        setUploadedImage(imageData);
        setModifiedImage(imageData);
        setModifiedImageForComparison(imageData);

        // Add to history
        addToHistory(imageData);

        setNotificationWithTimeout(
          "success",
          `White balance applied successfully using ${method} method!`
        );

        console.log("White balance successful:", {
          method: result.method_used,
          parameters: result.parameters,
        });
      }
    } catch (err) {
      console.error("White balance failed:", err);
      setNotificationWithTimeout(
        "error",
        "Failed to apply white balance. Please try again."
      );
    }
  };

  // Handle image generation
  const handleEdit = async (prompt: string) => {
    try {
      clearAllNotifications();
      clearError();

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
        maskImageData = await getFinalMask();
        if (!maskImageData) {
          setNotificationWithTimeout(
            "error",
            "Failed to export mask. Please try again."
          );
          return;
        }
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

      // Get settings from state
      // Note: width and height are not set - backend will use original image size automatically
      const settings = {
        num_inference_steps: inferenceSteps,
        guidance_scale: guidanceScale,
        true_cfg_scale: cfgScale,
        negative_prompt: negativePrompt || undefined,
        generator_seed: undefined, // Add seed if needed
      };

      // Use uploadedImage (current image) for API call
      // originalImage is preserved for comparison view
      // Include reference image if task is object-insert
      const referenceImageForApi =
        aiTask === "object-insert" ? referenceImage : null;

      // Debug logging
      console.log("🔍 [Frontend Debug] Generation Request:", {
        aiTask,
        hasReferenceImage: !!referenceImage,
        referenceImageForApi: referenceImageForApi ? "provided" : "null",
        taskType: aiTask === "object-insert" ? "insertion" : "removal",
      });

      // Map aiTask to task_type for backend
      const taskType = aiTask === "white-balance" 
        ? "white-balance" 
        : aiTask === "object-insert" 
        ? "object-insert" 
        : "object-removal";

      const result = await generateImage(
        prompt,
        uploadedImage,
        maskImageData || "", // Empty string for white balance (mask not required)
        settings,
        referenceImageForApi,
        taskType
      );

      if (result && result.image) {
        const imageData = `data:image/png;base64,${result.image}`;

        // Update image dimensions from the generated image to ensure correct display
        const img = new Image();
        img.onload = () => {
          // Update dimensions if they changed (should match original, but verify)
          if (imageDimensions && (imageDimensions.width !== img.width || imageDimensions.height !== img.height)) {
            console.log(`📐 Image size changed: ${imageDimensions.width}x${imageDimensions.height} -> ${img.width}x${img.height}`);
            // Keep original dimensions for display consistency
            // The generated image should match, but if not, we keep the original dimensions
          }
        };
        img.src = imageData;

        // Update displayed image
        setUploadedImage(imageData);
        setModifiedImage(imageData);

        // Set the AI-generated image for comparison (this is the new AI result)
        setModifiedImageForComparison(imageData);

        // Add to history
        addToHistory(imageData);

        // Save request_id for visualization download
        if (result.request_id) {
          setLastRequestId(result.request_id);
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
        });
      }
    } catch (err) {
      console.error("Generation failed:", err);
      setNotificationWithTimeout(
        "error",
        "Failed to edit image. Please try again."
      );
    }
  };

  // Effect to handle generation errors
  useEffect(() => {
    if (generationError) {
      setNotificationWithTimeout("error", generationError);
    }
  }, [generationError, setNotificationWithTimeout]);

  // Check API connectivity on mount with retry logic
  useEffect(() => {
    let isMounted = true;
    let retryTimeout: NodeJS.Timeout | null = null;

    const checkApiHealth = async (isRetry = false) => {
      // Don't show loading message on initial check, only on retries
      if (isRetry) {
        console.log("🔄 Retrying API health check...");
      } else {
        console.log("🔍 Checking API health...");
      }

      try {
        const { apiService } = await import("@/services/api");
        
        // Use healthCheck with retry options
        const health = await apiService.healthCheck({
          retries: 5, // Try 5 times
          retryDelay: 1000, // Start with 1 second delay
          timeout: 8000 // 8 seconds timeout per request
        });

        if (!isMounted) return;

        console.log("✅ API connection successful:", health);
        setNotificationWithTimeout(
          "success",
          `Connected to backend (${health.device || "unknown device"})`
        );
      } catch (err) {
        if (!isMounted) return;

        console.error("⚠️ API connection failed after retries:", err);
        
        let errorMessage = "Unable to connect to backend server. Please ensure the backend is running on port 8003.";

        if (err instanceof Error) {
          try {
            // Try to parse as JSON if it's a JSON string
            const errorData = JSON.parse(err.message);
            if (errorData.baseUrl) {
              errorMessage = `Cannot connect to ${errorData.baseUrl}. Please ensure the backend is running on port 8003.`;
            } else if (errorData.error) {
              // Only show detailed error if it's not a network error (status 0)
              if (errorData.status !== 0) {
                errorMessage = errorData.error;
              } else {
                errorMessage = "Backend server is not responding. Please check if the server is running on port 8003.";
              }
            } else if (errorData.status === 0) {
              errorMessage = "Backend server is not responding. Please check if the server is running on port 8003.";
            }
          } catch (parseError) {
            // If not JSON, use the error message directly
            if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError") || err.message.includes("timeout")) {
              errorMessage = "Backend server is not responding. Please check if the server is running on port 8003.";
            } else {
              errorMessage = err.message || errorMessage;
            }
          }
        } else {
          errorMessage = String(err) || errorMessage;
        }

        // Only show error notification after all retries have failed
        setNotificationWithTimeout("error", errorMessage);
      }
    };

    // Add a small delay before first check to give backend time to start
    const initialDelay = setTimeout(() => {
      checkApiHealth(false);
    }, 500); // 500ms delay before first check

    return () => {
      isMounted = false;
      if (retryTimeout) {
        clearTimeout(retryTimeout);
      }
      clearTimeout(initialDelay);
    };
  }, []);

  // Cleanup effect to clear notification timeouts
  useEffect(() => {
    return () => {
      clearNotificationTimeout();
    };
  }, [clearNotificationTimeout]);

  // Throttle function for better performance
  const throttle = useCallback((func: Function, delay: number) => {
    let timeoutId: NodeJS.Timeout | null = null;
    let lastExecTime = 0;

    return (...args: any[]) => {
      const currentTime = Date.now();

      if (currentTime - lastExecTime > delay) {
        func(...args);
        lastExecTime = currentTime;
      } else {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          func(...args);
          lastExecTime = Date.now();
        }, delay - (currentTime - lastExecTime));
      }
    };
  }, []);

  // Resize handlers for the sidebar - optimized for performance
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const updateSidebarWidth = useCallback((e: MouseEvent) => {
    const newWidth = window.innerWidth - e.clientX;
    const minWidth = 280; // Minimum sidebar width
    const maxWidth = Math.min(600, window.innerWidth * 0.5); // Maximum 50% of screen

    const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
    setSidebarWidth(clampedWidth);
  }, []);

  // Throttled resize function for smoother performance
  const throttledResize = useMemo(
    () => throttle(updateSidebarWidth, 16), // ~60fps
    [throttle, updateSidebarWidth]
  );

  const handleResizeMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      // Use requestAnimationFrame for smooth resizing
      requestAnimationFrame(() => {
        throttledResize(e);
      });
    },
    [isResizing, throttledResize]
  );

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
    // Save the width to localStorage with a slight delay
    setTimeout(() => {
      if (typeof window !== "undefined") {
        localStorage.setItem(
          "artmancer-sidebar-width",
          sidebarWidth.toString()
        );
      }
    }, 100);
  }, [sidebarWidth]);

  // Save sidebar width to localStorage when it changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("artmancer-sidebar-width", sidebarWidth.toString());
    }
  }, [sidebarWidth]);

  // Mouse event listeners for resizing with passive listeners for better performance
  useEffect(() => {
    if (isResizing) {
      const handleMove = (e: MouseEvent | TouchEvent) => {
        const clientX = "touches" in e ? e.touches[0]?.clientX : e.clientX;
        if (clientX !== undefined) {
          handleResizeMove({ clientX } as MouseEvent);
        }
      };
      const handleEnd = () => handleResizeEnd();

      document.addEventListener("mousemove", handleMove as EventListener, {
        passive: true,
      });
      document.addEventListener("mouseup", handleEnd);
      document.addEventListener("touchmove", handleMove as EventListener, {
        passive: true,
      });
      document.addEventListener("touchend", handleEnd);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      // Prevent text selection during resize
      document.body.style.webkitUserSelect = "none";

      return () => {
        document.removeEventListener("mousemove", handleMove as EventListener);
        document.removeEventListener("mouseup", handleEnd);
        document.removeEventListener("touchmove", handleMove as EventListener);
        document.removeEventListener("touchend", handleEnd);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        document.body.style.webkitUserSelect = "";
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  return (
    <div className="min-h-screen max-h-screen bg-[var(--primary-bg)] text-[var(--text-primary)] flex flex-col dots-pattern-small overflow-hidden">
      {/* Header */}
       <Header
         onSummon={handleEdit}
         onEvaluate={handleEvaluate}
         isCustomizeOpen={isCustomizeOpen}
         onToggleCustomize={() => setIsCustomizeOpen(!isCustomizeOpen)}
        isGenerating={isGenerating}
        isEvaluating={isEvaluating}
        appMode={appMode}
         onAppModeChange={handleAppModeChange}
         aiTask={aiTask}
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
        {(isGenerating || isEvaluating) && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-40">
            <div className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg p-6 text-center min-w-[300px]">
              <div className="animate-spin w-8 h-8 border-4 border-[var(--primary-accent)] border-t-transparent rounded-full mx-auto mb-3"></div>
              <p className="text-[var(--text-primary)] font-medium">
                {isGenerating ? "Generating image..." : "Evaluating images..."}
              </p>
              {isEvaluating && evaluationProgress && (
                <div className="mt-4">
                  <div className="w-full bg-[var(--border-color)] rounded-full h-2 mb-2">
                    <div
                      className="bg-[var(--primary-accent)] h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${(evaluationProgress.current / evaluationProgress.total) * 100}%`,
                      }}
                    ></div>
                  </div>
                  <p className="text-[var(--text-secondary)] text-sm">
                    {evaluationProgress.current} / {evaluationProgress.total} pairs
                    {evaluationProgress.currentPair && (
                      <span className="block text-xs mt-1 opacity-75">
                        {evaluationProgress.currentPair}
                      </span>
                    )}
                  </p>
                </div>
              )}
              {!evaluationProgress && (
                <p className="text-[var(--text-secondary)] text-sm mt-1">
                  {isGenerating ? "This may take a few seconds" : "Please wait..."}
                </p>
              )}
              {isGenerating && (
                <button
                  onClick={cancelGeneration}
                  className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium transition-colors"
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
          isMaskDrawing={isMaskDrawing}
          maskBrushSize={maskBrushSize}
          isSmartMaskLoading={isSmartMaskLoading}
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
          appMode={appMode}
          evaluationImagePairs={evaluationImagePairs}
          evaluationDisplayLimit={evaluationDisplayLimit}
          onEvaluationDisplayLimitChange={setEvaluationDisplayLimit}
          onRemoveEvaluationPair={handleRemoveEvaluationPair}
        />

        {/* Help Box Component */}
        <HelpBox isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />

        {/* Right Side - Customize Panel */}
        <Sidebar
          isOpen={isCustomizeOpen}
          width={sidebarWidth}
          isResizing={isResizing}
          uploadedImage={uploadedImage}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          maskToolType={maskToolType}
          referenceImage={referenceImage}
          aiTask={aiTask}
          appMode={appMode}
          onImageUpload={handleImageUploadWrapper}
          onRemoveImage={handleRemoveImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
          onMaskToolTypeChange={setMaskToolType}
          maskHistoryIndex={maskHistoryIndex}
          maskHistoryLength={maskHistoryLength}
          onMaskUndo={undoMask}
          onMaskRedo={redoMask}
          hasMaskContent={hasMaskContent}
          enableSmartMasking={enableSmartMasking}
          isSmartMaskLoading={isSmartMaskLoading}
          onSmartMaskingChange={setEnableSmartMasking}
          onReferenceImageUpload={handleReferenceImageUpload}
          onRemoveReferenceImage={handleRemoveReferenceImage}
          onAiTaskChange={handleAiTaskChange}
          onResizeStart={handleResizeStart}
          onWidthChange={setSidebarWidth}
          originalImage={originalImage}
          modifiedImage={modifiedImageForComparison}
          onReturnToOriginal={handleReturnToOriginal}
          lastRequestId={lastRequestId}
          negativePrompt={negativePrompt}
          guidanceScale={guidanceScale}
          inferenceSteps={inferenceSteps}
          cfgScale={cfgScale}
          onNegativePromptChange={handleNegativePromptChange}
          onGuidanceScaleChange={handleGuidanceScaleChange}
          onInferenceStepsChange={handleInferenceStepsChange}
          onCfgScaleChange={handleCfgScaleChange}
          onWhiteBalance={handleWhiteBalance}
          whiteBalanceTemperature={whiteBalanceTemperature}
          whiteBalanceTint={whiteBalanceTint}
          onWhiteBalanceTemperatureChange={handleWhiteBalanceTemperatureChange}
          onWhiteBalanceTintChange={handleWhiteBalanceTintChange}
          // Evaluation mode props
          evaluationMode={evaluationMode}
          evaluationTask={evaluationTask}
          evaluationSingleOriginal={evaluationSingleOriginal}
          evaluationSingleTarget={evaluationSingleTarget}
          evaluationImagePairs={evaluationImagePairs}
          evaluationConditionalImages={evaluationConditionalImages}
          evaluationReferenceImage={evaluationReferenceImage}
          evaluationDisplayLimit={evaluationDisplayLimit}
          allowMultipleFolders={allowMultipleFolders}
          onEvaluationModeChange={handleEvaluationModeChange}
          onEvaluationTaskChange={handleEvaluationTaskChange}
          onEvaluationSingleOriginalUpload={
            handleEvaluationSingleOriginalUpload
          }
          onEvaluationSingleTargetUpload={handleEvaluationSingleTargetUpload}
          onEvaluationMultipleUpload={handleEvaluationMultipleUpload}
          onEvaluationOriginalFolderUpload={handleEvaluationOriginalFolderUpload}
          onEvaluationTargetFolderUpload={handleEvaluationTargetFolderUpload}
          onEvaluationConditionalImagesUpload={
            handleEvaluationConditionalImagesUpload
          }
          onEvaluationReferenceImageUpload={handleEvaluationReferenceImageUpload}
          onAllowMultipleFoldersChange={setAllowMultipleFolders}
          onEvaluationDisplayLimitChange={setEvaluationDisplayLimit}
          evaluationResults={evaluationResults}
          evaluationResponse={evaluationResponse}
          onExportEvaluationJSON={handleExportEvaluationJSON}
          onExportEvaluationCSV={handleExportEvaluationCSV}
        />
      </main>
    </div>
  );
}
