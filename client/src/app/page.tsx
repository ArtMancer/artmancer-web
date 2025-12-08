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
import type { InputQualityPreset, DebugInfo } from "@/services/api";
import { apiService } from "@/services/api";
import DebugPanel from "@/components/DebugPanel";
import ReferenceImageEditor from "@/components/ReferenceImageEditor";

export default function Home() {
  // Basic UI state
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  // Legacy error/success state - kept for backward compatibility with notification system
  const [, setError] = useState<string | null>(null);
  const [, setSuccess] = useState<string | null>(null);

  // Enhanced notification state
  const [notificationType, setNotificationType] =
    useState<NotificationType>("success");
  const [notificationMessage, setNotificationMessage] = useState<string>("");
  const [isNotificationVisible, setIsNotificationVisible] = useState(false);

  // App Mode state (Inference or Benchmark)
  const [appMode, setAppMode] = useState<"inference" | "benchmark">(
    "inference"
  );

  // AI Task state
  const [aiTask, setAiTask] = useState<
    "white-balance" | "object-insert" | "object-removal" | "evaluation"
  >("white-balance");
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  const [referenceMaskR, setReferenceMaskR] = useState<string | null>(null);
  
  // Saved mask states - lưu mask khi generation bắt đầu để không bị mất
  // savedMaskData: mask binary (đen trắng) để gửi API - được lưu nhưng không cần restore
  // savedMaskCanvasState: mask canvas UI (đỏ trong suốt) để restore mask brush sau generation
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [savedMaskData, setSavedMaskData] = useState<string | null>(null);
  const [savedMaskCanvasState, setSavedMaskCanvasState] = useState<string | null>(null);
  const [maskVisible, setMaskVisible] = useState(true);

  // Reference Image Editor state
  const [isRefEditorOpen, setIsRefEditorOpen] = useState(false);
  const [pendingRefImage, setPendingRefImage] = useState<string | null>(null);

  // Advanced options state
  const [negativePrompt, setNegativePrompt] = useState<string>("");
  const [seed, setSeed] = useState<number>(42); // Default seed: 42 (famous default)
  // Task-specific default parameters
  const getDefaultParametersForTask = useCallback((task: string) => {
    switch (task) {
      case "object-removal":
        return {
          guidanceScale: 7.0, // Strict for removal (higher in 1-10 range)
          inferenceSteps: 20, // Default 20 steps
          cfgScale: 4.0, // Keep default
        };
      case "object-insert":
        return {
          guidanceScale: 4.0, // Balanced for insertion (middle of 1-10 range)
          inferenceSteps: 20, // Default 20 steps
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
          inferenceSteps: 20,
          cfgScale: 4.0,
        };
    }
  }, []);

  const [guidanceScale, setGuidanceScale] = useState<number>(1.0);
  const [inferenceSteps, setInferenceSteps] = useState<number>(10); // Default steps
  const [cfgScale, setCfgScale] = useState<number>(4.0);
  const [inputQuality, setInputQuality] =
    useState<InputQualityPreset>("resized");
  const [customSquareSize, setCustomSquareSize] = useState<number>(1024);
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
  const [isApplyingQuality, setIsApplyingQuality] = useState(false);
  const qualityChangeRequestRef = useRef(0);

  // White balance state

  // Visualization request ID state
  const [lastRequestId, setLastRequestId] = useState<string | null>(null);

  // Debug panel state
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [isDebugPanelVisible, setIsDebugPanelVisible] = useState(true); // Always on by default

  // Auto-refine prompt state

  // Benchmark mode state
  const [benchmarkFolder, setBenchmarkFolder] = useState<string>("");
  const [benchmarkValidation, setBenchmarkValidation] = useState<{
    success: boolean;
    message: string;
    image_count: number;
    details?: Record<string, number>;
  } | null>(null);
  const [benchmarkSampleCount, setBenchmarkSampleCount] = useState<number>(0);
  const [benchmarkTask, setBenchmarkTask] = useState<
    "white-balance" | "object-insert" | "object-removal"
  >("object-removal");
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState<{
    current: number;
    total: number;
    currentImage?: string;
  } | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<any>(null);
  const [benchmarkPrompt, setBenchmarkPrompt] = useState<string>("");
  const [isValidatingBenchmark, setIsValidatingBenchmark] =
    useState<boolean>(false);

  // Legacy evaluation state (kept for backward compatibility, will be removed)
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
  } | null>(null);

  // Track step timing for ETR calculation
  const stepTimingRef = useRef<{ step: number; timestamp: number }[]>([]);

  // API integration
  const {
    generateImage,
    applyWhiteBalance,
    isGenerating,
    error: generationError,
    clearError,
    cancelGeneration,
  } = useImageGeneration();

  const {
    viewportZoom,
    imageContainerRef,
    containerRef,
    handleWheel,
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
          // Open editor instead of setting directly
          setPendingRefImage(e.target.result);
          setIsRefEditorOpen(true);
          setError(null);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRefEditorSubmit = async (
    processedImage: string,
    maskData?: string | null
  ) => {
    // Store both reference image and Mask R (for two-source mask workflow)
    setReferenceImage(processedImage);
    setReferenceMaskR(maskData || null);
    setPendingRefImage(null);
    setIsRefEditorOpen(false);
  };

  const handleRefEditorClose = () => {
    setPendingRefImage(null);
    setIsRefEditorOpen(false);
  };

  const handleRemoveReferenceImage = () => {
    setReferenceImage(null);
    setReferenceMaskR(null);
  };

  const handleEditReferenceImage = useCallback(() => {
    if (referenceImage) {
      // Open editor with current reference image
      setPendingRefImage(referenceImage);
      setIsRefEditorOpen(true);
    }
  }, [referenceImage]);

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
      if (appMode === "benchmark") {
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
        const remainingPairs = evaluationImagePairs.slice(
          originalImages.length
        );
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

  const handleEvaluate = async (prompt: string) => {
    // If in benchmark mode, run benchmark instead of evaluation
    if (appMode === "benchmark") {
      // Update benchmark prompt state
      setBenchmarkPrompt(prompt);
      // Run benchmark with the prompt from header
      await handleRunBenchmarkWithPrompt(prompt);
    } else {
      // Legacy evaluation mode
      await handleEvaluateImages(prompt);
    }
  };

  const handleRunBenchmarkWithPrompt = async (prompt: string) => {
    console.log("🚀 Running benchmark with prompt:", {
      hasFile: !!benchmarkFile,
      file: benchmarkFile?.name,
      validation: benchmarkValidation,
      prompt: prompt,
      task: benchmarkTask,
    });

    if (!benchmarkFile) {
      setNotificationWithTimeout(
        "error",
        "Please upload a benchmark ZIP file first"
      );
      return;
    }

    if (!benchmarkValidation?.success) {
      setNotificationWithTimeout(
        "error",
        "Please validate the benchmark ZIP file first. Make sure it contains input/, mask/, and groundtruth/ folders."
      );
      return;
    }

    if (!prompt || prompt.trim().length === 0) {
      setNotificationWithTimeout(
        "error",
        "Please enter a benchmark prompt in the header. This prompt will be used for all images."
      );
      return;
    }

    if (benchmarkTask !== "object-removal") {
      setNotificationWithTimeout(
        "error",
        `Task "${benchmarkTask}" is not yet implemented. Only Object Removal is available.`
      );
      return;
    }

    setIsRunningBenchmark(true);
    setBenchmarkProgress({
      current: 0,
      total: benchmarkValidation.image_count,
    });
    setBenchmarkResults(null);

    try {
      const { apiService } = await import("@/services/api");
      // benchmarkFile can be File (ZIP) or File[] (folder)
      const result = await apiService.runBenchmark(
        benchmarkFile as File | File[],
        {
          task_type: benchmarkTask,
          prompt: prompt.trim(), // Use prompt from header for ALL images
          sample_count: benchmarkSampleCount,
          num_inference_steps: inferenceSteps,
          guidance_scale: guidanceScale,
          // Only send true_cfg_scale if negative prompt is provided
          true_cfg_scale:
            negativePrompt.trim().length > 0 ? cfgScale : undefined,
          negative_prompt: negativePrompt || undefined,
          input_quality: inputQuality,
        }
      );

      setBenchmarkResults(result);
      setNotificationWithTimeout(
        "success",
        `Benchmark completed! Processed ${
          result.summary?.successful_evaluations || 0
        } images`
      );
    } catch (error: any) {
      setNotificationWithTimeout("error", `Benchmark failed: ${error.message}`);
    } finally {
      setIsRunningBenchmark(false);
      setBenchmarkProgress(null);
    }
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
      const totalPairs =
        evaluationMode === "single"
          ? 1
          : evaluationImagePairs.filter((pair) => pair.original && pair.target)
              .length;

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
            const prompts = prompt.split("\n").filter((p) => p.trim());
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
    link.download = `evaluation-results-${
      new Date().toISOString().split("T")[0]
    }.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export evaluation results as CSV
  const handleExportEvaluationCSV = () => {
    if (!evaluationResults || evaluationResults.length === 0) return;

    const headers = [
      "Filename",
      "PSNR (dB)",
      "SSIM",
      "LPIPS",
      "ΔE00",
      "Time (s)",
      "Status",
    ];
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
    const avgPsnr =
      evaluationResults
        .filter((r) => r.success && r.metrics?.psnr)
        .reduce((sum, r) => sum + (r.metrics.psnr || 0), 0) /
        evaluationResults.filter((r) => r.success && r.metrics?.psnr).length ||
      0;
    const avgSsim =
      evaluationResults
        .filter((r) => r.success && r.metrics?.ssim)
        .reduce((sum, r) => sum + (r.metrics.ssim || 0), 0) /
        evaluationResults.filter((r) => r.success && r.metrics?.ssim).length ||
      0;

    rows.push([]);
    rows.push(["Summary", "", "", "", "", "", ""]);
    rows.push([
      "Average",
      isNaN(avgPsnr) ? "N/A" : avgPsnr.toFixed(3),
      isNaN(avgSsim) ? "N/A" : avgSsim.toFixed(4),
      "",
      "",
      evaluationResponse?.total_evaluation_time?.toFixed(2) || "N/A",
      `${evaluationResponse?.successful_evaluations || 0}/${
        evaluationResponse?.total_pairs || 0
      }`,
    ]);

    const csvContent = [headers, ...rows]
      .map((row) => row.map((cell) => `"${cell}"`).join(","))
      .join("\n");

    const dataBlob = new Blob([csvContent], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `evaluation-results-${
      new Date().toISOString().split("T")[0]
    }.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Benchmark handlers
  const [benchmarkFile, setBenchmarkFile] = useState<File | null>(null);

  const handleBenchmarkFolderChange = (path: string) => {
    setBenchmarkFolder(path);
  };

  const handleBenchmarkFolderValidate = async (
    fileOrPath: File | File[] | string
  ) => {
    setIsValidatingBenchmark(true);
    try {
      // Handle folder upload (File[])
      if (Array.isArray(fileOrPath)) {
        const files = fileOrPath;
        console.log("📁 Validating benchmark folder:", files.length, "files");

        const { apiService } = await import("@/services/api");
        const validation = await apiService.validateBenchmarkFolder(files);

        console.log("✅ Validation result:", validation);

        setBenchmarkValidation(validation);
        if (validation.success) {
          // Store files for later use
          setBenchmarkFile(files as any); // Store File[] as marker
          const folderName =
            files[0]?.webkitRelativePath?.split("/")[0] || "folder";
          setBenchmarkFolder(folderName);
          setNotificationWithTimeout("success", validation.message);
        } else {
          setNotificationWithTimeout("error", validation.message);
          setBenchmarkFile(null);
          setBenchmarkFolder("");
        }
        return;
      }

      // Handle ZIP file upload (File)
      if (fileOrPath instanceof File) {
        const file = fileOrPath;
        // Set file and folder name first
        setBenchmarkFile(file);
        setBenchmarkFolder(file.name);
        // Clear previous validation
        setBenchmarkValidation(null);

        console.log(
          "📦 Validating benchmark ZIP file:",
          file.name,
          file.size,
          "bytes"
        );

        const { apiService } = await import("@/services/api");
        const validation = await apiService.validateBenchmarkFolder(file);

        console.log("✅ Validation result:", validation);

        setBenchmarkValidation(validation);
        if (validation.success) {
          setNotificationWithTimeout("success", validation.message);
        } else {
          setNotificationWithTimeout("error", validation.message);
          // Clear file if validation failed
          setBenchmarkFile(null);
          setBenchmarkFolder("");
        }
        return;
      }

      // Legacy support - string path (should not happen with new UI)
      setNotificationWithTimeout("error", "Please upload a ZIP file or folder");
      setBenchmarkFile(null);
      setBenchmarkFolder("");
      setBenchmarkValidation(null);
    } catch (error: any) {
      console.error("❌ Validation error:", error);
      setNotificationWithTimeout(
        "error",
        `Validation failed: ${error.message}`
      );
      setBenchmarkValidation({
        success: false,
        message: `❌ Validation failed: ${error.message}`,
        image_count: 0,
      });
      setBenchmarkFile(null);
      setBenchmarkFolder("");
    } finally {
      setIsValidatingBenchmark(false);
    }
  };

  const handleBenchmarkSampleCountChange = (count: number) => {
    // Validate and normalize count
    const totalImages = benchmarkValidation?.image_count || 0;

    let normalizedCount = count;

    // Handle edge cases
    if (isNaN(normalizedCount) || normalizedCount < 0) {
      normalizedCount = 0; // Fallback to all images
      if (count < 0) {
        setNotificationWithTimeout(
          "warning",
          `Invalid sample count (${count}). Using 0 to process all images.`
        );
      }
    } else if (totalImages > 0 && normalizedCount > totalImages) {
      normalizedCount = totalImages; // Fallback to all available images
      setNotificationWithTimeout(
        "warning",
        `Requested ${count} samples but only ${totalImages} available. Will process all ${totalImages} images.`
      );
    }

    setBenchmarkSampleCount(normalizedCount);
    console.log(
      `🔢 Sample count changed: ${count} → ${normalizedCount} (total: ${totalImages})`
    );
  };

  const handleBenchmarkTaskChange = (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => {
    setBenchmarkTask(task);
  };

  const handleBenchmarkPromptChange = (prompt: string) => {
    setBenchmarkPrompt(prompt);
  };

  const handleRunBenchmark = async () => {
    if (!benchmarkFile || !benchmarkValidation?.success) {
      setNotificationWithTimeout(
        "error",
        "Please upload and validate a benchmark dataset (ZIP file or folder) first"
      );
      return;
    }

    if (!benchmarkPrompt || benchmarkPrompt.trim().length === 0) {
      setNotificationWithTimeout(
        "error",
        "Please enter a benchmark prompt. This prompt will be used for all images."
      );
      return;
    }

    if (benchmarkTask !== "object-removal") {
      setNotificationWithTimeout(
        "error",
        `Task "${benchmarkTask}" is not yet implemented. Only Object Removal is available.`
      );
      return;
    }

    setIsRunningBenchmark(true);
    setBenchmarkProgress({
      current: 0,
      total: benchmarkValidation.image_count,
    });
    setBenchmarkResults(null);

    try {
      const { apiService } = await import("@/services/api");
      // benchmarkFile can be File (ZIP) or File[] (folder)
      const result = await apiService.runBenchmark(
        benchmarkFile as File | File[],
        {
          task_type: benchmarkTask,
          prompt: benchmarkPrompt.trim(), // Use the entered prompt for ALL images
          sample_count: benchmarkSampleCount,
          num_inference_steps: inferenceSteps,
          guidance_scale: guidanceScale,
          // Only send true_cfg_scale if negative prompt is provided
          true_cfg_scale:
            negativePrompt.trim().length > 0 ? cfgScale : undefined,
          negative_prompt: negativePrompt || undefined,
          input_quality: inputQuality,
        }
      );

      setBenchmarkResults(result);
      setNotificationWithTimeout(
        "success",
        `Benchmark completed! Processed ${
          result.summary?.successful_evaluations || 0
        } images`
      );
    } catch (error: any) {
      setNotificationWithTimeout("error", `Benchmark failed: ${error.message}`);
    } finally {
      setIsRunningBenchmark(false);
      setBenchmarkProgress(null);
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

    // Legacy support: "evaluation" task maps to benchmark mode
    // Note: "evaluation" is kept in aiTask type for backward compatibility
    if (task === "evaluation") {
      setAppMode("benchmark");
    } else if (appMode === "benchmark") {
      // Switch back to inference mode if selecting other tasks (not evaluation)
      setAppMode("inference");
    }
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
    saveMaskCanvasState,
    restoreMaskCanvasState,
    enableSmartMasking,
    setEnableSmartMasking,
    smartMaskModelType,
    setSmartMaskModelType,
    isSmartMaskLoading,
    edgeOverlayCanvasRef,
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

  // Reset trạng thái hiển thị mask và batch results khi thay ảnh gốc/ảnh generate mới
  useEffect(() => {
    setMaskVisible(true);
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
  };

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

  const applyInputQualityPreset = useCallback(
    async (
      quality: InputQualityPreset,
      options?: {
        sourceData?: string;
        sourceDimensions?: { width: number; height: number };
        silent?: boolean;
        squareSize?: number; // Custom square size override
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
    ]
  );

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
    setIsApplyingQuality(false);
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
    [applyInputQualityPreset, baseImageData, inputQuality]
  );

  const handleCustomSquareSizeChange = useCallback(
    async (size: number) => {
      setCustomSquareSize(size);
      // Re-apply quality preset with new size if in resized mode
      if (inputQuality === "resized" && baseImageData) {
        await applyInputQualityPreset("resized", { squareSize: size });
      }
    },
    [applyInputQualityPreset, baseImageData, inputQuality]
  );

  // Handle white balance
  const handleWhiteBalance = async (
    method: "auto" | "manual" | "ai" = "auto"
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

      const result = await applyWhiteBalance(file, method);

      if (result && result.corrected_image) {
        const imageData = `data:image/png;base64,${result.corrected_image}`;
        setUploadedImage(imageData);
        setModifiedImage(imageData);
        setModifiedImageForComparison(imageData);

        // Add to history
        addToHistory(imageData);

        const wbImg = new Image();
        wbImg.onload = () => {
          setBaseImage(imageData, {
            width: wbImg.width,
            height: wbImg.height,
          });
        };
        wbImg.src = imageData;

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
  const handleEdit = async (prompt: string): Promise<void> => {
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
          console.log("💾 [handleEdit] Saving mask canvas state (UI mask brush)");
          setSavedMaskCanvasState(canvasState);
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
      });

      // Get settings from state
      // Note: width and height are not set - backend will use original image size automatically
      const settings = {
        num_inference_steps: inferenceSteps,
        guidance_scale: guidanceScale,
        // Only send true_cfg_scale if negative prompt is provided
        true_cfg_scale: negativePrompt.trim().length > 0 ? cfgScale : undefined,
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

      // Progress callback to update from SSE events
      const progressCallback = (progress: {
        current_step?: number;
        total_steps?: number;
        status: string;
        progress: number;
        loading_message?: string;
      }) => {
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
            progress.status === "initializing_a100" ||
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

          // Reset current to 0 when processing starts (if no step info yet)
          if (
            status === "processing" &&
            (progress.current_step === null ||
              progress.current_step === undefined)
          ) {
            current = 0;
          }

          if (
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
          } else if (progress.progress !== undefined) {
            // For loading phase, use progress percentage
            // Map progress (0.0-1.0) to current step based on total
            current = Math.floor(progress.progress * total);
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
        progressCallback
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
      setGenerationProgress(null);

      if (result && result.image) {
        const imageData = `data:image/png;base64,${result.image}`;

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

        // Restore mask canvas state (UI mask brush) sau khi generation hoàn thành
        // Đợi tất cả resizeCanvas hoàn thành (resizeCanvas chạy ở 0ms, 50ms, 200ms)
        if (savedMaskCanvasState) {
          console.log("🔄 [handleEdit] Restoring mask canvas state after generation", {
            savedStateLength: savedMaskCanvasState.length,
            imageDimensions: imageDimensions ? `${imageDimensions.width}x${imageDimensions.height}` : 'null'
          });
          // Đợi sau khi tất cả resizeCanvas hoàn thành (sau 250ms)
          // Sau đó restore một lần, nếu không được thì xóa savedMaskCanvasState
          setTimeout(() => {
            const restored = restoreMaskCanvasState(savedMaskCanvasState);
            if (restored) {
              console.log("✅ [handleEdit] Mask canvas restored successfully");
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
      setGenerationProgress(null);
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
  // Removed automatic backend health check notification
  // Backend status is now managed by ServerContext and displayed in Header
  // This reduces redundant API calls and improves cost optimization

  // Cleanup effect to clear notification timeouts
  useEffect(() => {
    return () => {
      clearNotificationTimeout();
    };
  }, [clearNotificationTimeout]);

  // Throttle function for better performance
  const throttle = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    <T extends (...args: any[]) => void>(func: T, delay: number): T => {
      let timeoutId: NodeJS.Timeout | null = null;
      let lastExecTime = 0;

      return ((...args: Parameters<T>) => {
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
      }) as T;
    },
    []
  );

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
    () => throttle(updateSidebarWidth as (e: MouseEvent) => void, 16), // ~60fps
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
        onSummon={appMode === "benchmark" ? handleEvaluate : handleEdit}
        isCustomizeOpen={isCustomizeOpen}
        onToggleCustomize={() => setIsCustomizeOpen(!isCustomizeOpen)}
        isGenerating={isGenerating || isEvaluating}
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
        {(isGenerating || isEvaluating) && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-40">
            <div className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg p-6 text-center min-w-[300px]">
              <div className="animate-spin w-8 h-8 border-4 border-[var(--primary-accent)] border-t-transparent rounded-full mx-auto mb-3"></div>
              <p className="text-[var(--text-primary)] font-medium">
                {isGenerating ? "Generating image..." : "Evaluating images..."}
              </p>
              {/* Evaluation Progress */}
              {isEvaluating && evaluationProgress && (
                <div className="mt-4">
                  <div className="w-full bg-[var(--border-color)] rounded-full h-2 mb-2">
                    <div
                      className="bg-[var(--primary-accent)] h-2 rounded-full transition-all duration-300"
                      style={{
                        width: `${
                          (evaluationProgress.current /
                            evaluationProgress.total) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <p className="text-[var(--text-secondary)] text-sm">
                    {evaluationProgress.current} / {evaluationProgress.total}{" "}
                    pairs
                    {evaluationProgress.currentPair && (
                      <span className="block text-xs mt-1 opacity-75">
                        {evaluationProgress.currentPair}
                      </span>
                    )}
                  </p>
                </div>
              )}
              {/* Generation Progress */}
              {isGenerating && generationProgress && (
                <div className="mt-4">
                  <div className="w-full bg-[var(--border-color)] rounded-full h-2 mb-2 overflow-hidden">
                    <div
                      className="bg-[var(--primary-accent)] h-2 rounded-full transition-all duration-500 ease-out"
                      style={{
                        width: `${
                          ((generationProgress.current ?? 0) /
                            generationProgress.total) *
                          100
                        }%`,
                      }}
                    ></div>
                  </div>
                  <p className="text-[var(--text-secondary)] text-sm">
                    {generationProgress.status === "loading_pipeline"
                      ? generationProgress.loadingMessage || "Loading model..."
                      : generationProgress.status === "processing"
                      ? `Processing: ${generationProgress.current ?? 0} / ${
                          generationProgress.total
                        } steps`
                      : `Step ${generationProgress.current ?? 0} / ${
                          generationProgress.total
                        }`}
                    {generationProgress.total > 0 && (
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
                  </p>
                </div>
              )}
              {/* Fallback message when no progress available */}
              {!evaluationProgress && !generationProgress && (
                <p className="text-[var(--text-secondary)] text-sm mt-1">
                  {isGenerating
                    ? "This may take a few seconds"
                    : "Please wait..."}
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
          imageDimensions={imageDimensions}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          maskToolType={maskToolType}
          maskVisible={maskVisible}
          referenceImage={referenceImage}
          aiTask={aiTask}
          appMode={appMode}
          inputQuality={inputQuality}
          customSquareSize={customSquareSize}
          isApplyingQuality={isApplyingQuality}
          onImageUpload={handleImageUploadWrapper}
          onRemoveImage={handleRemoveImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
          onMaskToolTypeChange={setMaskToolType}
          onToggleMaskVisible={handleToggleMaskVisible}
          maskHistoryIndex={maskHistoryIndex}
          maskHistoryLength={maskHistoryLength}
          onMaskUndo={undoMask}
          onMaskRedo={redoMask}
          hasMaskContent={hasMaskContent}
          enableSmartMasking={enableSmartMasking}
          isSmartMaskLoading={isSmartMaskLoading}
          onSmartMaskingChange={setEnableSmartMasking}
          smartMaskModelType={smartMaskModelType}
          onSmartMaskModelTypeChange={setSmartMaskModelType}
          borderAdjustment={borderAdjustment}
          onBorderAdjustmentChange={handleBorderAdjustmentChange}
          onReferenceImageUpload={handleReferenceImageUpload}
          onRemoveReferenceImage={handleRemoveReferenceImage}
          onEditReferenceImage={handleEditReferenceImage}
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
          seed={seed}
          onSeedChange={handleSeedChange}
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
          onEvaluationOriginalFolderUpload={
            handleEvaluationOriginalFolderUpload
          }
          onEvaluationTargetFolderUpload={handleEvaluationTargetFolderUpload}
          onEvaluationConditionalImagesUpload={
            handleEvaluationConditionalImagesUpload
          }
          onEvaluationReferenceImageUpload={
            handleEvaluationReferenceImageUpload
          }
          onAllowMultipleFoldersChange={setAllowMultipleFolders}
          onEvaluationDisplayLimitChange={setEvaluationDisplayLimit}
          evaluationResults={evaluationResults}
          evaluationResponse={evaluationResponse}
          onExportEvaluationJSON={handleExportEvaluationJSON}
          onExportEvaluationCSV={handleExportEvaluationCSV}
          onInputQualityChange={handleInputQualityChange}
          onCustomSquareSizeChange={handleCustomSquareSizeChange}
          // Debug panel props
          isDebugPanelVisible={isDebugPanelVisible}
          onDebugPanelVisibilityChange={setIsDebugPanelVisible}
          // Benchmark mode props
          benchmarkFolder={benchmarkFolder}
          benchmarkValidation={benchmarkValidation}
          isValidatingBenchmark={isValidatingBenchmark}
          benchmarkSampleCount={benchmarkSampleCount}
          benchmarkTask={benchmarkTask}
          isRunningBenchmark={isRunningBenchmark}
          benchmarkProgress={benchmarkProgress}
          benchmarkResults={benchmarkResults}
          onBenchmarkFolderChange={handleBenchmarkFolderChange}
          onBenchmarkFolderValidate={handleBenchmarkFolderValidate}
          onBenchmarkSampleCountChange={handleBenchmarkSampleCountChange}
          onBenchmarkTaskChange={handleBenchmarkTaskChange}
          benchmarkPrompt={benchmarkPrompt}
          onBenchmarkPromptChange={handleBenchmarkPromptChange}
          onRunBenchmark={handleRunBenchmark}
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
        />
      )}
    </div>
  );
}
