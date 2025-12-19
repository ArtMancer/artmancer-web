import { useState, useCallback, useRef, useEffect } from 'react';
import { apiService, GenerationRequest, GenerationResponse, ModelSettings, WhiteBalanceRequest, WhiteBalanceResponse, TaskType } from '@/services/api';
import useImageFetcher from '@/hooks/useImageFetcher';

// Global singleton to track active generation and prevent duplicate calls
// This is necessary because React dev mode can mount components twice
let activeGenerationController: AbortController | null = null;

// Map UI task types to backend task types
const mapUITaskTypeToBackend = (uiTaskType: "white-balance" | "object-insert" | "object-removal"): TaskType => {
  switch (uiTaskType) {
    case "object-insert":
      return "insertion";
    case "object-removal":
      return "removal";
    case "white-balance":
      return "white-balance";
    default:
      return "removal"; // Default fallback
  }
};

// Validate task type
const validateTaskType = (taskType: TaskType): void => {
  const validTypes: TaskType[] = ["insertion", "removal", "white-balance"];
  if (!validTypes.includes(taskType)) {
    throw new Error(`Invalid task_type: ${taskType}. Must be one of: ${validTypes.join(", ")}`);
  }
};

export function useImageGeneration() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastGeneration, setLastGeneration] = useState<GenerationResponse | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentTaskIdRef = useRef<string | null>(null); // Track task_id for async generation
  const isGeneratingRef = useRef<boolean>(false); // Ref to track generation state across renders

  // Dùng chung hook fetch ảnh blob → Object URL (tự cleanup)
  const { fetchImage } = useImageFetcher();

  const generateImage = useCallback(async (
    prompt: string,
    inputImage: string,
    maskImage: string,
    settings?: ModelSettings,
    referenceImage?: string | null,
    referenceMaskR?: string | null,
    taskType?: "white-balance" | "object-insert" | "object-removal",
    onProgress?: (progress: { current_step?: number; total_steps?: number; status: string; progress: number }) => void,
    maskToolType?: "brush" | "box",
    enableMaeRefinement?: boolean,
    enableDebug?: boolean
  ): Promise<GenerationResponse | null> => {
    // 🔍 DEBUG: Log call stack to see who's calling this
    console.log("🎯 [useImageGeneration] generateImage called", {
      timestamp: new Date().toISOString(),
      isGeneratingRef: isGeneratingRef.current,
      activeController: !!activeGenerationController,
      stackTrace: new Error().stack?.split('\n').slice(2, 5).join('\n'),
    });

    // 🛡️ CRITICAL: Synchronous lock to prevent race conditions
    // Check AND set in single synchronous operation
    if (isGeneratingRef.current || activeGenerationController) {
      console.error("🚫 [useImageGeneration] BLOCKED: Already generating!", {
        isGeneratingRef: isGeneratingRef.current,
        hasActiveController: !!activeGenerationController,
      });
      return null;
    }
    
    // IMMEDIATELY mark as generating (synchronous, no gap for race condition)
    isGeneratingRef.current = true;

    // Prompt is optional for white balance
    if (taskType !== "white-balance" && !prompt.trim()) {
      setError('Please enter a prompt');
      return null;
    }

    if (!inputImage) {
      setError('Please upload an image to edit');
      return null;
    }

    // Mask is not required for white balance
    if (taskType !== "white-balance" && !maskImage) {
      setError('Please create a mask for the area to edit');
      return null;
    }

    // At this point, lock is acquired (isGeneratingRef.current = true)
    setIsGenerating(true);
    setError(null);

    // Create new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    activeGenerationController = abortController; // Store globally
    
    console.log("✅ [useImageGeneration] Lock acquired, generation starting");

    try {
      // Convert data URL to base64 if needed
      const base64InputImage = inputImage.startsWith('data:')
        ? inputImage.split(',')[1]
        : inputImage;

      // Convert mask_image to conditional_images format (backend expects conditional_images[0] = mask)
      const base64MaskImage = maskImage.startsWith('data:')
        ? maskImage.split(',')[1]
        : maskImage;

      // Map UI task type to backend task type
      const backendTaskType: TaskType | undefined = taskType 
        ? mapUITaskTypeToBackend(taskType)
        : undefined;

      // Validate task type if provided
      if (backendTaskType) {
        validateTaskType(backendTaskType);
      }

      // Build conditional_images array in correct order:
      // [0] = mask (required for insertion/removal, optional for white-balance)
      // [1+] = additional conditional images (if any in the future)
      const conditional_images: string[] = [];
      
      // For white-balance, mask is optional (backend will create full white mask if not provided)
      // For insertion/removal, mask is required
      if (backendTaskType !== "white-balance" || base64MaskImage) {
        if (base64MaskImage) {
          conditional_images.push(base64MaskImage); // [0] = mask
        }
      }

      // Convert reference image to base64 if provided (for insertion task)
      // IMPORTANT: Use reference image directly without any processing
      let base64ReferenceImage: string | undefined = undefined;
      if (referenceImage) {
        base64ReferenceImage = referenceImage.startsWith('data:')
          ? referenceImage.split(',')[1]
          : referenceImage;
      }

      // Convert reference mask R to base64 if provided (for two-source mask workflow)
      let base64ReferenceMaskR: string | undefined = undefined;
      if (referenceMaskR) {
        base64ReferenceMaskR = referenceMaskR.startsWith('data:')
          ? referenceMaskR.split(',')[1]
          : referenceMaskR;
      }

      const request: GenerationRequest = {
        prompt: prompt.trim(),
        input_image: base64InputImage,
        conditional_images: conditional_images.length > 0 ? conditional_images : undefined,
        // Reference image for insertion task - use original image directly, no processing
        reference_image: base64ReferenceImage,
        // Reference mask R for two-source mask workflow (Mask R - object shape)
        reference_mask_R: base64ReferenceMaskR,
        // Legacy fields kept for backward compatibility but not used by backend
        mask_image: base64MaskImage,
        num_inference_steps: settings?.num_inference_steps,
        guidance_scale: settings?.guidance_scale,
        true_cfg_scale: settings?.true_cfg_scale,
        negative_prompt: settings?.negative_prompt,
        seed: settings?.generator_seed,
        width: settings?.width,
        height: settings?.height,
        task_type: backendTaskType, // Use backend format: "insertion", "removal", or "white-balance"
        input_quality: settings?.input_quality,
        // Low-end optimization flags
        enable_flowmatch_scheduler: settings?.enable_flowmatch_scheduler,
        mask_tool_type: maskToolType, // Mask creation tool type: "brush" or "box"
        enable_mae_refinement: enableMaeRefinement, // Enable MAE refinement (default: true)
        enable_debug: enableDebug, // Enable debug mode (default: false)
      };

      // Debug logging
      console.log("🔍 [Frontend API] Request details:", {
        hasConditionalImages: !!request.conditional_images,
        conditionalImagesCount: request.conditional_images?.length || 0,
        taskType: request.task_type,
        hasMask: !!request.mask_image,
        hasReferenceImage: !!request.reference_image,
        referenceImageLength: request.reference_image?.length || 0,
        hasReferenceMaskR: !!request.reference_mask_R,
        referenceMaskRLength: request.reference_mask_R?.length || 0,
        usingTwoSourceMasks: !!(request.reference_image && request.reference_mask_R),
      });

      // Two-step flow: 
      // 1. POST /api/generate/async → get task_id
      // 2. GET /api/generate/stream/{task_id} → SSE stream for progress
      const baseUrl = apiService.getBaseUrl();

      console.log("🌐 [useImageGeneration] Submitting generation job...");

      // Step 1: Submit job and get task_id
      const submitResponse = await fetch(`${baseUrl}/api/generate/async`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
        signal: abortController.signal,
      });

      if (!submitResponse.ok) {
        const errorText = await submitResponse.text();
        throw new Error(
          JSON.stringify({
            status: submitResponse.status,
            error: `HTTP ${submitResponse.status}: ${errorText || submitResponse.statusText || "Submit failed"}`,
            endpoint: "/api/generate/async",
          })
        );
      }

      const submitResult = await submitResponse.json() as { task_id?: string; status?: string; message?: string };
      const taskIdFromSubmit = submitResult.task_id;

      if (!taskIdFromSubmit) {
        throw new Error(
          JSON.stringify({
            status: 500,
            error: "No task_id returned from /api/generate/async",
            endpoint: "/api/generate/async",
          })
        );
      }

      currentTaskIdRef.current = taskIdFromSubmit;
      console.log("📝 [useImageGeneration] Got task_id:", taskIdFromSubmit);

      // Step 2: Connect to SSE stream for progress updates
      const streamUrl = `${baseUrl}/api/generate/stream/${taskIdFromSubmit}`;
      console.log("🌐 [useImageGeneration] Connecting to SSE stream:", streamUrl, {
        taskId: taskIdFromSubmit,
        timestamp: new Date().toISOString(),
        abortSignal: abortController.signal,
      });

      const streamResponse = await fetch(streamUrl, {
        method: "GET",
        headers: {
          "Accept": "text/event-stream",
        },
        signal: abortController.signal,
      });

      if (!streamResponse.ok || !streamResponse.body) {
        throw new Error(
          JSON.stringify({
            status: streamResponse.status,
            error: `HTTP ${streamResponse.status}: ${streamResponse.statusText || "Stream failed"}`,
            endpoint: `/api/generate/stream/${taskIdFromSubmit}`,
          })
        );
      }

      const reader = streamResponse.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Read SSE stream with persistent buffer
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let separatorIndex = buffer.indexOf("\n\n");
        while (separatorIndex !== -1) {
          const rawEvent = buffer.slice(0, separatorIndex).trim();
          buffer = buffer.slice(separatorIndex + 2);

          if (rawEvent.startsWith("data:")) {
            const jsonStr = rawEvent.replace(/^data:\s*/, "");
            try {
              const data = JSON.parse(jsonStr) as {
                task_id?: string;
                status?: string;
                progress?: number;
                current_step?: number | null;
                total_steps?: number | null;
                error?: string | null;
                loading_message?: string | null;
              };

              if (onProgress && data.status) {
                onProgress({
                  status: data.status,
                  progress: data.progress ?? 0,
                  current_step: data.current_step ?? undefined,
                  total_steps: data.total_steps ?? undefined,
                });
              }

              if (data.status === "error" || data.status === "cancelled") {
                throw new Error(
                  JSON.stringify({
                    status: 500,
                    error:
                      data.error ||
                      `Generation ${data.status === "cancelled" ? "cancelled" : "failed"}`,
                    endpoint: `/api/generate/stream/${taskIdFromSubmit}`,
                  })
                );
              }

              if (data.status === "done") {
                // Stream is finished; break outer loops
                separatorIndex = -1;
                buffer = "";
                break;
              }
            } catch (parseErr) {
              console.error("❌ [useImageGeneration] Failed to parse SSE event:", {
                error: parseErr,
                raw: jsonStr,
              });
            }
          }

          separatorIndex = buffer.indexOf("\n\n");
        }
      }

      const taskIdFromStream = taskIdFromSubmit;

      // Sau khi SSE hoàn tất, fetch ảnh + debug info SONG SONG để tăng tốc
      console.log(
        "📥 [useImageGeneration] Fetching final result (image + debug) for task_id:",
        taskIdFromStream
      );
      
      const [imageObjectUrl, finalResultDebug] = await Promise.all([
        fetchImage({
          url: `${baseUrl}/api/generate/result-image/${taskIdFromStream}`,
        }),
        apiService.getGenerationResult(taskIdFromStream),
      ]);

      if (!imageObjectUrl) {
        throw new Error(
          JSON.stringify({
            status: 500,
            error: "Failed to fetch binary image result",
            endpoint: `/api/generate/result-image/${taskIdFromStream}`,
          })
        );
      }

      const wrappedResponse: GenerationResponse = {
        success: true,
        // image giờ là Object URL (blob:...) dùng trực tiếp cho <img src>
        image: imageObjectUrl,
        generation_time: 0,
        model_used: "qwen-image-edit",
        parameters_used: {},
        request_id: taskIdFromStream,
        debug_info: finalResultDebug.debug_info,
      };

      setLastGeneration(wrappedResponse);
      console.log(
        "✅ [useImageGeneration] Generation completed successfully, clearing task_id"
      );
      currentTaskIdRef.current = null;
      // Clear global controller on success
      if (activeGenerationController === abortController) {
        activeGenerationController = null;
      }

      return wrappedResponse;
    } catch (err) {
      // Check if request was cancelled
      let errorMessage = 'Failed to generate image';
      let isCancelled = false;

      try {
        // Handle AbortError (user cancellation) explicitly
        if (err instanceof DOMException && err.name === "AbortError") {
          isCancelled = true;
          errorMessage = "Generation cancelled";
          console.log("🚫 [useImageGeneration] Request aborted by user");
          return null;
        }

        // Try to parse error message as JSON
        const errorMessageStr = (err as Error).message || String(err);
        
        // Define error data type
        interface ParsedErrorData {
          error_type?: string;
          error?: string;
          status?: number;
          detail?: string | { error?: string };
        }
        
        let errorData: ParsedErrorData | null = null;
        
        try {
          const parsed = JSON.parse(errorMessageStr);
          errorData = parsed as ParsedErrorData;
        } catch {
          // If not JSON, treat as plain error message
          errorMessage = errorMessageStr;
        }

        if (errorData) {
          if (errorData.error_type === 'cancelled' || errorData.error === 'Request cancelled') {
            isCancelled = true;
            errorMessage = 'Generation cancelled';
            // Don't set error for cancelled requests
            return null;
          } else if (errorData.status === 0) {
            // Network error
            errorMessage =
              errorData.error ||
              'Unable to connect to API server. Please check your connection and API endpoint configuration.';
        } else if (errorData.status === 503) {
          // Service unavailable
          errorMessage = errorData.error || 'Service temporarily unavailable. Please try again in a moment.';
        } else if (errorData.status === 500) {
          // Server error - check for OOM errors
          const errorText = errorData.error || '';
          if (errorText.toLowerCase().includes('out of memory') || 
              errorText.toLowerCase().includes('cuda out of memory') ||
              errorText.toLowerCase().includes('oom')) {
            errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
          } else {
            errorMessage = errorText || 'Server error occurred. Please check the server logs and try again.';
          }
        } else if (errorData.error) {
          // API error with message - check for OOM
          const errorText = typeof errorData.error === 'string' ? errorData.error : String(errorData.error);
          if (errorText.toLowerCase().includes('out of memory') || 
              errorText.toLowerCase().includes('cuda out of memory') ||
              errorText.toLowerCase().includes('oom')) {
            errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
          } else {
            errorMessage = errorText;
          }
        } else if (errorData.detail) {
          // FastAPI error format - handle both string and object
          let detailText: string;
          if (typeof errorData.detail === 'string') {
            detailText = errorData.detail;
          } else if (typeof errorData.detail === 'object' && errorData.detail?.error) {
            detailText = errorData.detail.error;
          } else {
            detailText = String(errorData.detail);
          }
          // Check for OOM in detail
          if (detailText.toLowerCase().includes('out of memory') || 
              detailText.toLowerCase().includes('cuda out of memory') ||
              detailText.toLowerCase().includes('oom')) {
            errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
          } else {
            errorMessage = detailText;
          }
        }

        console.error('Generation error details:', errorData);
        }
      } catch (fallbackErr) {
        // Fallback for any unexpected errors
        console.error('Error handling failed:', fallbackErr);
        errorMessage = (err as Error).message || String(err) || errorMessage;
      }

      if (!isCancelled) {
        // Ensure errorMessage is always a string
        const finalErrorMessage = typeof errorMessage === 'string' ? errorMessage : String(errorMessage || 'Failed to generate image');
        setError(finalErrorMessage);
      }
      
      // Clear task_id on error (but not on cancellation - that's handled in cancelGeneration)
      if (!isCancelled) {
        console.log('❌ [useImageGeneration] Generation failed, clearing task_id');
        currentTaskIdRef.current = null;
      } else {
        console.log('🚫 [useImageGeneration] Generation cancelled, task_id will be cleared in cancelGeneration');
      }
      
      return null;
    } finally {
      // Only clear isGenerating state and abortController
      // DON'T clear task_id here - it should only be cleared when:
      // 1. Generation completes successfully (in try block after response)
      // 2. Generation fails with error (in catch block, but NOT on cancellation)
      // 3. User cancels (in cancelGeneration function)
      // This ensures task_id is available for cancellation even if generation is in progress
      setIsGenerating(false);
      isGeneratingRef.current = false; // Clear ref guard
      
      // Clear global controller on cleanup
      if (activeGenerationController === abortController) {
        activeGenerationController = null;
      }
      abortControllerRef.current = null;
    }
  }, [fetchImage]);

  const applyWhiteBalance = useCallback(async (
    file: File,
    method: 'auto' | 'manual' | 'ai' = 'auto'
  ): Promise<WhiteBalanceResponse | null> => {
    setIsGenerating(true);
    setError(null);

    try {
      const options: WhiteBalanceRequest = {
        method
      };

      const response = await apiService.whiteBalance(file, options);
      return response;
    } catch (err) {
      let errorMessage = 'Failed to apply white balance';

      try {
        const errorData = JSON.parse((err as Error).message);
        errorMessage = errorData.detail || errorMessage;
      } catch {
        errorMessage = (err as Error).message || errorMessage;
      }

      setError(errorMessage);
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const cancelGeneration = useCallback(async () => {
    console.log('🚫 [useImageGeneration] Cancelling image generation...');
    
    // Cancel backend task FIRST (before closing connections)
    // This ensures the backend job is marked for cancellation
    if (currentTaskIdRef.current) {
      try {
        console.log(`📞 [useImageGeneration] Calling cancel API for task_id: ${currentTaskIdRef.current}`);
        await apiService.cancelAsyncGenerationTask(currentTaskIdRef.current);
        console.log(`✅ [useImageGeneration] Async generation task ${currentTaskIdRef.current} cancelled on backend`);
      } catch (error) {
        console.warn(`⚠️ [useImageGeneration] Failed to cancel async generation task: ${error}`);
        // Continue with cleanup even if cancel API call fails
      }
    } else {
      console.warn('⚠️ [useImageGeneration] No task_id found to cancel');
    }
    
    
    // Cancel HTTP request
    if (abortControllerRef.current) {
      console.log('🚫 [useImageGeneration] Aborting HTTP request...');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    
    // Clear task_id tracking AFTER attempting to cancel
    currentTaskIdRef.current = null;
    
    setIsGenerating(false);
    setError(null);
  }, []);

  // 🧹 Cleanup effect: abort any active generation when component unmounts
  // Critical for React 18 Strict Mode which mounts components twice in dev mode
  useEffect(() => {
    return () => {
      if (activeGenerationController) {
        console.log("🧹 [useImageGeneration] Component unmounting, aborting active generation (React Strict Mode cleanup)");
        activeGenerationController.abort();
        activeGenerationController = null;
      }
      isGeneratingRef.current = false; // Reset ref guard on unmount
    };
  }, []);

  return {
    generateImage,
    applyWhiteBalance,
    isGenerating,
    error,
    lastGeneration,
    clearError,
    cancelGeneration
  };
}