import { useState, useCallback, useRef } from 'react';
import { apiService, GenerationRequest, GenerationResponse, ModelSettings, WhiteBalanceRequest, WhiteBalanceResponse, TaskType } from '@/services/api';

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

  const generateImage = useCallback(async (
    prompt: string,
    inputImage: string,
    maskImage: string,
    settings?: ModelSettings,
    referenceImage?: string | null,
    taskType?: "white-balance" | "object-insert" | "object-removal"
  ): Promise<GenerationResponse | null> => {
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

    setIsGenerating(true);
    setError(null);

    // Create new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

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

      const request: GenerationRequest = {
        prompt: prompt.trim(),
        input_image: base64InputImage,
        conditional_images: conditional_images.length > 0 ? conditional_images : undefined,
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
        enable_4bit_text_encoder: settings?.enable_4bit_text_encoder,
        enable_cpu_offload: settings?.enable_cpu_offload,
        enable_memory_optimizations: settings?.enable_memory_optimizations,
        enable_flowmatch_scheduler: settings?.enable_flowmatch_scheduler,
      };

      // Debug logging
      console.log("🔍 [Frontend API] Request details:", {
        hasConditionalImages: !!request.conditional_images,
        conditionalImagesCount: request.conditional_images?.length || 0,
        taskType: request.task_type,
        hasMask: !!request.mask_image,
      });

      const response = await apiService.generateImage(request, abortController.signal);
      setLastGeneration(response);
      return response;
    } catch (err) {
      // Check if request was cancelled
      let errorMessage = 'Failed to generate image';
      let isCancelled = false;

      try {
        // Try to parse error message as JSON (from makeRequest)
        const errorMessageStr = (err as Error).message || String(err);
        let errorData: any = null;
        
        try {
          errorData = JSON.parse(errorMessageStr);
        } catch (parseErr) {
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
        } else if (errorData.status === 500) {
          // Server error
            errorMessage = errorData.error || 'Server error occurred. Please check the server logs and try again.';
        } else if (errorData.error) {
          // API error with message
          errorMessage = errorData.error;
          } else if (errorData.detail) {
            // FastAPI error format
            errorMessage = errorData.detail;
        }

        console.error('Generation error details:', errorData);
        }
      } catch (fallbackErr) {
        // Fallback for any unexpected errors
        console.error('Error handling failed:', fallbackErr);
        errorMessage = (err as Error).message || String(err) || errorMessage;
      }

      if (!isCancelled) {
      setError(errorMessage);
      }
      return null;
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, []);

  const applyWhiteBalance = useCallback(async (
    file: File,
    method: 'auto' | 'manual' | 'ai' = 'auto',
    temperature?: number,
    tint?: number
  ): Promise<WhiteBalanceResponse | null> => {
    setIsGenerating(true);
    setError(null);

    try {
      const options: WhiteBalanceRequest = {
        method,
        temperature,
        tint
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

  const cancelGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      console.log('🚫 Cancelling image generation...');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsGenerating(false);
      setError(null);
    }
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