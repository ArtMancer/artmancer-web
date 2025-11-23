import { useState, useCallback, useRef } from 'react';
import { apiService, GenerationRequest, GenerationResponse, ModelSettings, WhiteBalanceRequest, WhiteBalanceResponse } from '@/services/api';

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

      const base64MaskImage = maskImage.startsWith('data:')
        ? maskImage.split(',')[1]
        : maskImage;

      const base64ReferenceImage = referenceImage
        ? (referenceImage.startsWith('data:')
          ? referenceImage.split(',')[1]
          : referenceImage)
        : undefined;

      const request: GenerationRequest = {
        prompt: prompt.trim(),
        input_image: base64InputImage,
        mask_image: base64MaskImage,
        reference_image: base64ReferenceImage,
        num_inference_steps: settings?.num_inference_steps,
        guidance_scale: settings?.guidance_scale,
        true_cfg_scale: settings?.true_cfg_scale,
        negative_prompt: settings?.negative_prompt,
        seed: settings?.generator_seed,
        width: settings?.width,
        height: settings?.height,
        task_type: taskType,
      };

      // Debug logging
      console.log("🔍 [Frontend API] Request details:", {
        hasReferenceImage: !!request.reference_image,
        referenceImageLength: request.reference_image?.length || 0,
        inferredTaskType: request.reference_image ? "insertion" : "removal",
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
            errorMessage = errorData.error || 'Unable to connect to server. Please check if the API server is running on port 8003.';
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