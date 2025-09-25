import { useState, useCallback } from 'react';
import { apiService, GenerationRequest, GenerationResponse, ModelSettings } from '@/services/api';

export function useImageGeneration() {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastGeneration, setLastGeneration] = useState<GenerationResponse | null>(null);

  const generateImage = useCallback(async (
    prompt: string,
    inputImage?: string, // Now optional for multi-image support
    settings?: ModelSettings,
    preset?: string
  ): Promise<GenerationResponse | null> => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return null;
    }

    if (!inputImage) {
      setError('Please upload an image to edit');
      return null;
    }

    setIsGenerating(true);
    setError(null);

    try {
      // Convert data URL to base64 if needed
      const base64Image = inputImage.startsWith('data:') 
        ? inputImage.split(',')[1] 
        : inputImage;

      const request: GenerationRequest = {
        prompt: prompt.trim(),
        input_image: base64Image,
        settings,
        return_format: 'base64'
      };

      let response: GenerationResponse;
      
      if (preset) {
        response = await apiService.generateWithPreset(preset, request);
      } else {
        response = await apiService.generateImage(request);
      }

      setLastGeneration(response);
      return response;
    } catch (err) {
      let errorMessage = 'Failed to generate image';
      
      try {
        const errorData = JSON.parse((err as Error).message);
        
        if (errorData.status === 0) {
          // Network error
          errorMessage = 'Unable to connect to server. Please check if the API server is running on port 8080.';
        } else if (errorData.status === 500) {
          // Server error
          errorMessage = 'Server error occurred. Please check the server logs and try again.';
        } else if (errorData.error) {
          // API error with message
          errorMessage = errorData.error;
        }
        
        console.error('Generation error details:', errorData);
      } catch {
        // Fallback for non-JSON errors
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

  return {
    generateImage,
    isGenerating,
    error,
    lastGeneration,
    clearError
  };
}