// API service for ArtMancer backend integration
// *** DEMO MODE ACTIVE - ALL API CALLS ARE MOCKED WITH DUMMY DATA ***
// To re-enable real API calls, uncomment the code in makeRequest() method
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export interface ModelSettings {
  model?: string;
  true_cfg_scale?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  negative_prompt?: string;
  generator_seed?: number;
  num_images_per_prompt?: number;
}

export interface GenerationRequest {
  prompt: string;
  input_image?: string; // Base64 encoded input image (for single image editing)
  input_images?: string[]; // List of base64 encoded input images (for multi-image editing)
  settings?: ModelSettings;
  return_format?: 'base64' | 'url' | 'both';
}

export interface GenerationResponse {
  success: boolean;
  prompt: string;
  model_used: string;
  settings_used: Record<string, any>;
  generated_text?: string;
  image_base64?: string;
  image_url?: string;
  generation_time: number;
  preset_used?: string;
  preset_description?: string;
}

export interface ApiError {
  success: false;
  error: string;
  error_type: string;
  details?: Record<string, any>;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // COMMENTED OUT REAL API CALLS - USING DUMMY DATA FOR DEMO
    /*
    const url = `${this.baseUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const responseData = await response.json();

      if (!response.ok) {
        // The server returned an error response
        const errorMessage = responseData?.error || `HTTP ${response.status}: ${response.statusText}`;
        const errorType = responseData?.error_type || 'unknown_error';
        
        // Create user-friendly error messages based on error type
        let friendlyMessage = errorMessage;
        
        switch (errorType) {
          case 'quota_exceeded':
            friendlyMessage = responseData?.error || '⏱️ API quota exceeded. Please try again later or upgrade your plan.';
            break;
          case 'api_key_error':
            friendlyMessage = responseData?.error || '🔑 API authentication failed. Please check your API key configuration.';
            break;
          case 'permission_denied':
            friendlyMessage = responseData?.error || '🔐 Access denied. Please check your API permissions and billing.';
            break;
          case 'invalid_request':
            friendlyMessage = responseData?.error || '❌ Invalid request. Please check your input and try again.';
            break;
          default:
            // For quota errors specifically (status 429), provide helpful message even without error_type
            if (response.status === 429) {
              friendlyMessage = responseData?.error || '⏱️ Rate limit exceeded. Please wait a moment and try again.';
            } else {
              friendlyMessage = errorMessage;
            }
        }
        
        const errorDetails = {
          status: response.status,
          statusText: response.statusText,
          error: friendlyMessage,
          error_type: errorType,
          details: responseData?.details || null,
          endpoint: endpoint
        };
        
        console.error('API Error:', errorDetails);
        throw new Error(JSON.stringify(errorDetails));
      }

      return responseData;
    } catch (error) {
      // Network or parsing error
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        const networkError = {
          status: 0,
          error: 'Network error: Unable to connect to server. Please check if the server is running.',
          endpoint: endpoint,
          baseUrl: this.baseUrl
        };
        console.error('Network Error:', networkError);
        throw new Error(JSON.stringify(networkError));
      }
      
      // Re-throw other errors (including our formatted API errors)
      throw error;
    }
    */

    // DUMMY DATA FOR DEMO PURPOSES - NO REAL API CALLS
    console.log(`[DEMO MODE] Simulating API call to: ${endpoint}`);
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
    
    return this.getDummyResponse<T>(endpoint, options);
  }

  private getDummyResponse<T>(endpoint: string, options: RequestInit = {}): T {
    // Generate dummy responses based on endpoint
    if (endpoint === '/api/health') {
      return {
        success: true,
        status: 'healthy',
        service: 'ArtMancer API (Demo Mode)',
        gemini_client: 'connected'
      } as T;
    }

    if (endpoint === '/api/models') {
      return {
        success: true,
        models: [
          { name: 'gemini-pro-vision', info: { description: 'Google Gemini Pro Vision Model' } },
          { name: 'stable-diffusion-xl', info: { description: 'Stable Diffusion XL Model' } },
          { name: 'midjourney-v6', info: { description: 'Midjourney Version 6 Model' } }
        ],
        default: 'gemini-pro-vision'
      } as T;
    }

    if (endpoint === '/api/presets') {
      return {
        success: true,
        presets: {
          'portrait': { description: 'Portrait photography style', settings: { guidance_scale: 7.5 } },
          'landscape': { description: 'Landscape photography style', settings: { guidance_scale: 8.0 } },
          'artistic': { description: 'Artistic illustration style', settings: { guidance_scale: 10.0 } }
        },
        default_preset: 'portrait'
      } as T;
    }

    if (endpoint === '/api/generate' || endpoint.startsWith('/api/generate/preset/')) {
      const requestBody = options.body ? JSON.parse(options.body as string) : {};
      const prompt = requestBody.prompt || 'sample prompt';
      
      // Generate a dummy base64 image (1x1 pixel placeholder)
      const dummyImageBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
      
      return {
        success: true,
        prompt: prompt,
        model_used: 'gemini-pro-vision',
        settings_used: {
          guidance_scale: 7.5,
          num_inference_steps: 20,
          generator_seed: Math.floor(Math.random() * 1000000)
        },
        generated_text: `Generated art for: "${prompt}" - This is a demo response showing how the UI works!`,
        image_base64: dummyImageBase64,
        image_url: '/placeholder-image.png',
        generation_time: 2.5,
        preset_used: endpoint.includes('/preset/') ? endpoint.split('/').pop() : undefined,
        preset_description: endpoint.includes('/preset/') ? 'Demo preset description' : undefined
      } as T;
    }

    if (endpoint === '/api/generate/batch') {
      const requestBody = options.body ? JSON.parse(options.body as string) : [];
      const requests = Array.isArray(requestBody) ? requestBody : [requestBody];
      
      return {
        success: true,
        batch_id: `batch_${Date.now()}`,
        total_requests: requests.length,
        results: requests.map((_, index) => ({
          index,
          success: true,
          result: {
            success: true,
            prompt: `Batch request ${index + 1}`,
            model_used: 'gemini-pro-vision',
            settings_used: { guidance_scale: 7.5 },
            generated_text: `Batch generated art ${index + 1} - Demo mode`,
            image_base64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
            generation_time: 2.0
          }
        }))
      } as T;
    }

    if (endpoint === '/api/config') {
      return {
        success: true,
        config: {
          available_formats: ['base64', 'url', 'both'],
          max_prompt_length: 1000,
          supported_models: ['gemini-pro-vision', 'stable-diffusion-xl', 'midjourney-v6'],
          model_info: {
            'gemini-pro-vision': { description: 'Google Gemini Pro Vision Model' },
            'stable-diffusion-xl': { description: 'Stable Diffusion XL Model' },
            'midjourney-v6': { description: 'Midjourney Version 6 Model' }
          },
          available_presets: ['portrait', 'landscape', 'artistic'],
          default_settings: {
            guidance_scale: 7.5,
            num_inference_steps: 20,
            generator_seed: -1
          },
          preset_descriptions: {
            'portrait': 'Portrait photography style',
            'landscape': 'Landscape photography style',
            'artistic': 'Artistic illustration style'
          }
        }
      } as T;
    }

    // Default response for unknown endpoints
    return {
      success: true,
      message: `Demo response for ${endpoint}`,
      data: null
    } as T;
  }

  // Health check
  async healthCheck() {
    return this.makeRequest<{
      success: boolean;
      status: string;
      service: string;
      gemini_client: string;
    }>('/api/health');
  }

  // Get available models
  async getModels() {
    return this.makeRequest<{
      success: boolean;
      models: Array<{
        name: string;
        info: Record<string, any>;
      }>;
      default: string;
    }>('/api/models');
  }

  // Get available presets
  async getPresets() {
    return this.makeRequest<{
      success: boolean;
      presets: Record<string, any>;
      default_preset: string;
    }>('/api/presets');
  }

  // Generate image
  async generateImage(request: GenerationRequest): Promise<GenerationResponse> {
    return this.makeRequest<GenerationResponse>('/api/generate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Generate image with preset
  async generateWithPreset(
    presetName: string,
    request: Omit<GenerationRequest, 'settings'>
  ): Promise<GenerationResponse> {
    return this.makeRequest<GenerationResponse>(`/api/generate/preset/${presetName}`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Batch generation
  async generateBatch(requests: GenerationRequest[]) {
    return this.makeRequest<{
      success: boolean;
      batch_id: string;
      total_requests: number;
      results: Array<{
        index: number;
        success: boolean;
        result?: GenerationResponse;
        error?: ApiError;
      }>;
    }>('/api/generate/batch', {
      method: 'POST',
      body: JSON.stringify(requests),
    });
  }

  // Get API configuration
  async getConfig() {
    return this.makeRequest<{
      success: boolean;
      config: {
        available_formats: string[];
        max_prompt_length: number;
        supported_models: string[];
        model_info: Record<string, any>;
        available_presets: string[];
        default_settings: ModelSettings;
        preset_descriptions: Record<string, string>;
      };
    }>('/api/config');
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;