// API service for ArtMancer backend integration
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