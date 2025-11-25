// API service for ArtMancer backend integration
// API URL is configured via NEXT_PUBLIC_API_URL environment variable
// Default: http://localhost:8003 (backend default port)
// To change: Set NEXT_PUBLIC_API_URL in .env.local file
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003';

export type InputQualityPreset = "super_low" | "low" | "medium" | "high" | "original";

export interface ModelSettings {
  true_cfg_scale?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  negative_prompt?: string;
  generator_seed?: number;
  width?: number;
  height?: number;
  input_quality?: InputQualityPreset;
}

export interface GenerationRequest {
  prompt: string;
  input_image: string; // Base64 encoded input image (required)
  mask_image: string; // Base64 encoded mask image (required)
  reference_image?: string; // Base64 encoded reference image (optional, for object insertion)
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  true_cfg_scale?: number;
  negative_prompt?: string;
  seed?: number;
  task_type?: "white-balance" | "object-insert" | "object-removal";
  input_quality?: InputQualityPreset;
}

export interface GenerationResponse {
  success: boolean;
  image: string; // Base64 encoded image
  generation_time: number;
  model_used: string;
  parameters_used: Record<string, string | number | null>;
  request_id?: string; // Request ID for accessing visualization images
}

export interface ApiError {
  success: false;
  error: string;
  error_type: string;
  details?: Record<string, any>;
}

export interface UploadResponse {
  success: boolean;
  image: string;
  original_filename: string;
  image_size: [number, number];
  content_type: string;
}

export interface WhiteBalanceRequest {
  method: 'auto' | 'manual' | 'ai';
  temperature?: number;
  tint?: number;
}

export interface WhiteBalanceResponse {
  success: boolean;
  original_image: string;
  corrected_image: string;
  method_used: string;
  parameters?: {
    temperature?: number;
    tint?: number;
  };
}

export interface EvaluationImagePair {
  original_image: string;
  target_image: string;
  filename?: string;
}

export interface EvaluationRequest {
  original_image?: string;
  target_image?: string;
  image_pairs?: EvaluationImagePair[];
  conditional_images?: string[];
  input_image?: string;
}

export interface EvaluationMetrics {
  psnr?: number;
  ssim?: number;
  lpips?: number;
  fid?: number;
  custom_metric_1?: number;
  custom_metric_2?: number;
  evaluation_time?: number;
}

export interface EvaluationResult {
  filename?: string;
  metrics: EvaluationMetrics;
  success: boolean;
  error?: string;
}

export interface EvaluationResponse {
  success: boolean;
  results: EvaluationResult[];
  total_pairs: number;
  successful_evaluations: number;
  failed_evaluations: number;
  total_evaluation_time: number;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit & { signal?: AbortSignal } = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    // Debug logging
    console.log('🌐 API Request:', {
      method: options.method || 'GET',
      url,
      baseUrl: this.baseUrl,
      endpoint,
      hasBody: !!options.body,
      hasSignal: !!options.signal,
    });

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
        signal: options.signal,
      });

      console.log('📡 API Response:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries()),
      });

      let responseData;
      try {
        const text = await response.text();
        if (!text || text.trim() === '') {
          throw new Error('Empty response from server');
        }
        responseData = JSON.parse(text);
      } catch (parseError) {
        console.error('❌ Failed to parse response as JSON:', parseError);
        const text = await response.text().catch(() => 'Unable to read response');
        console.error('Response text:', text);
        throw new Error(JSON.stringify({
          status: response.status,
          error: `Invalid JSON response: ${text.substring(0, 100)}`,
          endpoint: endpoint
        }));
      }

      if (!response.ok) {
        // The server returned an error response
        // Handle empty object or missing error fields
        let errorMessage = responseData?.detail || responseData?.error || responseData?.message;
        if (!errorMessage || (typeof responseData === 'object' && Object.keys(responseData).length === 0)) {
          errorMessage = `HTTP ${response.status}: ${response.statusText || 'Unknown error'}`;
        }
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
      // Check if request was aborted
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('🚫 Request cancelled by user');
        throw new Error(JSON.stringify({
          status: 0,
          error: 'Request cancelled',
          error_type: 'cancelled',
          endpoint: endpoint
        }));
      }

      // Network or parsing error
      const errorInfo = {
        errorType: error instanceof Error ? error.constructor.name : typeof error,
        errorMessage: error instanceof Error ? error.message : String(error),
        errorStack: error instanceof Error ? error.stack : undefined,
        endpoint,
        url,
        baseUrl: this.baseUrl,
      };
      
      console.error('❌ API Request Error:', errorInfo);
      console.error('❌ Raw Error Object:', error);

      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        const networkError = {
          status: 0,
          error: `Network error: Unable to connect to server at ${this.baseUrl}. Please check:
1. Is the backend server running on port 8003?
2. Is the server accessible at ${this.baseUrl}?
3. Are there any CORS issues?`,
          endpoint: endpoint,
          baseUrl: this.baseUrl,
          fullUrl: url,
        };
        console.error('🌐 Network Error Details:', networkError);
        throw new Error(JSON.stringify(networkError));
      }

      // Re-throw other errors (including our formatted API errors)
      throw error;
    }
  }

  // Health check with retry logic
  async healthCheck(options?: { retries?: number; retryDelay?: number; timeout?: number }) {
    const maxRetries = options?.retries ?? 5;
    const baseDelay = options?.retryDelay ?? 1000; // Start with 1 second
    const timeout = options?.timeout ?? 10000; // 10 seconds timeout per request

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // Create a timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Request timeout')), timeout);
        });

        // Race between the actual request and timeout
        const requestPromise = this.makeRequest<{
      status: string;
      model_loaded: boolean;
      device: string;
      device_info?: Record<string, any>;
    }>('/api/health');

        const result = await Promise.race([requestPromise, timeoutPromise]);
        return result;
      } catch (error) {
        const isLastAttempt = attempt === maxRetries - 1;
        
        if (isLastAttempt) {
          // On last attempt, throw the error
          throw error;
        }

        // Calculate exponential backoff delay: 1s, 2s, 4s, 8s, 16s
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`⚠️ Health check attempt ${attempt + 1}/${maxRetries} failed, retrying in ${delay}ms...`);
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // This should never be reached, but TypeScript needs it
    throw new Error('Health check failed after all retries');
  }

  // Get available models (not implemented in backend yet)
  async getModels() {
    // Backend doesn't have this endpoint yet
    return {
      success: true,
      models: [],
      default: 'qwen-image-edit'
    };
  }

  // Get available presets (not implemented in backend yet)
  async getPresets() {
    // Backend doesn't have this endpoint yet
    return {
      success: true,
      presets: {},
      default_preset: 'default'
    };
  }

  // Generate image
  async generateImage(request: GenerationRequest, signal?: AbortSignal): Promise<GenerationResponse> {
    return this.makeRequest<GenerationResponse>('/api/generate', {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
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

  // Get API configuration (not implemented in backend yet)
  async getConfig() {
    // Backend doesn't have this endpoint yet
    return {
      success: true,
      config: {
        available_formats: ['base64'],
        max_prompt_length: 1000,
        supported_models: ['qwen-image-edit'],
        model_info: {
          'qwen-image-edit': { description: 'Qwen Image Edit Model' }
        },
        available_presets: [],
        default_settings: {
          guidance_scale: 3.5,
          num_inference_steps: 30,
          true_cfg_scale: 6.0
        },
        preset_descriptions: {}
      }
    };
  }

  // Upload image
  async uploadImage(file: File): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${this.baseUrl}/upload-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Error uploading image:', error);
      throw error;
    }
  }

  // Evaluate images
  async evaluateImages(request: EvaluationRequest): Promise<EvaluationResponse> {
    return this.makeRequest<EvaluationResponse>('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Apply white balance
  async whiteBalance(file: File, options: WhiteBalanceRequest): Promise<WhiteBalanceResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('method', options.method);

      if (options.temperature !== undefined) {
        formData.append('temperature', options.temperature.toString());
      }
      if (options.tint !== undefined) {
        formData.append('tint', options.tint.toString());
      }

      const response = await fetch(`${this.baseUrl}/api/white-balance`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'White balance failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Error applying white balance:', error);
      throw error;
    }
  }

  // Generate smart mask using FastSAM
  async generateSmartMask(
    image: string | null,
    imageId: string | null,
    bbox?: [number, number, number, number],
    points?: Array<[number, number]>,
    dilateAmount: number = 10,
    useBlur: boolean = false
  ): Promise<{ success: boolean; mask_base64: string; image_id?: string; error?: string }> {
    return this.makeRequest<{
      success: boolean;
      mask_base64: string;
      image_id?: string;
      error?: string;
    }>('/api/smart-mask', {
      method: 'POST',
      body: JSON.stringify({
        image: image,
        image_id: imageId,
        bbox: bbox,
        points: points,
        dilate_amount: dilateAmount,
        use_blur: useBlur,
      }),
    });
  }

  // Download visualization images
  async downloadVisualization(requestId: string, format: string = "zip"): Promise<void> {
    const url = `${this.baseUrl}/api/visualization/${requestId}/download?format=${format}`;
    window.open(url, '_blank');
  }

  // Get visualization image URLs
  getVisualizationOriginalUrl(requestId: string): string {
    return `${this.baseUrl}/api/visualization/${requestId}/original`;
  }

  getVisualizationGeneratedUrl(requestId: string): string {
    return `${this.baseUrl}/api/visualization/${requestId}/generated`;
  }

  // Benchmark methods
  async validateBenchmarkFolder(file: File | File[]): Promise<{
    success: boolean;
    message: string;
    image_count: number;
    details?: Record<string, number>;
  }> {
    try {
      const formData = new FormData();
      
      if (Array.isArray(file)) {
        // Folder upload: append all files with their relative paths
        file.forEach((f) => {
          const relativePath = (f as any).webkitRelativePath || f.name;
          formData.append('files', f, relativePath);
        });
        formData.append('upload_type', 'folder');
      } else {
        // ZIP file upload
        formData.append('file', file);
        formData.append('upload_type', 'zip');
      }

      const response = await fetch(`${this.baseUrl}/api/benchmark/validate`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Validation failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Error validating benchmark folder:', error);
      throw error;
    }
  }

  async runBenchmark(
    file: File | File[],
    options: {
      task_type?: string;
      prompt?: string;
      sample_count?: number;
      num_inference_steps?: number;
      guidance_scale?: number;
      true_cfg_scale?: number;
      negative_prompt?: string;
      seed?: number;
      input_quality?: string;
    }
  ): Promise<any> {
    try {
      const formData = new FormData();
      
      if (Array.isArray(file)) {
        // Folder upload: append all files with their relative paths
        file.forEach((f) => {
          const relativePath = (f as any).webkitRelativePath || f.name;
          formData.append('files', f, relativePath);
        });
        formData.append('upload_type', 'folder');
      } else {
        // ZIP file upload
        formData.append('file', file);
        formData.append('upload_type', 'zip');
      }
      
      formData.append('task_type', options.task_type || 'object-removal');
      // Prompt is required - will be used for ALL images
      formData.append('prompt', options.prompt || '');
      if (options.sample_count !== undefined) formData.append('sample_count', options.sample_count.toString());
      if (options.num_inference_steps) formData.append('num_inference_steps', options.num_inference_steps.toString());
      if (options.guidance_scale) formData.append('guidance_scale', options.guidance_scale.toString());
      if (options.true_cfg_scale) formData.append('true_cfg_scale', options.true_cfg_scale.toString());
      if (options.negative_prompt) formData.append('negative_prompt', options.negative_prompt);
      if (options.seed !== undefined) formData.append('seed', options.seed.toString());
      if (options.input_quality) formData.append('input_quality', options.input_quality);

      const response = await fetch(`${this.baseUrl}/api/benchmark/run`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Benchmark failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Error running benchmark:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;