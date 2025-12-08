// API service for ArtMancer backend integration
// API Gateway is the single entry point for all requests
// All requests go through API Gateway, which routes to appropriate services
const sanitizeUrl = (url?: string | null) => url?.trim().replace(/\/+$/, '');
const DEFAULT_API_GATEWAY_ENDPOINT = 'https://nxan2911--api-gateway.modal.run';

// @ts-ignore - process.env is available in Next.js runtime
const RAW_API_GATEWAY_URL = sanitizeUrl(
  process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
  process.env.NEXT_PUBLIC_API_URL
);

// API Gateway is the main entry point
const API_BASE_URL = RAW_API_GATEWAY_URL ?? DEFAULT_API_GATEWAY_ENDPOINT;

const isRemoteEndpoint = (url: string) =>
  url.includes('modal.run') || url.includes('api.runpod.ai');

export type InputQualityPreset = "resized" | "original";

export interface ModelSettings {
  true_cfg_scale?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  negative_prompt?: string;
  generator_seed?: number;
  width?: number;
  height?: number;
  input_quality?: InputQualityPreset;
  enable_flowmatch_scheduler?: boolean;
}

// Valid task types matching backend schema
export type TaskType = "insertion" | "removal" | "white-balance";

export interface GenerationRequest {
  prompt: string;
  input_image: string; // Base64 encoded input image (required)
  conditional_images?: string[]; // List of base64 encoded conditional images (first element is mask, optional extras)
  // Legacy fields for backward compatibility (will be converted to conditional_images)
  mask_image?: string; // Base64 encoded mask image (will be converted to conditional_images[0])
  reference_image?: string; // Base64 encoded reference image (not used in new Qwen flow)
  reference_mask_R?: string; // Base64 encoded mask isolating object in reference image (Mask R - for two-source mask workflow)
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  true_cfg_scale?: number;
  negative_prompt?: string;
  seed?: number;
  task_type?: TaskType; // Must be "insertion", "removal", or "white-balance" (matches backend exactly)
  input_quality?: InputQualityPreset;
  // Low-end optimization flags (for GPU 12GB or lower)
  enable_flowmatch_scheduler?: boolean; // Use FlowMatchEulerDiscreteScheduler instead of default
}

export interface DebugInfo {
  conditional_images?: string[]; // Base64 encoded conditional images (mask, background, object, mae)
  conditional_labels?: string[]; // Labels for each conditional image
  input_image_size?: string;
  output_image_size?: string;
  lora_adapter?: string;
  loaded_adapters?: string[];
  positioned_mask_R?: string; // Base64 encoded positioned mask R (reference mask R after being pasted into main mask A, only for reference-guided insertion)
  // Additional debug images
  original_image?: string;
  mask_A?: string;
  reference_image?: string;
  reference_mask_R?: string;
  // Prompt info
  original_prompt?: string;
  refined_prompt?: string;
  prompt_was_refined?: boolean;
  session_name?: string; // Debug session name for downloading
  debug_path?: string; // Debug session path
}

export interface GenerationResponse {
  success: boolean;
  image: string; // Base64 encoded image
  generation_time: number;
  model_used: string;
  parameters_used: Record<string, string | number | null>;
  request_id?: string; // Request ID for accessing visualization images
  debug_info?: DebugInfo; // Debug information (conditional images, parameters)
  debug_path?: string; // Debug session path for downloading
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
  private baseUrl: string; // API Gateway (single entry point)

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl; // API Gateway
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
          error: `Network error: Unable to connect to API server at ${this.baseUrl}. Please check:\n1. Is the API endpoint reachable?\n2. Is your internet connection stable?\n3. Are there any CORS/network issues?`,
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
  async generateImage(
    request: GenerationRequest,
    signal?: AbortSignal,
    onProgress?: (progress: { current_step?: number; total_steps?: number; status: string; progress: number; loading_message?: string }) => void,
    onAsyncStart?: (taskId: string, eventSource: EventSource) => void
  ): Promise<GenerationResponse> {
    // Always use async generation endpoint for progress updates
    const url = `${this.baseUrl}/api/generate/async`;

    console.log('🌐 API Request:', {
      method: 'POST',
      url,
      endpoint: 'API Gateway (Async)',
      hasBody: true,
      hasSignal: !!signal,
    });

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
        },
        method: 'POST',
        body: JSON.stringify(request),
        signal,
      });

      console.log('📡 API Response:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
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
          endpoint: '/api/generate/async'
        }));
      }

      if (!response.ok) {
        // Extract error message - handle both string and object formats
        let errorMessage: string;
        if (typeof responseData?.detail === 'string') {
          errorMessage = responseData.detail;
        } else if (typeof responseData?.detail === 'object' && responseData?.detail?.error) {
          errorMessage = responseData.detail.error;
        } else if (typeof responseData?.error === 'string') {
          errorMessage = responseData.error;
        } else if (typeof responseData?.error === 'object' && responseData?.error?.error) {
          errorMessage = responseData.error.error;
        } else if (typeof responseData?.message === 'string') {
          errorMessage = responseData.message;
        } else {
          errorMessage = `HTTP ${response.status}: ${response.statusText || 'Unknown error'}`;
        }

        // Ensure errorMessage is always a string
        if (typeof errorMessage !== 'string') {
          errorMessage = String(errorMessage || `HTTP ${response.status}: ${response.statusText || 'Unknown error'}`);
        }

        let errorType = responseData?.error_type || responseData?.detail?.error_type || 'unknown_error';

        // Handle specific error status codes
        if (response.status === 400) {
          errorMessage = 'No worker available. The endpoint may be initializing. Please try again.';
          errorType = 'no_worker_available';
        } else if (response.status === 503) {
          // Service unavailable - generation service may be cold starting
          if (!errorMessage || errorMessage.includes('HTTP 503')) {
            errorMessage = 'Service temporarily unavailable. The generation service may be starting up. Please try again in a moment.';
          }
          errorType = 'service_unavailable';
        } else if (response.status === 524) {
          errorMessage = 'Request processing timeout (exceeded 5.5 minutes). The image may be too complex or the server is overloaded.';
          errorType = 'processing_timeout';
        } else if (response.status === 502) {
          errorMessage = 'Worker misconfigured or unavailable. Please check the endpoint configuration.';
          errorType = 'worker_misconfigured';
        }

        const errorDetails = {
          status: response.status,
          statusText: response.statusText,
          error: errorMessage,
          error_type: errorType,
          details: responseData?.details || null,
          endpoint: '/api/generate/async'
        };

        console.error('API Error:', errorDetails);
        throw new Error(JSON.stringify(errorDetails));
      }

      // Async endpoint always returns task_id
      if (responseData.task_id && responseData.status === "queued") {
        // Use SSE to stream progress and poll for result
        return await this.generateImageAsync(responseData.task_id, signal, onProgress, onAsyncStart);
      }

      // If no task_id, this is an error (async endpoint should always return task_id)
      throw new Error(JSON.stringify({
        status: response.status,
        error: 'Async generation endpoint did not return task_id. This should not happen.',
        endpoint: '/api/generate/async',
        response: responseData
      }));
    } catch (error) {
      // Check if request was aborted
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('🚫 Request cancelled by user');
        throw new Error(JSON.stringify({
          status: 0,
          error: 'Request cancelled',
          error_type: 'cancelled',
          endpoint: '/api/generate/async'
        }));
      }

      // Network or parsing error
      const errorInfo = {
        errorType: error instanceof Error ? error.constructor.name : typeof error,
        errorMessage: error instanceof Error ? error.message : String(error),
        errorStack: error instanceof Error ? error.stack : undefined,
        endpoint: '/api/generate/async',
        url,
      };

      console.error('❌ API Request Error:', errorInfo);
      console.error('❌ Raw Error Object:', error);

      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        const networkError = {
          status: 0,
          error: `Network error: Unable to connect to API Gateway at ${this.baseUrl}. Please check:
1. Is the API Gateway accessible?
2. Are there any CORS issues?
3. Is the API Gateway URL correct?`,
          endpoint: '/api/generate/async',
          baseUrl: this.baseUrl,
          fullUrl: url,
        };
        console.error('🌐 Network Error Details:', networkError);
        throw new Error(JSON.stringify(networkError));
      }

      // Re-throw other errors
      throw error;
    }
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

      // White-balance endpoint (if exists, otherwise use baseUrl)
      // Note: white-balance may not be available in image-utils service
      const whiteBalanceUrl = `${this.baseUrl}/api/image-utils/white-balance`;
      const response = await fetch(whiteBalanceUrl, {
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

  // Generate smart mask using FastSAM or BiRefNet
  async generateSmartMask(
    image: string | null,
    imageId: string | null,
    bbox?: [number, number, number, number],
    points?: Array<[number, number]>,
    borderAdjustment: number = 0,
    useBlur: boolean = false,
    modelType: string = "segmentation"  // "segmentation" (FastSAM) or "birefnet"
  ): Promise<{ success: boolean; mask_base64: string; image_id?: string; request_id?: string; error?: string }> {
    // Smart-mask through API Gateway
    const url = `${this.baseUrl}/api/smart-mask`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: image,
          image_id: imageId,
          bbox: bbox,
          points: points,
          border_adjustment: borderAdjustment,
          use_blur: useBlur,
          model_type: modelType,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage =
          errorData.detail ||
          errorData.error ||
          `Smart mask API error: HTTP ${response.status} ${response.statusText}`;

        // If 404 and error mentions image_id expired/not found, and we have image but used imageId
        // This indicates the image cache expired, we should retry with image instead
        if (response.status === 404 &&
          imageId &&
          image &&
          (errorMessage.includes('Image ID not found') || errorMessage.includes('expired'))) {
          console.log('🔄 Image ID expired, retrying with image data...');
          // Retry with image instead of image_id
          const retryResponse = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: image,
              image_id: null, // Clear image_id
              bbox: bbox,
              points: points,
              border_adjustment: borderAdjustment,
              use_blur: useBlur,
            }),
          });

          if (!retryResponse.ok) {
            const retryErrorData = await retryResponse.json().catch(() => ({}));
            const retryErrorMessage =
              retryErrorData.detail ||
              retryErrorData.error ||
              `Smart mask API error: HTTP ${retryResponse.status} ${retryResponse.statusText}`;
            throw new Error(retryErrorMessage);
          }

          return await retryResponse.json();
        }

        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('❌ Smart Mask Request Error:', error);

      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        throw new Error(
          `Network error: Unable to connect to API Gateway at ${this.baseUrl}. ` +
          'Please check:\n1. Is the API Gateway running?\n' +
          '2. Is the API Gateway URL correct?\n3. Are there any CORS/network issues?'
        );
      }

      throw error;
    }
  }

  // Get debug session details (metadata, lora_log, image_files)
  async getDebugSession(sessionName: string): Promise<{
    session_name: string;
    metadata: any;
    lora_log: string;
    image_files: string[];
  }> {
    const url = `${this.baseUrl}/api/debug/sessions/${sessionName}`;

    try {
      const response = await fetch(url, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || `Failed to get debug session: HTTP ${response.status}`;
        
        if (response.status === 404) {
          throw new Error(`Debug session "${sessionName}" not found. It may have been deleted or never existed.`);
        }
        
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting debug session:', error);
      throw error;
    }
  }

  // Get a specific debug image
  async getDebugImage(sessionName: string, imageName: string): Promise<Blob> {
    const url = `${this.baseUrl}/api/debug/sessions/${sessionName}/images/${imageName}`;

    try {
      const response = await fetch(url, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || `Failed to get debug image: HTTP ${response.status}`;
        
        if (response.status === 404) {
          throw new Error(`Debug image "${imageName}" not found in session "${sessionName}".`);
        }
        
        throw new Error(errorMessage);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error getting debug image:', error);
      throw error;
    }
  }

  // Download debug session as ZIP (kept for backward compatibility)
  async downloadDebugSession(sessionName: string): Promise<Blob> {
    const url = `${this.baseUrl}/api/debug/sessions/${sessionName}/download`;

    try {
      const response = await fetch(url, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.detail || `Failed to download debug session: HTTP ${response.status}`;
        
        // Xử lý 404 với message rõ ràng hơn
        if (response.status === 404) {
          throw new Error(`Debug session "${sessionName}" not found. It may have been deleted or never existed.`);
        }
        
        throw new Error(errorMessage);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error downloading debug session:', error);
      throw error;
    }
  }

  // Download visualization images
  // Cancel generation task
  // Cancel async generation task
  async cancelAsyncGenerationTask(taskId: string): Promise<{ success: boolean; message: string; task_id: string }> {
    return this.makeRequest<{ success: boolean; message: string; task_id: string }>(
      `/api/generate/async/cancel/${taskId}`,
      {
        method: 'POST',
      }
    );
  }

  // Cancel smart mask segmentation
  async cancelSmartMask(requestId: string): Promise<{ success: boolean; message: string; request_id: string }> {
    return this.makeRequest<{ success: boolean; message: string; request_id: string }>(
      `/api/smart-mask/cancel/${requestId}`,
      {
        method: 'POST',
      }
    );
  }

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

  async generateImageAsync(
    taskId: string,
    signal?: AbortSignal,
    onProgress?: (progress: { current_step?: number; total_steps?: number; status: string; progress: number; loading_message?: string }) => void,
    onAsyncStart?: (taskId: string, eventSource: EventSource) => void
  ): Promise<GenerationResponse> {
    return new Promise((resolve, reject) => {
      let pollInterval: NodeJS.Timeout | null = null;
      let isResolved = false;

      // Cleanup function
      const cleanup = () => {
        if (pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
        if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
          eventSource.close();
        }
      };

      // Open SSE connection for progress updates
      const streamUrl = `${this.baseUrl}/api/generate/stream/${taskId}`;
      const eventSource = new EventSource(streamUrl);

      // Notify caller about async start (for cancellation tracking)
      if (onAsyncStart) {
        onAsyncStart(taskId, eventSource);
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          console.log('📡 [SSE] Received progress update:', {
            status: data.status,
            progress: data.progress,
            current_step: data.current_step,
            total_steps: data.total_steps,
          });

          // Check for error in data (even if status is not "error" yet)
          if (data.error) {
            eventSource.close();
            // Check if it's an OOM error
            const errorText = typeof data.error === 'string' ? data.error : String(data.error);
            let errorMessage = errorText;
            if (errorText.toLowerCase().includes('out of memory') ||
              errorText.toLowerCase().includes('cuda out of memory') ||
              errorText.toLowerCase().includes('oom')) {
              errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
            }
            reject(new Error(JSON.stringify({
              status: 500,
              error: errorMessage,
              endpoint: `/api/generate/stream/${taskId}`
            })));
            return;
          }

          // Call progress callback for all status updates (not just processing)
          if (onProgress) {
            console.log('📢 [SSE] Calling onProgress callback with:', data.status);
            onProgress({
              current_step: data.current_step,
              total_steps: data.total_steps,
              status: data.status,
              progress: data.progress,
              loading_message: data.loading_message,
            });
          }

          // If done, fetch result and close connection
          if (data.status === "done") {
            cleanup();
            if (!isResolved) {
              isResolved = true;
              this.getGenerationResult(taskId)
                .then(result => resolve(result))
                .catch(err => reject(err));
            }
            return;
          }

          // If error or cancelled, reject
          if (data.status === "error" || data.status === "cancelled") {
            cleanup();
            if (!isResolved) {
              isResolved = true;
              // Check if it's an OOM error
              const errorText = data.error || "Generation failed";
              let errorMessage = typeof errorText === 'string' ? errorText : String(errorText);
              if (errorText.toLowerCase().includes('out of memory') ||
                errorText.toLowerCase().includes('cuda out of memory') ||
                errorText.toLowerCase().includes('oom')) {
                errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
              }
              reject(new Error(JSON.stringify({
                status: 500,
                error: errorMessage,
                endpoint: `/api/generate/stream/${taskId}`
              })));
            }
            return;
          }
        } catch (parseError) {
          console.error('Failed to parse SSE event:', parseError);
        }
      };

      eventSource.onerror = async (error) => {
        console.warn('⚠️ [SSE] Connection error, attempting to recover with status polling...');

        // Try to recover by checking status and polling if task is still active
        try {
          const status = await this.getGenerationStatus(taskId);

          // If task is done, fetch result
          if (status.status === "done") {
            cleanup();
            if (!isResolved) {
              isResolved = true;
              this.getGenerationResult(taskId)
                .then(result => resolve(result))
                .catch(err => reject(err));
            }
            return;
          }

          // If task has error, reject
          if (status.status === "error" || status.status === "cancelled") {
            cleanup();
            if (!isResolved) {
              isResolved = true;
              const errorText = status.error || "Generation failed";
              let errorMessage = typeof errorText === 'string' ? errorText : String(errorText);
              if (errorText.toLowerCase().includes('out of memory') ||
                errorText.toLowerCase().includes('cuda out of memory') ||
                errorText.toLowerCase().includes('oom')) {
                errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
              }
              reject(new Error(JSON.stringify({
                status: 500,
                error: errorMessage,
                endpoint: `/api/generate/stream/${taskId}`
              })));
            }
            return;
          }

          // Task is still active, update progress and continue polling
          if (onProgress) {
            onProgress({
              current_step: status.current_step,
              total_steps: status.total_steps,
              status: status.status,
              progress: status.progress,
              loading_message: status.loading_message,
            });
          }

          // Start polling as fallback
          pollInterval = setInterval(async () => {
            try {
              const pollStatus = await this.getGenerationStatus(taskId);

              // Update progress
              if (onProgress) {
                onProgress({
                  current_step: pollStatus.current_step,
                  total_steps: pollStatus.total_steps,
                  status: pollStatus.status,
                  progress: pollStatus.progress,
                  loading_message: pollStatus.loading_message,
                });
              }

              // Check if done
              if (pollStatus.status === "done") {
                cleanup();
                if (!isResolved) {
                  isResolved = true;
                  this.getGenerationResult(taskId)
                    .then(result => resolve(result))
                    .catch(err => reject(err));
                }
              } else if (pollStatus.status === "error" || pollStatus.status === "cancelled") {
                cleanup();
                if (!isResolved) {
                  isResolved = true;
                  const errorText = pollStatus.error || "Generation failed";
                  let errorMessage = typeof errorText === 'string' ? errorText : String(errorText);
                  if (errorText.toLowerCase().includes('out of memory') ||
                    errorText.toLowerCase().includes('cuda out of memory') ||
                    errorText.toLowerCase().includes('oom')) {
                    errorMessage = 'GPU hết bộ nhớ (Out of Memory). Vui lòng thử lại sau hoặc giảm kích thước ảnh/số bước inference.';
                  }
                  reject(new Error(JSON.stringify({
                    status: 500,
                    error: errorMessage,
                    endpoint: `/api/generate/status/${taskId}`
                  })));
                }
              }
            } catch (pollError) {
              console.error('Error polling status:', pollError);
              cleanup();
              if (!isResolved) {
                isResolved = true;
                reject(pollError);
              }
            }
          }, 2000); // Poll every 2 seconds
        } catch (statusError) {
          // If status check also fails, reject
          cleanup();
          if (!isResolved) {
            isResolved = true;
            reject(new Error(JSON.stringify({
              status: 500,
              error: 'SSE connection error and status check failed',
              endpoint: `/api/generate/stream/${taskId}`
            })));
          }
        }
      };

      // Handle cancellation
      if (signal) {
        signal.addEventListener('abort', () => {
          cleanup();
          if (!isResolved) {
            isResolved = true;
            reject(new Error(JSON.stringify({
              status: 0,
              error: 'Request cancelled',
              error_type: 'cancelled',
              endpoint: `/api/generate/stream/${taskId}`
            })));
          }
        });
      }
    });
  }

  async getGenerationStatus(taskId: string): Promise<{
    status: string;
    progress: number;
    current_step?: number;
    total_steps?: number;
    error?: string;
    loading_message?: string;
  }> {
    const url = `${this.baseUrl}/api/generate/status/${taskId}`;
    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(JSON.stringify({
        status: response.status,
        error: errorData.detail || errorData.error || 'Failed to get generation status',
        endpoint: `/api/generate/status/${taskId}`
      }));
    }

    const data = await response.json();
    return {
      status: data.status || 'unknown',
      progress: data.progress || 0,
      current_step: data.current_step,
      total_steps: data.total_steps,
      error: data.error,
      loading_message: data.loading_message,
    };
  }

  async getGenerationResult(taskId: string): Promise<GenerationResponse> {
    const url = `${this.baseUrl}/api/generate/result/${taskId}`;
    const response = await fetch(url);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(JSON.stringify({
        status: response.status,
        error: errorData.detail || errorData.error || 'Failed to get generation result',
        endpoint: `/api/generate/result/${taskId}`
      }));
    }

    const result = await response.json();
    return {
      success: true,
      image: result.image,
      generation_time: 0, // Not provided by async endpoint
      model_used: "qwen-image-edit",
      parameters_used: {},
      debug_info: result.debug_info,
    };
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;