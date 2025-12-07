import { useState, useRef, useCallback, useEffect } from "react";
import { detectEdges, floodFill } from "../utils/edgeDetection";
import { apiService } from "../services/api";

// Command Pattern Interface
interface Command {
  execute(): void;
  undo(): void;
  redo(): void;
  hasContent(): boolean;
  isEmpty(): boolean; // Added for optimization
}

// Canvas state snapshot for commands
interface CanvasState {
  imageData: ImageData | null;
  hasContent: boolean;
}

// Concrete Command: Canvas Draw Operation
class CanvasDrawCommand implements Command {
  private canvas: HTMLCanvasElement;
  private previousState: CanvasState;
  private currentState: CanvasState;

  constructor(canvas: HTMLCanvasElement, previousState: CanvasState, currentState: CanvasState) {
    this.canvas = canvas;
    this.previousState = previousState;
    this.currentState = currentState;
  }

  private restoreCanvasState(state: CanvasState): void {
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (state.imageData && state.hasContent) {
        ctx.putImageData(state.imageData, 0, 0);
      }
    } catch (error) {
      console.error('Failed to restore canvas state:', error);
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  execute(): void {
    this.restoreCanvasState(this.currentState);
  }

  undo(): void {
    this.restoreCanvasState(this.previousState);
  }

  redo(): void {
    this.execute();
  }

  hasContent(): boolean {
    return this.currentState.hasContent;
  }

  isEmpty(): boolean {
    return !this.currentState.hasContent;
  }
}

// Concrete Command: Canvas Clear Operation
class CanvasClearCommand implements Command {
  private canvas: HTMLCanvasElement;
  private previousState: CanvasState;

  constructor(canvas: HTMLCanvasElement, previousState: CanvasState) {
    this.canvas = canvas;
    this.previousState = previousState;
  }

  execute(): void {
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (ctx) {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  undo(): void {
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (this.previousState.imageData) {
        ctx.putImageData(this.previousState.imageData, 0, 0);
      }
    } catch (error) {
      console.error('Failed to restore canvas state during clear undo:', error);
    }
  }

  redo(): void {
    this.execute();
  }

  hasContent(): boolean {
    return false; // Clear command always results in no content
  }

  isEmpty(): boolean {
    return true; // Clear command always results in empty canvas
  }
}

// Command History Manager
class CommandHistory {
  private commands: Command[] = [];
  private currentIndex: number = -1;
  private maxHistorySize: number;

  constructor(maxHistorySize: number = 20) {
    this.maxHistorySize = maxHistorySize;
  }

  executeCommand(command: Command): void {
    // Remove any future commands if we're not at the end
    if (this.currentIndex < this.commands.length - 1) {
      this.commands = this.commands.slice(0, this.currentIndex + 1);
    }

    // Add new command
    this.commands.push(command);
    this.currentIndex = this.commands.length - 1;

    // Execute the command after adding to history
    command.execute();

    // Limit history size
    if (this.commands.length > this.maxHistorySize) {
      this.commands.shift();
      this.currentIndex = Math.max(0, this.currentIndex - 1);
    }
  }

  undo(): boolean {
    if (this.canUndo()) {
      const command = this.commands[this.currentIndex];
      command.undo();
      this.currentIndex--;
      return true;
    }
    return false;
  }

  redo(): boolean {
    if (this.canRedo()) {
      this.currentIndex++;
      const command = this.commands[this.currentIndex];
      command.redo();
      return true;
    }
    return false;
  }

  canUndo(): boolean {
    return this.currentIndex >= 0;
  }

  canRedo(): boolean {
    return this.currentIndex < this.commands.length - 1;
  }

  hasContent(): boolean {
    if (this.currentIndex >= 0 && this.currentIndex < this.commands.length) {
      const command = this.commands[this.currentIndex];
      return command.hasContent();
    }
    return false;
  }

  getHistoryLength(): number {
    return this.commands.length;
  }

  getCurrentIndex(): number {
    return this.currentIndex;
  }

  clear(): void {
    this.commands = [];
    this.currentIndex = -1;
  }

  getHistoryInfo() {
    return {
      totalCommands: this.commands.length,
      currentIndex: this.currentIndex,
      canUndo: this.canUndo(),
      canRedo: this.canRedo()
    };
  }
}

export function useMasking(
  uploadedImage: string | null,
  imageDimensions: { width: number; height: number } | null,
  imageContainerRef: React.RefObject<HTMLDivElement | null>,
  transform: { scale: number },
  viewportZoom: number,
  imageRef?: React.RefObject<HTMLImageElement | null>,
  onNotification?: (type: 'success' | 'error' | 'info' | 'warning', message: string) => void,
  borderAdjustment: number = 0,
  skipMaskResetRef?: React.MutableRefObject<boolean> // Ref to skip mask reset (e.g., when returning to original)
) {
  // Masking state
  const [isMaskingMode, setIsMaskingMode] = useState(false);
  const [isMaskDrawing, setIsMaskDrawing] = useState(false);
  const [maskBrushSize, setMaskBrushSize] = useState(20); // Default 20% for better visibility
  const [maskToolType, setMaskToolType] = useState<'brush' | 'box'>('brush');
  const [enableEdgeDetection, setEnableEdgeDetection] = useState(false);
  const [enableFloodFill, setEnableFloodFill] = useState(false);
  const [edgeMask, setEdgeMask] = useState<ImageData | null>(null);
  // State to track actual canvas content (for display purposes, independent of command history)
  const [hasCanvasContent, setHasCanvasContent] = useState(false);

  // Box drawing state
  const boxStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const boxCurrentPosRef = useRef<{ x: number; y: number } | null>(null);

  // Smart masking state
  const [enableSmartMasking, setEnableSmartMasking] = useState(false);
  const [smartMaskModelType, setSmartMaskModelType] = useState<'segmentation' | 'birefnet'>('segmentation');
  const [smartMaskImageId, setSmartMaskImageId] = useState<string | null>(null);
  const [isSmartMaskLoading, setIsSmartMaskLoading] = useState(false);
  const [currentSmartMaskRequestId, setCurrentSmartMaskRequestId] = useState<string | null>(null);

  // Store stroke points for smart masking
  const strokePointsRef = useRef<Array<{ x: number; y: number }>>([]);
  const smartMaskDebounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Track click-only mode detection
  const strokeStartTimeRef = useRef<number>(0);
  const strokeStartPosRef = useRef<{ x: number; y: number } | null>(null);
  const strokeMovementRef = useRef<number>(0); // Total movement distance in pixels

  // Command Pattern for undo/redo
  const commandHistory = useRef(new CommandHistory(20));
  const [historyState, setHistoryState] = useState({
    canUndo: false,
    canRedo: false,
    hasContent: false,
    historyLength: 0,
    currentIndex: -1
  });

  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const edgeOverlayCanvasRef = useRef<HTMLCanvasElement>(null);

  // Track previous uploadedImage to detect actual changes (not just re-setting same value)
  const previousUploadedImageRef = useRef<string | null>(null);

  // Helper function to update history state
  const updateHistoryState = useCallback(() => {
    const history = commandHistory.current;
    const newState = {
      canUndo: history.canUndo(),
      canRedo: history.canRedo(),
      hasContent: history.hasContent(),
      historyLength: history.getHistoryLength(),
      currentIndex: history.getCurrentIndex()
    };

    setHistoryState(newState);
  }, []);

  // Generate edge mask from image
  const generateEdgeMask = useCallback(async (): Promise<ImageData | null> => {
    if (!imageRef?.current || !imageDimensions) return null;

    try {
      // Create a temporary canvas to draw the image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = imageDimensions.width;
      tempCanvas.height = imageDimensions.height;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) return null;

      // Draw the image to the canvas
      tempCtx.drawImage(imageRef.current, 0, 0, imageDimensions.width, imageDimensions.height);

      // Use detectEdges function from utils
      const edgeData = detectEdges(tempCanvas, 50); // threshold = 50
      return edgeData;
    } catch (error) {
      console.error('Error generating edge mask:', error);
      return null;
    }
  }, [imageRef, imageDimensions]);

  // Generate edge mask when enableEdgeDetection is true and image is available
  useEffect(() => {
    if (!enableEdgeDetection || !uploadedImage || !imageDimensions || !imageRef?.current) {
      setEdgeMask(null);
      return;
    }

    let isCancelled = false;

    // Generate edge mask asynchronously
    generateEdgeMask().then((mask) => {
      if (!isCancelled) {
        setEdgeMask(mask);
      }
    }).catch((error) => {
      if (!isCancelled) {
        console.error('Error generating edge mask:', error);
        setEdgeMask(null);
      }
    });

    return () => {
      isCancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enableEdgeDetection, uploadedImage, imageDimensions?.width, imageDimensions?.height, imageRef?.current]);

  // Reset mask history when uploaded image changes (but not when returning to original)
  useEffect(() => {
    // Skip reset if skipMaskResetRef is set to true (e.g., when returning to original)
    // Don't reset the flag here - let the resize canvas useEffect handle it after preserving content
    if (skipMaskResetRef?.current === true) {
      // Update ref, but don't clear mask and don't reset flag yet
      // Flag will be reset in resize canvas useEffect after content is preserved
      previousUploadedImageRef.current = uploadedImage;
      return;
    }

    // Only reset if image actually changed (not just re-setting same value)
    if (uploadedImage === previousUploadedImageRef.current) {
      // Image didn't actually change, just skip reset
      return;
    }

    // Image actually changed, update ref and reset mask
    previousUploadedImageRef.current = uploadedImage;

    // Only reset if we have a new image (not null)
    if (uploadedImage) {
      commandHistory.current.clear();
      updateHistoryState();
      setIsMaskingMode(false);
      setIsMaskDrawing(false);
      setEdgeMask(null); // Clear edge mask when image changes
      setSmartMaskImageId(null); // Clear smart mask image ID
      strokePointsRef.current = []; // Clear stroke points
      if (smartMaskDebounceTimerRef.current) {
        clearTimeout(smartMaskDebounceTimerRef.current);
      }
    }
  }, [uploadedImage, updateHistoryState, skipMaskResetRef]);

  // Clear command history when exiting masking mode
  useEffect(() => {
    // Skip clear if we're returning to original (mask should be preserved)
    if (skipMaskResetRef?.current === true) {
      return;
    }

    // Only clear history when transitioning from masking mode to non-masking mode
    // This ensures we don't clear history when entering masking mode
    if (!isMaskingMode) {
      commandHistory.current.clear();
      updateHistoryState();

      // Also clear the canvas when exiting masking mode
      // But only if we're not returning to original
      const canvas = maskCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    }
  }, [isMaskingMode, updateHistoryState, skipMaskResetRef]);

  // Initialize canvas context
  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      return;
    }

    // Set canvas size if image dimensions are available
    if (imageDimensions) {
      canvas.width = imageDimensions.width;
      canvas.height = imageDimensions.height;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      return;
    }

    // Set brush to maximum hardness with sharp edges
    ctx.lineCap = 'butt';  // Sharp line ends instead of round
    ctx.lineJoin = 'miter';  // Sharp corners instead of round
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha

    // Enable anti-aliasing for smooth brush edges (like Krita/Photoshop)
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Set initial drawing properties
    if (imageDimensions) {
      const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
      // Dynamic brush size: scale from 0.5% to 10% of base image size
      // This gives better range: for 1024px image, brush ranges from ~5px to ~100px
      const brushSize = (maskBrushSize / 100) * (baseImageSize / 5);
      ctx.lineWidth = brushSize;
    } else {
      ctx.lineWidth = 5; // Default line width
    }
    ctx.lineCap = 'round'; // Round brush tip like Krita/Photoshop
    ctx.lineJoin = 'round'; // Round line joins
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

    ctxRef.current = ctx;
  }, [imageDimensions, maskBrushSize]);

  // Update canvas drawing properties when brush settings change
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !imageDimensions) return;

    // Calculate brush size relative to the base image size
    const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
    // Dynamic brush size: scale from 0.5% to 10% of base image size
    // This gives better range: for 1024px image, brush ranges from ~5px to ~100px
    const brushSize = (maskBrushSize / 100) * (baseImageSize / 5);
    const opacity = 0.5; // Fixed 50% opacity

    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round'; // Round brush tip like Krita/Photoshop
    ctx.lineJoin = 'round'; // Round line joins
    ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
    ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;
    // Enable anti-aliasing for smooth brush edges
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
  }, [maskBrushSize, imageDimensions]);

  // Helper function to check if canvas has content (for display purposes)
  const checkCanvasHasContent = useCallback((): boolean => {
    const canvas = maskCanvasRef.current;
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      return false;
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      return false;
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Check full buffer to avoid missing sparse masks
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        if (r > 0 || g > 0 || b > 0 || a > 0) {
          return true;
        }
      }
      return false;
    } catch (e) {
      console.warn('Could not check canvas content:', e);
      return false;
    }
  }, []);

  // Helper function to capture current canvas state
  const captureCanvasState = useCallback((): CanvasState => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      console.warn('No canvas available for state capture');
      return { imageData: null, hasContent: false };
    }

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      console.warn('No canvas context available for state capture');
      return { imageData: null, hasContent: false };
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      let hasContent = false;

      // Check full buffer to avoid missing sparse masks
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        if (r > 0 || g > 0 || b > 0 || a > 0) {
          hasContent = true;
          break;
        }
      }

      return { imageData, hasContent };
    } catch (error) {
      console.error('Failed to capture canvas state:', error);
      return { imageData: null, hasContent: false };
    }
  }, []);

  // Helper functions for smart masking
  const calculateBoundingBox = useCallback((points: Array<{ x: number; y: number }>): [number, number, number, number] | null => {
    if (points.length === 0) return null;

    let minX = points[0].x;
    let minY = points[0].y;
    let maxX = points[0].x;
    let maxY = points[0].y;

    for (const point of points) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }

    return [minX, minY, maxX, maxY];
  }, []);

  const samplePointsFromStroke = useCallback(
    (points: Array<{ x: number; y: number }>, maxPoints: number = 10): Array<[number, number]> => {
      if (points.length === 0) return [];
      if (points.length <= maxPoints) {
        return points.map(p => [p.x, p.y]);
      }

      // Sample evenly distributed points
      const step = Math.floor(points.length / maxPoints);
      const sampled: Array<[number, number]> = [];

      for (let i = 0; i < points.length; i += step) {
        sampled.push([points[i].x, points[i].y]);
        if (sampled.length >= maxPoints) break;
      }

      // Always include first and last point
      if (sampled.length > 0) {
        sampled[0] = [points[0].x, points[0].y];
        sampled[sampled.length - 1] = [points[points.length - 1].x, points[points.length - 1].y];
      }

      return sampled;
    },
    []
  );

  const scaleCoordinatesToOriginal = useCallback(
    (
      coords: { bbox?: [number, number, number, number]; points?: Array<[number, number]> },
      canvasElement: HTMLCanvasElement,
      imageElement: HTMLImageElement | null
    ): { bbox?: [number, number, number, number]; points?: Array<[number, number]> } => {
      if (!imageElement || !imageDimensions) {
        return coords;
      }

      // Canvas coordinates are already in imageDimensions space
      // But we need to scale to naturalWidth/naturalHeight if they differ
      const naturalWidth = imageElement.naturalWidth;
      const naturalHeight = imageElement.naturalHeight;

      // If dimensions match, no scaling needed
      if (naturalWidth === imageDimensions.width && naturalHeight === imageDimensions.height) {
        return coords;
      }

      // Calculate scale factors from imageDimensions to natural dimensions
      const scaleX = naturalWidth / imageDimensions.width;
      const scaleY = naturalHeight / imageDimensions.height;

      // Scale coordinates with bounds checking
      const scaleCoord = (x: number, y: number): [number, number] => {
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const clampedX = Math.max(0, Math.min(scaledX, naturalWidth - 1));
        const clampedY = Math.max(0, Math.min(scaledY, naturalHeight - 1));

        // Log warning if coordinates were clamped
        if (scaledX !== clampedX || scaledY !== clampedY) {
          console.warn('Coordinates clamped to image bounds:', {
            original: [x, y],
            scaled: [scaledX, scaledY],
            clamped: [clampedX, clampedY],
            naturalSize: [naturalWidth, naturalHeight]
          });
        }

        return [clampedX, clampedY];
      };

      const result: { bbox?: [number, number, number, number]; points?: Array<[number, number]> } = {};

      if (coords.bbox) {
        const [xMin, yMin, xMax, yMax] = coords.bbox;
        const [scaledXMin, scaledYMin] = scaleCoord(xMin, yMin);
        const [scaledXMax, scaledYMax] = scaleCoord(xMax, yMax);

        // Ensure bbox is valid (min < max)
        const finalXMin = Math.min(scaledXMin, scaledXMax);
        const finalYMin = Math.min(scaledYMin, scaledYMax);
        const finalXMax = Math.max(scaledXMin, scaledXMax);
        const finalYMax = Math.max(scaledYMin, scaledYMax);

        result.bbox = [finalXMin, finalYMin, finalXMax, finalYMax];
      }

      if (coords.points) {
        result.points = coords.points.map(([x, y]) => scaleCoord(x, y));
      }

      return result;
    },
    [imageDimensions]
  );

  const getCanvasCoordinates = useCallback((e: React.MouseEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !imageDimensions) return { x: 0, y: 0 };

    // Get the canvas bounding rect
    // Note: Canvas is inside TransformLayer which applies transform.scale
    // The bounding rect already accounts for the transform applied by CSS
    const canvasRect = canvas.getBoundingClientRect();

    // Calculate the mouse position relative to the canvas CSS size
    // The canvas CSS size = imageDimensions * displayScale
    // But the canvas internal size = imageDimensions (no displayScale)
    const relativeX = (e.clientX - canvasRect.left) / canvasRect.width;
    const relativeY = (e.clientY - canvasRect.top) / canvasRect.height;

    // Convert to canvas internal coordinates
    // Canvas internal resolution = imageDimensions (no displayScale)
    // CSS size = imageDimensions * displayScale
    // Transform is applied at TransformLayer level, bounding rect already accounts for it
    const x = relativeX * imageDimensions.width;
    const y = relativeY * imageDimensions.height;

    return { x, y };
  }, [imageDimensions]);

  // Store initial state when starting to draw (for command creation)
  const initialDrawState = useRef<CanvasState | null>(null);

  // Auto detect edge and fill after stroke
  const autoDetectAndFillAfterStroke = useCallback(async () => {
    const canvas = maskCanvasRef.current;
    const ctx = ctxRef.current;
    const edgeCanvas = edgeOverlayCanvasRef.current;
    if (!canvas || !ctx || !imageRef?.current || !imageDimensions) return;

    // Only proceed if edge detection or flood fill is enabled
    if (!enableEdgeDetection && !enableFloodFill) {
      // Clear edge overlay if disabled
      if (edgeCanvas) {
        const edgeCtx = edgeCanvas.getContext('2d', { willReadFrequently: true });
        if (edgeCtx) {
          edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);
        }
      }
      return;
    }

    try {
      // Get current mask data
      const maskData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      // Find all pixels that have mask (red pixels with alpha > 0)
      const maskPixels: Array<{ x: number; y: number }> = [];
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const idx = (y * canvas.width + x) * 4;
          // Check if pixel has mask (red channel > 0 and alpha > 0)
          if (maskData.data[idx] > 0 && maskData.data[idx + 3] > 0) {
            maskPixels.push({ x, y });
          }
        }
      }

      if (maskPixels.length === 0) return;

      // Create temporary canvas for edge detection in masked region
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = imageDimensions.width;
      tempCanvas.height = imageDimensions.height;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) return;

      // Draw the image
      tempCtx.drawImage(imageRef.current, 0, 0, imageDimensions.width, imageDimensions.height);

      // Detect edges in the image
      const detectedEdges = detectEdges(tempCanvas, 50);
      if (!detectedEdges) return;

      // Initialize edge overlay canvas
      if (edgeCanvas) {
        edgeCanvas.width = imageDimensions.width;
        edgeCanvas.height = imageDimensions.height;
        const edgeCtx = edgeCanvas.getContext('2d', { willReadFrequently: true });
        if (!edgeCtx) return;

        // Clear previous edge overlay
        edgeCtx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);

        // Draw edges only in masked regions (yellow/cyan overlay)
        const edgeData = detectedEdges.data;
        const edgeImageData = edgeCtx.createImageData(edgeCanvas.width, edgeCanvas.height);

        for (const { x, y } of maskPixels) {
          const edgeIdx = (y * edgeCanvas.width + x) * 4;
          const edgeValue = edgeData[edgeIdx];

          if (edgeValue > 128) {
            // Draw edge in yellow/cyan color
            const pixelIdx = (y * edgeCanvas.width + x) * 4;
            edgeImageData.data[pixelIdx] = 255;     // R - Yellow
            edgeImageData.data[pixelIdx + 1] = 255; // G
            edgeImageData.data[pixelIdx + 2] = 0;   // B
            edgeImageData.data[pixelIdx + 3] = 150; // A - Semi-transparent
          }
        }

        edgeCtx.putImageData(edgeImageData, 0, 0);
      }

      // If flood fill is enabled, fill mask to edges
      if (enableFloodFill && detectedEdges) {
        // Get image data for flood fill
        const imageData = tempCtx.getImageData(0, 0, imageDimensions.width, imageDimensions.height);

        // For each mask pixel, perform flood fill to edge
        // We'll use the center of the mask region as starting point
        if (maskPixels.length > 0) {
          const centerX = Math.floor(
            maskPixels.reduce((sum, p) => sum + p.x, 0) / maskPixels.length
          );
          const centerY = Math.floor(
            maskPixels.reduce((sum, p) => sum + p.y, 0) / maskPixels.length
          );

          const fillColor = { r: 255, g: 0, b: 0, a: 128 };
          const filledData = floodFill(
            imageData,
            centerX,
            centerY,
            fillColor,
            detectedEdges,
            10 // tolerance
          );

          // Merge filled data with existing mask
          const currentMaskData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let i = 0; i < filledData.data.length; i += 4) {
            // If filled pixel has mask, add to current mask
            if (filledData.data[i] > 0 || filledData.data[i + 3] > 0) {
              currentMaskData.data[i] = 255; // R
              currentMaskData.data[i + 1] = 0; // G
              currentMaskData.data[i + 2] = 0; // B
              currentMaskData.data[i + 3] = Math.max(
                currentMaskData.data[i + 3],
                filledData.data[i + 3]
              ); // A
            }
          }
          ctx.putImageData(currentMaskData, 0, 0);
        }
      }
    } catch (error) {
      console.error('Error in auto detect and fill:', error);
    }
  }, [imageRef, imageDimensions, enableEdgeDetection, enableFloodFill]);

  const startDrawing = useCallback((e: React.MouseEvent) => {
    if (!isMaskingMode || !uploadedImage || !imageDimensions) return;

    // Lock canvas if smart mask is loading
    if (isSmartMaskLoading) return;

    e.preventDefault();
    e.stopPropagation();

    // Capture initial state before drawing for command creation
    initialDrawState.current = captureCanvasState();

    const { x, y } = getCanvasCoordinates(e);
    const ctx = ctxRef.current;
    if (!ctx) return;

    // Ensure opacity and composite operation are set correctly before starting to draw
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha
    // Ensure strokeStyle with opacity is set (should already be set, but ensure it)
    if (!ctx.strokeStyle || ctx.strokeStyle === 'rgba(0, 0, 0, 0)') {
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    }

    if (maskToolType === 'box') {
      // Box mode: store start position
      boxStartPosRef.current = { x, y };
      boxCurrentPosRef.current = { x, y };
      setIsMaskDrawing(true);
    } else {
      // Brush mode: start path
      // Reset stroke tracking for smart masking
      if (enableSmartMasking) {
        strokePointsRef.current = [{ x, y }];
        strokeStartTimeRef.current = Date.now();
        strokeStartPosRef.current = { x, y };
        strokeMovementRef.current = 0;
      }

      ctx.beginPath();
      ctx.moveTo(x, y);
      setIsMaskDrawing(true);
    }
  }, [isMaskingMode, uploadedImage, imageDimensions, getCanvasCoordinates, captureCanvasState, enableSmartMasking, isSmartMaskLoading, maskToolType]);

  const draw = useCallback((e: React.MouseEvent) => {
    if (!isMaskDrawing || !isMaskingMode) return;

    // Lock canvas if smart mask is loading
    if (isSmartMaskLoading) return;

    e.preventDefault();

    const { x, y } = getCanvasCoordinates(e);
    const ctx = ctxRef.current;
    const canvas = maskCanvasRef.current;
    if (!ctx || !canvas) {
      return;
    }

    if (maskToolType === 'box') {
      // Box mode: draw preview box
      boxCurrentPosRef.current = { x, y };

      // Restore canvas state and draw preview box
      if (initialDrawState.current?.imageData) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(initialDrawState.current.imageData, 0, 0);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      // Draw preview box
      if (boxStartPosRef.current) {
        const startX = boxStartPosRef.current.x;
        const startY = boxStartPosRef.current.y;
        const width = x - startX;
        const height = y - startY;

        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;

        ctx.fillRect(startX, startY, width, height);
        ctx.strokeRect(startX, startY, width, height);
      }
    } else {
      // Brush mode: continue drawing path
      // Ensure opacity and composite operation are set correctly before drawing
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha
      // Ensure brush properties are set for round, smooth brush
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      // Ensure strokeStyle with opacity is set (should already be set, but ensure it)
      if (!ctx.strokeStyle || ctx.strokeStyle === 'rgba(0, 0, 0, 0)') {
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
      }

      // Store stroke points and track movement for smart masking
      if (enableSmartMasking) {
        strokePointsRef.current.push({ x, y });

        // Calculate movement distance from start position
        if (strokeStartPosRef.current) {
          const dx = x - strokeStartPosRef.current.x;
          const dy = y - strokeStartPosRef.current.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          strokeMovementRef.current = Math.max(strokeMovementRef.current, distance);
        }
      }

      ctx.lineTo(x, y);
      ctx.stroke();
    }
  }, [isMaskDrawing, isMaskingMode, getCanvasCoordinates, enableSmartMasking, isSmartMaskLoading, maskToolType]);

  // Helper function to convert binary mask (white/black) to red transparent mask
  const convertBinaryMaskToRedTransparent = useCallback((
    binaryMaskImage: HTMLImageElement,
    width: number,
    height: number
  ): ImageData => {
    // Create temporary canvas to process the mask
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    if (!tempCtx) throw new Error('Failed to get canvas context');

    // Draw the binary mask onto temp canvas
    tempCtx.drawImage(binaryMaskImage, 0, 0, width, height);

    // Get image data
    const imageData = tempCtx.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Convert binary mask to red transparent
    // White pixels (255, 255, 255) → rgba(255, 0, 0, 0.5)
    // Black pixels (0, 0, 0) → transparent (rgba(0, 0, 0, 0))
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];

      // Check if pixel is white (mask area) or black (non-mask area)
      // Use grayscale value to determine
      const grayscale = (r + g + b) / 3;

      if (grayscale > 128 && a > 0) {
        // White pixel (mask area) → red transparent
        data[i] = 255;     // R
        data[i + 1] = 0;   // G
        data[i + 2] = 0;   // B
        data[i + 3] = 128; // A (0.5 opacity = 128/255)
      } else {
        // Black pixel or transparent → fully transparent
        data[i] = 0;       // R
        data[i + 1] = 0;   // G
        data[i + 2] = 0;   // B
        data[i + 3] = 0;   // A (fully transparent)
      }
    }

    return imageData;
  }, []);

  // Function to merge smart mask into canvas
  // Each smart mask is created independently based on original image only
  // When merged, it preserves existing masks and adds new mask on top
  // If masks overlap spatially, they will merge visually (expected behavior)
  // If masks don't overlap, they remain separate (no "dính chùm" issue)
  const mergeSmartMask = useCallback(async (maskBase64: string) => {
    const canvas = maskCanvasRef.current;
    const ctx = ctxRef.current;
    if (!canvas || !ctx) return;

    try {
      // Create image from base64 mask
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = `data:image/png;base64,${maskBase64}`;
      });

      // Get current composite operation (preserve user's mode)
      const currentComposite = ctx.globalCompositeOperation;

      // Convert binary mask to red transparent mask
      // This mask is created independently from original image, not from previous masks
      const redTransparentMask = convertBinaryMaskToRedTransparent(
        img,
        canvas.width,
        canvas.height
      );

      // Create temporary canvas to convert ImageData to Image
      // This allows us to use drawImage with composite operations instead of putImageData
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
      if (!tempCtx) throw new Error('Failed to get temp canvas context');

      // Put the converted mask data onto temp canvas
      tempCtx.putImageData(redTransparentMask, 0, 0);

      // Use source-over to add mask (giống như stroke)
      // This preserves existing mask content and adds new smart mask on top
      // Each mask is independent - only merges visually if they overlap spatially
      ctx.globalCompositeOperation = 'source-over';

      // Draw converted mask onto canvas using drawImage (not putImageData)
      // This will properly merge with existing masks, keeping all masks visible
      ctx.drawImage(tempCanvas, 0, 0);

      // Restore original composite operation
      ctx.globalCompositeOperation = currentComposite;
    } catch (error) {
      console.error('Failed to merge smart mask:', error);
      throw error; // Re-throw to allow error handling upstream
    }
  }, [convertBinaryMaskToRedTransparent]);

  // Function to generate smart mask from box
  const generateSmartMaskFromBox = useCallback(async (bbox: [number, number, number, number]) => {
    if (!enableSmartMasking || !uploadedImage || !imageDimensions || !imageRef?.current) return;

    setIsSmartMaskLoading(true);

    try {
      // Scale bbox to original image dimensions
      const scaledCoords = scaleCoordinatesToOriginal(
        { bbox },
        maskCanvasRef.current!,
        imageRef.current
      );

      if (!scaledCoords.bbox) {
        console.error('Failed to scale bbox coordinates');
        return;
      }

      // Call API with bbox - Always send image to ensure independence from previous masks
      // Each smart mask request should be independent, based on original image only
      const result = await apiService.generateSmartMask(
        uploadedImage, // Always send image (not image_id) for independent segmentation
        null, // Don't use image_id - each mask is independent
        scaledCoords.bbox,
        undefined, // No points for box mode
        borderAdjustment, // border_adjustment
        false, // use_blur
        smartMaskModelType // model_type
      );

      if (result.success && result.mask_base64) {
        // Don't save image_id - each mask request is independent
        // Always send original image to ensure segmentation is based on original, not previous masks

        // Restore canvas to state before box was drawn (removes the box preview)
        // This ensures only the segmented mask area is kept, not the box preview
        const canvas = maskCanvasRef.current;
        const ctx = ctxRef.current;
        if (canvas && ctx && initialDrawState.current?.imageData) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.putImageData(initialDrawState.current.imageData, 0, 0);
        }

        // Track request_id for cancellation
        if (result.request_id) {
          setCurrentSmartMaskRequestId(result.request_id);
        }

        // Merge mask into canvas (now on clean canvas without box preview)
        await mergeSmartMask(result.mask_base64);

        // Update command history
        if (canvas && initialDrawState.current) {
          const currentState = captureCanvasState();
          const drawCommand = new CanvasDrawCommand(canvas, initialDrawState.current, currentState);
          commandHistory.current.executeCommand(drawCommand);
          updateHistoryState();
          initialDrawState.current = null;
        }

        // Show success notification
        if (onNotification) {
          onNotification('success', 'Smart mask generated successfully');
        }
      } else {
        const errorMsg = result.error || 'Smart mask generation failed';
        console.error('Smart mask generation failed:', errorMsg);
        if (onNotification) {
          onNotification('error', errorMsg);
        }
      }
    } catch (error: unknown) {
      console.error('Error generating smart mask from box:', error);
      let errorMessage = 'Failed to generate smart mask';

      // Parse error message (may be JSON stringified from API service)
      let parsedError: { error?: string } | null = null;
      const errorObj = error as { message?: string };
      if (errorObj?.message) {
        try {
          parsedError = JSON.parse(errorObj.message) as { error?: string };
        } catch {
          // Not JSON, use as is
          parsedError = { error: errorObj.message };
        }
      }

      // Check for "No masks found" error - this is a normal case, show friendly message
      const errorText = parsedError?.error || errorObj?.message || '';

      // If image_id expired, clear it so next request will send image
      if (errorText.includes('Image ID not found') || errorText.includes('expired')) {
        console.log('🔄 Image ID expired, clearing cached image_id');
        setSmartMaskImageId(null);
      }

      if (errorText.includes('No masks found') || errorText.includes('did not generate')) {
        // This is a normal case, not a real error - just show info message
        if (onNotification) {
          onNotification('info', 'FastSAM không detect được gì với vị trí mask, hãy chọn vị trí khác hay vật thể khác');
        }
        return; // Don't show error, just return silently
      }

      // Provide user-friendly error messages for other errors
      if (errorText.includes('Network') || errorText.includes('fetch')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      } else if (errorText.includes('timeout')) {
        errorMessage = 'Request timed out. Please try again.';
      } else if (parsedError?.error) {
        errorMessage = parsedError.error;
      } else if (errorObj?.message) {
        errorMessage = errorObj.message;
      }

      if (onNotification) {
        onNotification('error', errorMessage);
      }
    } finally {
      setIsSmartMaskLoading(false);
    }
  }, [
    enableSmartMasking,
    smartMaskModelType,
    uploadedImage,
    imageDimensions,
    imageRef,
    scaleCoordinatesToOriginal,
    mergeSmartMask,
    captureCanvasState,
    updateHistoryState,
    onNotification,
    borderAdjustment
  ]);

  // Function to generate smart mask from stroke
  const generateSmartMaskFromStroke = useCallback(async () => {
    if (!enableSmartMasking || !uploadedImage || !imageDimensions || !imageRef?.current) return;

    // BiRefNet does not support stroke/points, only bbox
    if (smartMaskModelType === 'birefnet') {
      console.warn('BiRefNet does not support stroke/points. Please use box tool instead.');
      return;
    }

    const points = strokePointsRef.current;
    if (points.length === 0) return;

    setIsSmartMaskLoading(true);

    try {
      // For single point (click-only), use point mode directly
      // For multiple points (stroke), calculate bbox and sample points
      const isSinglePoint = points.length === 1;
      const bbox = isSinglePoint ? null : calculateBoundingBox(points);
      const sampledPoints: Array<[number, number]> = isSinglePoint
        ? [[points[0].x, points[0].y]]
        : samplePointsFromStroke(points, 10);

      // Determine whether to use bbox or points mode
      let usePoints = isSinglePoint;
      let scaledCoords: { bbox?: [number, number, number, number]; points?: Array<[number, number]> } = {};

      if (!isSinglePoint && bbox) {
        const [xMin, yMin, xMax, yMax] = bbox;
        const bboxWidth = xMax - xMin;
        const bboxHeight = yMax - yMin;

        // If bbox is too small (< 10x10px), use point mode instead
        if (bboxWidth < 10 || bboxHeight < 10) {
          usePoints = true;
        } else {
          // Use bbox mode - scale coordinates to original image dimensions
          scaledCoords = scaleCoordinatesToOriginal(
            { bbox, points: sampledPoints },
            maskCanvasRef.current!,
            imageRef.current
          );
        }
      }

      if (usePoints && sampledPoints.length > 0) {
        // Use point mode - scale points to original image dimensions
        scaledCoords = scaleCoordinatesToOriginal(
          { points: sampledPoints },
          maskCanvasRef.current!,
          imageRef.current
        );
      }

      // Call API - Always send image to ensure independence from previous masks
      // Each smart mask request should be independent, based on original image only
      const result = await apiService.generateSmartMask(
        uploadedImage, // Always send image (not image_id) for independent segmentation
        null, // Don't use image_id - each mask is independent
        scaledCoords.bbox,
        scaledCoords.points,
        borderAdjustment, // border_adjustment
        false, // use_blur
        smartMaskModelType // model_type
      );

      if (result.success && result.mask_base64) {
        // Track request_id for cancellation
        if (result.request_id) {
          setCurrentSmartMaskRequestId(result.request_id);
        }

        // Don't save image_id - each mask request is independent
        // Always send original image to ensure segmentation is based on original, not previous masks

        // Restore canvas to state before stroke was drawn (removes the stroke)
        // This ensures only the segmented mask area is kept, not the stroke itself
        const canvas = maskCanvasRef.current;
        const ctx = ctxRef.current;
        if (canvas && ctx && initialDrawState.current?.imageData) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.putImageData(initialDrawState.current.imageData, 0, 0);
        }

        // Merge mask into canvas (now on clean canvas without stroke)
        await mergeSmartMask(result.mask_base64);

        // Update command history
        if (canvas && initialDrawState.current) {
          const currentState = captureCanvasState();
          const drawCommand = new CanvasDrawCommand(canvas, initialDrawState.current, currentState);
          commandHistory.current.executeCommand(drawCommand);
          updateHistoryState();
          initialDrawState.current = null;
        }

        // Show success notification
        if (onNotification) {
          onNotification('success', 'Smart mask generated successfully');
        }
      } else {
        const errorMsg = result.error || 'Smart mask generation failed';
        console.error('Smart mask generation failed:', errorMsg);
        if (onNotification) {
          onNotification('error', errorMsg);
        }
      }
    } catch (error: unknown) {
      console.error('Error generating smart mask:', error);
      let errorMessage = 'Failed to generate smart mask';

      // Parse error message (may be JSON stringified from API service)
      let parsedError: { error?: string } | null = null;
      const errorObj = error as { message?: string };
      if (errorObj?.message) {
        try {
          parsedError = JSON.parse(errorObj.message) as { error?: string };
        } catch {
          // Not JSON, use as is
          parsedError = { error: errorObj.message };
        }
      }

      // Check for "No masks found" error - this is a normal case, show friendly message
      const errorText = parsedError?.error || errorObj?.message || '';

      // If image_id expired, clear it so next request will send image
      if (errorText.includes('Image ID not found') || errorText.includes('expired')) {
        console.log('🔄 Image ID expired, clearing cached image_id');
        setSmartMaskImageId(null);
      }

      if (errorText.includes('No masks found') || errorText.includes('did not generate')) {
        // This is a normal case, not a real error - just show info message
        if (onNotification) {
          onNotification('info', 'FastSAM không detect được gì với vị trí mask, hãy chọn vị trí khác hay vật thể khác');
        }
        return; // Don't show error, just return silently
      }

      // Provide user-friendly error messages for other errors
      if (errorText.includes('Network') || errorText.includes('fetch')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      } else if (errorText.includes('timeout')) {
        errorMessage = 'Request timed out. Please try again.';
      } else if (parsedError?.error) {
        errorMessage = parsedError.error;
      } else if (errorObj?.message) {
        errorMessage = errorObj.message;
      }

      if (onNotification) {
        onNotification('error', errorMessage);
      }
    } finally {
      setIsSmartMaskLoading(false);
      strokePointsRef.current = [];
    }
  }, [
    enableSmartMasking,
    smartMaskModelType,
    uploadedImage,
    imageDimensions,
    imageRef,
    calculateBoundingBox,
    samplePointsFromStroke,
    scaleCoordinatesToOriginal,
    mergeSmartMask,
    captureCanvasState,
    updateHistoryState,
    onNotification,
    borderAdjustment
  ]);

  const stopDrawing = useCallback(() => {
    const ctx = ctxRef.current;
    const canvas = maskCanvasRef.current;

    if (maskToolType === 'box') {
      // Box mode: handle box drawing
      if (canvas && ctx && boxStartPosRef.current && boxCurrentPosRef.current) {
        const startX = boxStartPosRef.current.x;
        const startY = boxStartPosRef.current.y;
        const endX = boxCurrentPosRef.current.x;
        const endY = boxCurrentPosRef.current.y;

        // Calculate box dimensions
        const width = endX - startX;
        const height = endY - startY;

        // Only process if box has meaningful size
        if (Math.abs(width) > 2 && Math.abs(height) > 2) {
          // Calculate bbox [xMin, yMin, xMax, yMax]
          const xMin = Math.min(startX, endX);
          const yMin = Math.min(startY, endY);
          const xMax = Math.max(startX, endX);
          const yMax = Math.max(startY, endY);
          const bbox: [number, number, number, number] = [xMin, yMin, xMax, yMax];

          // Handle smart masking for box
          if (enableSmartMasking) {
            // Generate smart mask from box bbox
            generateSmartMaskFromBox(bbox);

            // Don't fill box manually - smart mask will be merged
            // Reset box state
            boxStartPosRef.current = null;
            boxCurrentPosRef.current = null;
            setIsMaskDrawing(false);
            return;
          } else {
            // No smart masking: fill the box manually
            // Restore canvas state and fill box
            if (initialDrawState.current?.imageData) {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
              ctx.putImageData(initialDrawState.current.imageData, 0, 0);
            }

            // Fill box with mask
            ctx.globalCompositeOperation = 'source-over';
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.fillRect(xMin, yMin, xMax - xMin, yMax - yMin);

            // Save to command history
            if (initialDrawState.current) {
              const currentState = captureCanvasState();
              const drawCommand = new CanvasDrawCommand(canvas, initialDrawState.current, currentState);
              commandHistory.current.executeCommand(drawCommand);
              updateHistoryState();
              initialDrawState.current = null;
            }
          }
        }
      }

      // Reset box state
      boxStartPosRef.current = null;
      boxCurrentPosRef.current = null;
      setIsMaskDrawing(false);
      return;
    }

    // Brush mode: continue with existing logic
    if (ctx) {
      ctx.closePath();
    }

    // Handle smart masking
    if (enableSmartMasking && strokePointsRef.current.length > 0) {
      // Detect click-only mode (single click without significant drag)
      const strokeDuration = Date.now() - strokeStartTimeRef.current;
      const isClickOnly = strokeDuration < 100 && strokeMovementRef.current < 5; // < 100ms and < 5px movement

      // Clear existing debounce timer
      if (smartMaskDebounceTimerRef.current) {
        clearTimeout(smartMaskDebounceTimerRef.current);
      }

      if (isClickOnly) {
        // Click-only: generate mask immediately without debounce
        generateSmartMaskFromStroke();
      } else {
        // Stroke mode: debounce smart mask generation (400ms)
        smartMaskDebounceTimerRef.current = setTimeout(() => {
          generateSmartMaskFromStroke();
        }, 400);
      }

      // Don't save stroke as command yet - will be saved after smart mask is merged
      setIsMaskDrawing(false);
      return;
    }

    // Create and execute draw command if we have initial state and canvas
    if (canvas && initialDrawState.current) {
      // Capture current state after drawing
      const currentState = captureCanvasState();

      const drawCommand = new CanvasDrawCommand(canvas, initialDrawState.current, currentState);
      commandHistory.current.executeCommand(drawCommand);
      updateHistoryState();

      initialDrawState.current = null;

      // Normalize opacity after drawing to ensure consistent visual appearance
      // This prevents opacity accumulation when strokes overlap
      if (canvas) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          // Get current image data
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;

          // Normalize opacity: any pixel with alpha > 0 should be set to rgba(255, 0, 0, 0.5)
          for (let i = 0; i < data.length; i += 4) {
            if (data[i + 3] > 0) {
              // Pixel has been drawn - normalize to consistent opacity
              data[i] = 255;     // R
              data[i + 1] = 0;   // G
              data[i + 2] = 0;   // B
              data[i + 3] = 128; // A (0.5 opacity = 128/255)
            }
          }

          // Put normalized data back
          ctx.putImageData(imageData, 0, 0);
        }
      }

      // Auto detect edge and fill after stroke
      if (enableEdgeDetection || enableFloodFill) {
        setTimeout(() => {
          autoDetectAndFillAfterStroke();
        }, 50); // Small delay to ensure canvas is updated
      }
    }

    setIsMaskDrawing(false);
  }, [updateHistoryState, enableEdgeDetection, enableFloodFill, autoDetectAndFillAfterStroke, enableSmartMasking, generateSmartMaskFromStroke, generateSmartMaskFromBox, maskToolType, captureCanvasState]);

  // Handle canvas resizing when image dimensions change  
  useEffect(() => {
    if (!maskCanvasRef.current || !imageDimensions) return;

    const canvas = maskCanvasRef.current;

    const resizeCanvas = () => {
      // Use actual image dimensions, not container size
      // This ensures mask matches the original image size when exported
      const newWidth = imageDimensions.width;
      const newHeight = imageDimensions.height;

      // Check if canvas dimensions are changing
      const dimensionsChanged = canvas.width !== newWidth || canvas.height !== newHeight;

      // Check if we're returning to original (should preserve mask)
      const isReturningToOriginal = skipMaskResetRef?.current === true;

      // Store canvas content before resize if:
      // 1. Dimensions are changing, OR
      // 2. We're returning to original (even if dimensions don't change, setting canvas.width/height clears it)
      let imageData = null;
      if ((dimensionsChanged || isReturningToOriginal) && canvas.width > 0 && canvas.height > 0) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          try {
            imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          } catch (e) {
            console.warn('Could not save canvas content:', e);
          }
        }
      }

      // Set canvas internal resolution to match the actual image dimensions
      // NOTE: Setting canvas.width/height ALWAYS clears the canvas in HTML5, even if values are the same
      // This is why we need to preserve content even when dimensions don't change
      canvas.width = newWidth;
      canvas.height = newHeight;

      // Canvas positioning is handled by CSS (absolute inset-0 inside transform div)
      // Canvas now fills the entire parent div automatically

      // Always re-initialize context settings after canvas resize
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        // Re-initialize context settings for maximum brush hardness
        ctx.lineCap = 'butt';  // Sharp line ends
        ctx.lineJoin = 'miter';  // Sharp corners
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0; // Opacity is handled by rgba color, not globalAlpha

        // Enable anti-aliasing for smooth brush edges (like Krita/Photoshop)
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // Apply current drawing properties
        if (imageDimensions) {
          const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
          // Dynamic brush size: scale from 0.5% to 10% of base image size
          const brushSize = (maskBrushSize / 100) * (baseImageSize / 5);
          const opacity = 0.5; // Fixed 50% opacity

          ctx.lineWidth = brushSize;
          ctx.lineCap = 'round'; // Round brush tip
          ctx.lineJoin = 'round'; // Round line joins
          ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
          ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;
        }

        // Restore canvas content if it was saved
        // Restore if: dimensions changed OR we're returning to original
        if (imageData && (dimensionsChanged || isReturningToOriginal)) {
          try {
            ctx.putImageData(imageData, 0, 0);
            // After restoring, update hasCanvasContent state
            // Use setTimeout to ensure state update happens after canvas operation
            setTimeout(() => {
              const hasContent = checkCanvasHasContent();
              setHasCanvasContent(hasContent);
            }, 0);
          } catch (e) {
            console.warn('Could not restore canvas content:', e);
          }
        }

        // Reset the flag after preserving canvas content (if we were returning to original)
        if (isReturningToOriginal && skipMaskResetRef) {
          skipMaskResetRef.current = false;
        }

        ctxRef.current = ctx;
      }
    };

    // Initial resize - try immediately, then retry after a short delay
    resizeCanvas();
    const timer = setTimeout(resizeCanvas, 50);
    const timer2 = setTimeout(resizeCanvas, 200);

    // Listen for window resize
    window.addEventListener('resize', resizeCanvas);

    return () => {
      clearTimeout(timer);
      clearTimeout(timer2);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [uploadedImage, transform.scale, imageDimensions, isMaskingMode, maskBrushSize, skipMaskResetRef]);

  // Check canvas content periodically to update hasCanvasContent state
  useEffect(() => {
    if (!maskCanvasRef.current || !imageDimensions) {
      setHasCanvasContent(false);
      return;
    }

    // Check canvas content after a short delay to allow canvas operations to complete
    const checkContent = () => {
      const hasContent = checkCanvasHasContent();
      setHasCanvasContent(hasContent);
    };

    // Check immediately and after a delay
    checkContent();
    const timer = setTimeout(checkContent, 100);
    const timer2 = setTimeout(checkContent, 300);

    return () => {
      clearTimeout(timer);
      clearTimeout(timer2);
    };
  }, [imageDimensions, checkCanvasHasContent, isMaskingMode]);

  // Canvas scaling is handled by the container, no separate zoom effect needed
  // useEffect(() => {
  //   const canvas = maskCanvasRef.current;
  //   if (!canvas) return;
  //   
  //   canvas.style.transform = `scale(${viewportZoom})`;
  //   canvas.style.transformOrigin = 'center center';
  // }, [viewportZoom]);

  const undoMask = useCallback(() => {
    if (commandHistory.current.undo()) {
      updateHistoryState();
    }
  }, [updateHistoryState]);

  const redoMask = useCallback(() => {
    if (commandHistory.current.redo()) {
      updateHistoryState();
    }
  }, [updateHistoryState]);

  const clearMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    // Capture current state before clearing
    const currentState = captureCanvasState();

    // Optimization: Don't create clear command if canvas is already empty
    if (!currentState.hasContent) {
      return;
    }

    // Create and execute clear command
    const clearCommand = new CanvasClearCommand(canvas, currentState);
    commandHistory.current.executeCommand(clearCommand);
    updateHistoryState();
  }, [captureCanvasState, updateHistoryState]);

  const resetMaskHistory = useCallback(() => {
    // Reset all mask-related state when image changes
    commandHistory.current.clear();
    updateHistoryState();
    setIsMaskingMode(false);
    setIsMaskDrawing(false);

    // Clear the canvas if it exists
    const canvas = maskCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, [updateHistoryState]);

  const toggleMaskingMode = useCallback(() => {
    const wasInMaskingMode = isMaskingMode;

    setIsMaskingMode(!isMaskingMode);

    if (wasInMaskingMode) {
      // Exiting masking mode
      setIsMaskDrawing(false);

      // Clear command history immediately for responsive UI
      commandHistory.current.clear();
      updateHistoryState();

      // Clear the canvas content
      const canvas = maskCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    } else {
      // Entering masking mode - ensure drawing state is reset
      setIsMaskDrawing(false);
    }
  }, [isMaskingMode, updateHistoryState]);

  // Hàm chuyển đổi Mask đỏ/trong suốt (UI) thành Mask Đen/Trắng chuẩn cho AI
  // Quy ước: TRẮNG = vùng cần sửa (mask area), ĐEN = vùng giữ nguyên (keep area)
  const getFinalMask = useCallback(async (): Promise<string | null> => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !imageDimensions) return null;

    // 1. Tạo canvas tạm để xử lý với kích thước đúng (bằng imageDimensions)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageDimensions.width;
    tempCanvas.height = imageDimensions.height;
    const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;

    // 2. Tô nền ĐEN (Vùng giữ nguyên - không sửa)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // 3. Lấy dữ liệu pixel từ canvas vẽ hiện tại (đang là màu đỏ trong suốt cho UI)
    const originalCtx = canvas.getContext('2d', { willReadFrequently: true });
    if (!originalCtx) return null;

    // Canvas đã có kích thước = imageDimensions, nên không cần scale
    const imageData = originalCtx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // 4. Xử lý Binary Mask: Chuyển vùng có vẽ (alpha > 0) thành TRẮNG (vùng cần sửa)
    // Tạo ImageData mới cho canvas tạm
    const newImageData = ctx.createImageData(tempCanvas.width, tempCanvas.height);
    const newData = newImageData.data;

    // Canvas size đã match imageDimensions, nên copy trực tiếp
    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3]; // Lấy độ trong suốt của nét vẽ hiện tại

      if (alpha > 0) {
        // Có nét vẽ -> Set thành TRẮNG (255, 255, 255, 255) = Vùng cần sửa
        newData[i] = 255;     // R
        newData[i + 1] = 255; // G
        newData[i + 2] = 255; // B
        newData[i + 3] = 255; // Alpha
      } else {
        // Không vẽ -> Giữ nguyên màu ĐEN (0, 0, 0, 255) = Vùng giữ nguyên
        newData[i] = 0;
        newData[i + 1] = 0;
        newData[i + 2] = 0;
        newData[i + 3] = 255; // Alpha nền luôn phải là 255
      }
    }

    // 5. Đưa dữ liệu đã chuẩn hóa vào canvas tạm
    ctx.putImageData(newImageData, 0, 0);

    // 6. Xuất ra base64 string (PNG format)
    return tempCanvas.toDataURL('image/png');
  }, [imageDimensions]);

  return {
    isMaskingMode,
    isMaskDrawing,
    maskBrushSize,
    maskToolType,
    maskCanvasRef,
    edgeOverlayCanvasRef,
    setMaskBrushSize,
    setMaskToolType,
    handleMaskMouseDown: startDrawing,
    handleMaskMouseMove: draw,
    handleMaskMouseUp: stopDrawing,
    clearMask,
    resetMaskHistory,
    toggleMaskingMode,
    // Mask history with Command Pattern
    maskHistoryIndex: historyState.currentIndex,
    maskHistoryLength: historyState.historyLength,
    undoMask,
    redoMask,
    hasMaskContent: historyState.hasContent || hasCanvasContent, // Use command history OR actual canvas content
    canUndo: historyState.canUndo,
    canRedo: historyState.canRedo,
    // Export final mask (Black/White binary mask for AI)
    getFinalMask,
    // Edge detection and flood fill
    enableEdgeDetection,
    enableFloodFill,
    setEnableEdgeDetection,
    setEnableFloodFill,
    // Smart masking
    enableSmartMasking,
    setEnableSmartMasking,
    smartMaskModelType,
    setSmartMaskModelType,
    isSmartMaskLoading,
    generateSmartMaskFromBox,
    // Cancel smart mask generation
    cancelSmartMask: async () => {
      if (currentSmartMaskRequestId) {
        try {
          await apiService.cancelSmartMask(currentSmartMaskRequestId);
          console.log(`✅ Smart mask request ${currentSmartMaskRequestId} cancelled`);
          setCurrentSmartMaskRequestId(null);
          setIsSmartMaskLoading(false);
        } catch (error) {
          console.warn(`⚠️ Failed to cancel smart mask: ${error}`);
        }
      }
    },
  };
}
