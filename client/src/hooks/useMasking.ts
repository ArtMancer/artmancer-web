import { useState, useRef, useCallback, useEffect } from "react";

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
    const ctx = this.canvas.getContext('2d');
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
    console.log('🔍 CanvasDrawCommand.hasContent():', {
      currentStateHasContent: this.currentState.hasContent,
      previousStateHasContent: this.previousState.hasContent
    });
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
    const ctx = this.canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  undo(): void {
    const ctx = this.canvas.getContext('2d');
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
      const hasContent = command.hasContent();
      console.log('🔍 CommandHistory.hasContent() check:', {
        currentIndex: this.currentIndex,
        commandsLength: this.commands.length,
        commandType: command.constructor.name,
        commandHasContent: hasContent
      });
      return hasContent;
    }
    console.log('🔍 CommandHistory.hasContent() - no valid command at current index');
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
  viewportZoom: number
) {
  // Masking state
  const [isMaskingMode, setIsMaskingMode] = useState(false);
  const [isMaskDrawing, setIsMaskDrawing] = useState(false);
  const [maskBrushSize, setMaskBrushSize] = useState(5);
  
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
    
    console.log('🔄 Updating history state:', newState);
    setHistoryState(newState);
  }, []);

  // Reset mask history when uploaded image changes
  useEffect(() => {
    commandHistory.current.clear();
    updateHistoryState();
    setIsMaskingMode(false);
    setIsMaskDrawing(false);
  }, [uploadedImage, updateHistoryState]);

  // Clear command history when exiting masking mode
  useEffect(() => {
    // Only clear history when transitioning from masking mode to non-masking mode
    // This ensures we don't clear history when entering masking mode
    if (!isMaskingMode) {
      console.log('Exiting masking mode - clearing command history');
      commandHistory.current.clear();
      updateHistoryState();
      
      // Also clear the canvas when exiting masking mode
      const canvas = maskCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          console.log('Canvas cleared on masking mode exit');
        }
      }
    } else {
      console.log('Entering masking mode - starting with fresh history');
    }
  }, [isMaskingMode, updateHistoryState]);

  // Initialize canvas context
  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      console.log('Canvas not found during initialization');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('Could not get canvas context');
      return;
    }

    console.log('Initializing canvas context');
    // Set brush to maximum hardness with sharp edges
    ctx.lineCap = 'butt';  // Sharp line ends instead of round
    ctx.lineJoin = 'miter';  // Sharp corners instead of round
    ctx.globalCompositeOperation = 'source-over';
    
    // Disable anti-aliasing for maximum hardness (crisp edges)
    ctx.imageSmoothingEnabled = false;
    
    // Set initial drawing properties
    ctx.lineWidth = 5; // Default line width
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    
    ctxRef.current = ctx;
    console.log('Canvas context initialized with:', {
      lineWidth: ctx.lineWidth,
      strokeStyle: ctx.strokeStyle,
      fillStyle: ctx.fillStyle
    });
  }, []);

  // Update canvas drawing properties when brush settings change
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !imageDimensions) return;

    // Calculate brush size relative to the base image size
    const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
    // Don't scale brush size by viewport zoom - keep it consistent relative to image
    const brushSize = (maskBrushSize / 100) * (baseImageSize / 10);
    const opacity = 0.5; // Fixed 50% opacity

    console.log('Updating canvas properties:', {
      maskBrushSize,
      brushSize,
      opacity,
      baseImageSize
    });

    ctx.lineWidth = brushSize;
    ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
    ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;
  }, [maskBrushSize, imageDimensions]);

  // Helper function to capture current canvas state
  const captureCanvasState = useCallback((): CanvasState => {
    const canvas = maskCanvasRef.current;
    if (!canvas) {
      console.warn('No canvas available for state capture');
      return { imageData: null, hasContent: false };
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn('No canvas context available for state capture');
      return { imageData: null, hasContent: false };
    }

    try {
      console.log('Capturing canvas state:', {
        canvasSize: `${canvas.width}x${canvas.height}`,
        contextExists: !!ctx
      });
      
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      let hasContent = false;
      let pixelsChecked = 0;
      let nonZeroPixels = 0;
      let rgbPixels = 0;
      let alphaPixels = 0;
      
      // Check if canvas has any visible content (RGB or alpha > 0)
      for (let i = 0; i < data.length; i += 4) {
        pixelsChecked++;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        
        if (r > 0 || g > 0 || b > 0) {
          rgbPixels++;
          hasContent = true;
        }
        if (a > 0) {
          alphaPixels++;
          hasContent = true;
        }
        
        if (hasContent && pixelsChecked > 1000) break; // Early exit after finding content
      }
      
      console.log('🔍 Canvas content detection:', {
        totalPixels: data.length / 4,
        pixelsChecked,
        rgbPixels,
        alphaPixels,
        hasContent,
        strokeStyle: ctx.strokeStyle,
        fillStyle: ctx.fillStyle,
        samplePixel: `rgba(${data[0]}, ${data[1]}, ${data[2]}, ${data[3]})`
      });
      
      return { imageData, hasContent };
    } catch (error) {
      console.error('Failed to capture canvas state:', error);
      return { imageData: null, hasContent: false };
    }
  }, []);

  const getCanvasCoordinates = useCallback((e: React.MouseEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    // Get the canvas bounding rect (includes CSS transform scaling)
    const canvasRect = canvas.getBoundingClientRect();
    
    // Calculate the mouse position relative to the canvas
    // The canvasRect already accounts for the CSS transform scale
    const relativeX = (e.clientX - canvasRect.left) / canvasRect.width;
    const relativeY = (e.clientY - canvasRect.top) / canvasRect.height;
    
    // Convert to canvas internal coordinates
    const x = relativeX * canvas.width;
    const y = relativeY * canvas.height;

    return { x, y };
  }, []);

  // Store initial state when starting to draw (for command creation)
  const initialDrawState = useRef<CanvasState | null>(null);

  const startDrawing = useCallback((e: React.MouseEvent) => {
    if (!isMaskingMode || !uploadedImage || !imageDimensions) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    // Capture initial state before drawing for command creation
    initialDrawState.current = captureCanvasState();
    console.log('🎨 Starting stroke - Initial state captured:', {
      hasContent: initialDrawState.current.hasContent,
      timestamp: new Date().toLocaleTimeString()
    });
    
    const { x, y } = getCanvasCoordinates(e);
    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsMaskDrawing(true);
  }, [isMaskingMode, uploadedImage, imageDimensions, getCanvasCoordinates, captureCanvasState]);

  const draw = useCallback((e: React.MouseEvent) => {
    if (!isMaskDrawing || !isMaskingMode) return;
    
    e.preventDefault();
    
    const { x, y } = getCanvasCoordinates(e);
    const ctx = ctxRef.current;
    if (!ctx) {
      console.log('No context available for drawing');
      return;
    }

    console.log('Drawing to:', { x, y }, 'with lineWidth:', ctx.lineWidth);
    ctx.lineTo(x, y);
    ctx.stroke();
  }, [isMaskDrawing, isMaskingMode, getCanvasCoordinates]);

  const stopDrawing = useCallback(() => {
    console.log('Stopping drawing operation');
    
    const ctx = ctxRef.current;
    const canvas = maskCanvasRef.current;
    
    if (ctx) {
      ctx.closePath();
    }
    
    // Create and execute draw command if we have initial state and canvas
    if (canvas && initialDrawState.current) {
      // Capture current state after drawing
      const currentState = captureCanvasState();
      console.log('💾 Saving stroke as command:', {
        previousState: { hasContent: initialDrawState.current.hasContent },
        currentState: { hasContent: currentState.hasContent },
        timestamp: new Date().toLocaleTimeString()
      });
      
      const drawCommand = new CanvasDrawCommand(canvas, initialDrawState.current, currentState);
      commandHistory.current.executeCommand(drawCommand);
      updateHistoryState();
      
      console.log('✅ Stroke saved successfully:', commandHistory.current.getHistoryInfo());
      initialDrawState.current = null;
    }
    
    setIsMaskDrawing(false);
  }, [updateHistoryState]);

  // Handle canvas resizing when image dimensions change  
  useEffect(() => {
    if (!maskCanvasRef.current || !imageContainerRef.current || !imageDimensions) return;
    
    const canvas = maskCanvasRef.current;
    const container = imageContainerRef.current;
    
    const resizeCanvas = () => {
      const containerRect = container.getBoundingClientRect();
      
      // Set canvas internal resolution to match the container size
      // This makes the canvas fill the entire div
      const newWidth = containerRect.width;
      const newHeight = containerRect.height;
      
      // Check if canvas dimensions are changing
      const dimensionsChanged = canvas.width !== newWidth || canvas.height !== newHeight;
      
      // Store canvas content before resize if dimensions are changing
      let imageData = null;
      if (dimensionsChanged && canvas.width > 0 && canvas.height > 0) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          try {
            imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          } catch (e) {
            console.warn('Could not save canvas content:', e);
          }
        }
      }
      
      // Set canvas internal resolution to match the full container size
      canvas.width = newWidth;
      canvas.height = newHeight;
      
      // Canvas positioning is handled by CSS (absolute inset-0 inside transform div)
      // Canvas now fills the entire parent div automatically
      
      // Always re-initialize context settings after canvas resize
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Re-initialize context settings for maximum brush hardness
        ctx.lineCap = 'butt';  // Sharp line ends
        ctx.lineJoin = 'miter';  // Sharp corners
        ctx.globalCompositeOperation = 'source-over';
        
        // Disable anti-aliasing for maximum hardness (crisp edges)
        ctx.imageSmoothingEnabled = false;
        
        // Apply current drawing properties
        if (imageDimensions) {
          const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
          const brushSize = (maskBrushSize / 100) * (baseImageSize / 10);
          const opacity = 0.5; // Fixed 50% opacity

          ctx.lineWidth = brushSize;
          ctx.strokeStyle = `rgba(255, 0, 0, ${opacity})`;
          ctx.fillStyle = `rgba(255, 0, 0, ${opacity})`;
        }
        
        // Restore canvas content if it was saved and dimensions changed
        if (imageData && dimensionsChanged) {
          try {
            ctx.putImageData(imageData, 0, 0);
          } catch (e) {
            console.warn('Could not restore canvas content:', e);
          }
        }
        
        ctxRef.current = ctx;
      }
    };
    
    // Initial resize
    const timer = setTimeout(resizeCanvas, 100);
    
    // Listen for window resize
    window.addEventListener('resize', resizeCanvas);
    
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [uploadedImage, transform.scale, imageDimensions, imageContainerRef, isMaskingMode]);

  // Canvas scaling is handled by the container, no separate zoom effect needed
  // useEffect(() => {
  //   const canvas = maskCanvasRef.current;
  //   if (!canvas) return;
  //   
  //   canvas.style.transform = `scale(${viewportZoom})`;
  //   canvas.style.transformOrigin = 'center center';
  // }, [viewportZoom]);

  const undoMask = useCallback(() => {
    console.log('🔙 Undo mask called, current state:', commandHistory.current.getHistoryInfo());
    if (commandHistory.current.undo()) {
      updateHistoryState();
      console.log('✅ Undo completed, new state:', commandHistory.current.getHistoryInfo());
    } else {
      console.log('❌ Undo failed - no commands to undo');
    }
  }, [updateHistoryState]);

  const redoMask = useCallback(() => {
    console.log('🔜 Redo mask called, current state:', commandHistory.current.getHistoryInfo());
    if (commandHistory.current.redo()) {
      updateHistoryState();
      console.log('✅ Redo completed, new state:', commandHistory.current.getHistoryInfo());
    } else {
      console.log('❌ Redo failed - no commands to redo');
    }
  }, [updateHistoryState]);

  const clearMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    
    // Capture current state before clearing
    const currentState = captureCanvasState();
    
    // Optimization: Don't create clear command if canvas is already empty
    if (!currentState.hasContent) {
      console.log('🚫 Clear skipped - canvas already empty');
      return;
    }
    
    console.log('🧹 Saving clear operation as command:', {
      previousState: { hasContent: currentState.hasContent },
      timestamp: new Date().toLocaleTimeString()
    });
    
    // Create and execute clear command
    const clearCommand = new CanvasClearCommand(canvas, currentState);
    commandHistory.current.executeCommand(clearCommand);
    updateHistoryState();
    
    console.log('✅ Clear command saved:', commandHistory.current.getHistoryInfo());
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
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, [updateHistoryState]);

  const toggleMaskingMode = useCallback(() => {
    const wasInMaskingMode = isMaskingMode;
    
    console.log('Toggling masking mode:', {
      currentMode: isMaskingMode,
      willEnterMode: !isMaskingMode,
      historyLength: commandHistory.current.getHistoryLength()
    });
    
    setIsMaskingMode(!isMaskingMode);
    
    if (wasInMaskingMode) {
      // Exiting masking mode
      console.log('Exiting masking mode - preparing for history clear');
      setIsMaskDrawing(false);
      
      // Clear command history immediately for responsive UI
      commandHistory.current.clear();
      updateHistoryState();
      
      // Clear the canvas content
      const canvas = maskCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          console.log('Canvas and history cleared on mode exit');
        }
      }
    } else {
      // Entering masking mode
      console.log('Entering masking mode - starting fresh');
      // History is already clear, just ensure drawing state is reset
      setIsMaskDrawing(false);
    }
  }, [isMaskingMode, updateHistoryState]);

  return {
    isMaskingMode,
    isMaskDrawing,
    maskBrushSize,
    maskCanvasRef,
    setMaskBrushSize,
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
    hasMaskContent: historyState.hasContent,
    canUndo: historyState.canUndo,
    canRedo: historyState.canRedo
  };
}
