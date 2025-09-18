import { useState, useRef, useCallback, useEffect } from "react";

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
  const [maskBrushSize, setMaskBrushSize] = useState(20);
  const [lastDrawPoint, setLastDrawPoint] = useState<{x: number, y: number} | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);

  // Drawing function
  const drawOnCanvas = useCallback((x: number, y: number) => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !imageDimensions) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Calculate brush size relative to the base image size
    // Since the canvas is scaled by CSS transform, we use the base size
    const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
    const brushSize = (maskBrushSize / 100) * (baseImageSize / 10);
    
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = Math.max(1, brushSize / 10);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    if (lastDrawPoint) {
      // Draw a line from the last point to current point for smooth lines
      ctx.beginPath();
      ctx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
      ctx.lineTo(x, y);
      ctx.stroke();
      
      // Fill circles along the line for solid coverage
      const distance = Math.sqrt(Math.pow(x - lastDrawPoint.x, 2) + Math.pow(y - lastDrawPoint.y, 2));
      const steps = Math.max(1, Math.floor(distance / 2));
      
      for (let i = 0; i <= steps; i++) {
        const ratio = i / steps;
        const interpolatedX = lastDrawPoint.x + (x - lastDrawPoint.x) * ratio;
        const interpolatedY = lastDrawPoint.y + (y - lastDrawPoint.y) * ratio;
        
        ctx.beginPath();
        ctx.arc(interpolatedX, interpolatedY, brushSize / 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    } else {
      // First point - just draw a circle
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    setLastDrawPoint({ x, y });
  }, [lastDrawPoint, maskBrushSize, imageDimensions]);

  // Handle canvas resizing when image dimensions change
  useEffect(() => {
    if (!maskCanvasRef.current || !imageContainerRef.current || !imageDimensions) return;
    
    const canvas = maskCanvasRef.current;
    const container = imageContainerRef.current;
    
    const resizeCanvas = () => {
      const containerRect = container.getBoundingClientRect();
      
      // Calculate the actual image display size within the container
      // Since the image uses object-contain, we need to calculate the contained size
      const containerAspect = containerRect.width / containerRect.height;
      const imageAspect = imageDimensions.width / imageDimensions.height;
      
      let actualImageWidth, actualImageHeight;
      
      if (imageAspect > containerAspect) {
        // Image is wider - width fills container, height is scaled
        actualImageWidth = containerRect.width;
        actualImageHeight = containerRect.width / imageAspect;
      } else {
        // Image is taller - height fills container, width is scaled
        actualImageHeight = containerRect.height;
        actualImageWidth = containerRect.height * imageAspect;
      }
      
      // Calculate image offset within container (for centering)
      const imageOffsetX = (containerRect.width - actualImageWidth) / 2;
      const imageOffsetY = (containerRect.height - actualImageHeight) / 2;
      
      // Set canvas internal resolution to match the base image display size (not scaled by zoom)
      // The canvas will be scaled by CSS transform just like the image
      canvas.width = actualImageWidth;
      canvas.height = actualImageHeight;
      
      // Position and style the canvas to overlay the image exactly
      canvas.style.position = 'absolute';
      canvas.style.left = `${imageOffsetX}px`;
      canvas.style.top = `${imageOffsetY}px`;
      canvas.style.width = `${actualImageWidth}px`;
      canvas.style.height = `${actualImageHeight}px`;
      
      // Apply the same transform as the image (scale by viewportZoom)
      canvas.style.transform = `scale(${viewportZoom})`;
      canvas.style.transformOrigin = 'center center';
      
      // Store the image bounds for coordinate conversion
      (canvas as any)._imageOffsetX = imageOffsetX;
      (canvas as any)._imageOffsetY = imageOffsetY;
      (canvas as any)._imageWidth = actualImageWidth;
      (canvas as any)._imageHeight = actualImageHeight;
      
      // Clear the canvas when resizing
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    };
    
    // Initial resize
    const timer = setTimeout(resizeCanvas, 100); // Small delay to ensure image is loaded
    
    // Listen for window resize
    window.addEventListener('resize', resizeCanvas);
    
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [uploadedImage, transform.scale, imageDimensions, imageContainerRef, viewportZoom]);

  // Mouse handlers for masking
  const handleMaskMouseDown = useCallback((e: React.MouseEvent) => {
    if (!isMaskingMode || !uploadedImage || !imageDimensions) return;
    
    e.preventDefault();
    e.stopPropagation();
    setIsMaskDrawing(true);
    
    const canvas = maskCanvasRef.current;
    const container = imageContainerRef.current;
    if (!canvas || !container) return;
    
    const canvasRect = canvas.getBoundingClientRect();
    
    // Get coordinates relative to the canvas (after CSS transform scaling)
    const screenX = e.clientX - canvasRect.left;
    const screenY = e.clientY - canvasRect.top;
    
    // Convert screen coordinates to canvas internal coordinates
    // Since canvas is scaled by CSS transform, we need to account for that
    const canvasX = (screenX / canvasRect.width) * canvas.width;
    const canvasY = (screenY / canvasRect.height) * canvas.height;
    
    // Check if the click is within the canvas bounds
    if (canvasX >= 0 && canvasX <= canvas.width && canvasY >= 0 && canvasY <= canvas.height) {
      setLastDrawPoint(null); // Reset for new stroke
      drawOnCanvas(canvasX, canvasY);
    }
  }, [isMaskingMode, uploadedImage, imageDimensions, drawOnCanvas, imageContainerRef]);

  const handleMaskMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isMaskDrawing || !isMaskingMode || !uploadedImage || !imageDimensions) return;
    
    e.preventDefault();
    const canvas = maskCanvasRef.current;
    const container = imageContainerRef.current;
    if (!canvas || !container) return;
    
    const canvasRect = canvas.getBoundingClientRect();
    
    // Get coordinates relative to the canvas (after CSS transform scaling)
    const screenX = e.clientX - canvasRect.left;
    const screenY = e.clientY - canvasRect.top;
    
    // Convert screen coordinates to canvas internal coordinates
    // Since canvas is scaled by CSS transform, we need to account for that
    const canvasX = (screenX / canvasRect.width) * canvas.width;
    const canvasY = (screenY / canvasRect.height) * canvas.height;
    
    // Check if the movement is within the canvas bounds
    if (canvasX >= 0 && canvasX <= canvas.width && canvasY >= 0 && canvasY <= canvas.height) {
      drawOnCanvas(canvasX, canvasY);
    }
  }, [isMaskDrawing, isMaskingMode, uploadedImage, imageDimensions, drawOnCanvas]);

  const handleMaskMouseUp = useCallback(() => {
    setIsMaskDrawing(false);
    setLastDrawPoint(null);
  }, []);

  const clearMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, []);

  const toggleMaskingMode = useCallback(() => {
    setIsMaskingMode(!isMaskingMode);
    if (isMaskingMode) {
      // Exiting masking mode
      setIsMaskDrawing(false);
      setLastDrawPoint(null);
    } else {
      // Entering masking mode - canvas will be initialized by the useEffect
      setLastDrawPoint(null);
    }
  }, [isMaskingMode]);

  return {
    isMaskingMode,
    isMaskDrawing,
    maskBrushSize,
    lastDrawPoint,
    maskCanvasRef,
    setMaskBrushSize,
    handleMaskMouseDown,
    handleMaskMouseMove,
    handleMaskMouseUp,
    clearMask,
    toggleMaskingMode
  };
}
