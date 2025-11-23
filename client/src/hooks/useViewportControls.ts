import { useState, useRef, useCallback, useEffect } from "react";

interface ImageDimensions {
  width: number;
  height: number;
}

export function useViewportControls(
  imageDimensions?: ImageDimensions | null,
  displayScale?: number
) {
  // Viewport state (zoom only)
  const [viewportZoom, setViewportZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  
  // Refs for performance
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate optimal initial zoom based on image dimensions
  const calculateOptimalZoom = useCallback((dimensions: ImageDimensions, scale: number) => {
    if (!containerRef.current) return 1;

    // Get available viewport space (accounting for padding)
    const containerWidth = containerRef.current.clientWidth - 64; // 2rem padding each side
    const containerHeight = containerRef.current.clientHeight - 64;
    
    // Calculate actual image display size
    const actualImageWidth = dimensions.width * scale;
    const actualImageHeight = dimensions.height * scale;
    
    // Calculate what zoom would make the image fit nicely in the viewport
    const zoomToFitWidth = containerWidth / actualImageWidth;
    const zoomToFitHeight = containerHeight / actualImageHeight;
    const zoomToFit = Math.min(zoomToFitWidth, zoomToFitHeight);
    
    // Smart zoom logic:
    // - If image is much larger than viewport, zoom out to fit (max 0.8 of fit)
    // - If image is smaller than viewport, zoom in slightly (max 1.5)
    // - Keep reasonable bounds
    if (zoomToFit < 0.7) {
      // Large image - zoom out to show more
      return Math.max(zoomToFit * 0.8, 0.3);
    } else if (zoomToFit > 1.5) {
      // Small image - zoom in to make it more visible
      return Math.min(zoomToFit * 0.7, 1.5);
    } else {
      // Good size - minor adjustment
      return Math.max(0.8, Math.min(zoomToFit, 1.2));
    }
  }, []);

  // Auto-adjust zoom when image changes
  useEffect(() => {
    if (imageDimensions && displayScale) {
      // Small delay to ensure container ref is ready
      const timer = setTimeout(() => {
        if (containerRef.current) {
          const optimalZoom = calculateOptimalZoom(imageDimensions, displayScale);
          setViewportZoom(optimalZoom);
        }
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [imageDimensions, displayScale, calculateOptimalZoom]);

  // Zoom functionality - zooms the viewport
  // Use native event listener to allow preventDefault
  const handleWheelNative = useCallback((e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Use 5% increments instead of multiplicative factors
    const zoomIncrement = 0.05; // 5% increment
    setViewportZoom(prevZoom => {
      let newZoom;
      if (e.deltaY > 0) {
        // Zoom out
        newZoom = prevZoom - zoomIncrement;
      } else {
        // Zoom in
        newZoom = prevZoom + zoomIncrement;
      }
      
      // Clamp to bounds and round to nearest 5%
      newZoom = Math.max(0.05, Math.min(3, newZoom));
      newZoom = Math.round(newZoom * 20) / 20; // Round to nearest 0.05 (5%)
      return newZoom;
    });
  }, []);

  // React event handler for compatibility (won't preventDefault but will still work)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    // Don't call preventDefault here - use native listener instead
    // Use 5% increments instead of multiplicative factors
    const zoomIncrement = 0.05; // 5% increment
    setViewportZoom(prevZoom => {
      let newZoom;
      if (e.deltaY > 0) {
        // Zoom out
        newZoom = prevZoom - zoomIncrement;
      } else {
        // Zoom in
        newZoom = prevZoom + zoomIncrement;
      }
      
      // Clamp to bounds and round to nearest 5%
      newZoom = Math.max(0.05, Math.min(3, newZoom));
      newZoom = Math.round(newZoom * 20) / 20; // Round to nearest 0.05 (5%)
      return newZoom;
    });
  }, []);

  // Register native wheel event listener with passive: false
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('wheel', handleWheelNative, { passive: false });

    return () => {
      container.removeEventListener('wheel', handleWheelNative);
    };
  }, [handleWheelNative]);

  // Viewport zoom functions
  const zoomViewportIn = useCallback(() => {
    setViewportZoom(prevZoom => {
      const newZoom = prevZoom + 0.05; // 5% increment
      const clampedZoom = Math.min(newZoom, 3); // Max 3x zoom
      return Math.round(clampedZoom * 20) / 20; // Round to nearest 5%
    });
  }, []);

  const zoomViewportOut = useCallback(() => {
    setViewportZoom(prevZoom => {
      const newZoom = prevZoom - 0.05; // 5% decrement
      const clampedZoom = Math.max(newZoom, 0.05); // Min 5% zoom
      return Math.round(clampedZoom * 20) / 20; // Round to nearest 5%
    });
  }, []);

  const resetViewportZoom = useCallback(() => {
    setViewportZoom(1);
  }, []);

  // Dummy mouse handlers (no functionality, just for compatibility)
  const handleMouseDown = useCallback(() => {}, []);
  const handleMouseMove = useCallback(() => {}, []);
  const handleMouseUp = useCallback(() => {}, []);

  return {
    viewportZoom,
    isDragging,
    imageContainerRef,
    containerRef,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    zoomViewportIn,
    zoomViewportOut,
    resetViewportZoom
  };
}
