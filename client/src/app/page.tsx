"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import HelpBox from "@/components/HelpBox";
import Canvas from "@/components/MainCanvas";
import {
  useImageUpload,
  useViewportControls,
  useMasking,
  useImageHistory,
  useImageTransform,
  useImageGeneration
} from "@/hooks";

export default function Home() {
  // Basic UI state
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Notification timeout refs
  const notificationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Resizable panel state - start with default, then load from localStorage
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Load sidebar width from localStorage after hydration
  useEffect(() => {
    const saved = localStorage.getItem('artmancer-sidebar-width');
    if (saved) {
      setSidebarWidth(parseInt(saved, 10));
    }
  }, []);

  // Notification helpers with auto-hide functionality
  const clearNotificationTimeout = useCallback(() => {
    if (notificationTimeoutRef.current) {
      clearTimeout(notificationTimeoutRef.current);
      notificationTimeoutRef.current = null;
    }
  }, []);

  const setNotificationWithTimeout = useCallback((
    type: 'success' | 'error',
    message: string,
    timeoutMs: number = 30000 // 30 seconds default
  ) => {
    clearNotificationTimeout();
    
    if (type === 'success') {
      setSuccess(message);
      setError(null);
    } else {
      setError(message);
      setSuccess(null);
    }
    
    notificationTimeoutRef.current = setTimeout(() => {
      if (type === 'success') {
        setSuccess(null);
      } else {
        setError(null);
      }
    }, timeoutMs);
  }, [clearNotificationTimeout]);

  const clearAllNotifications = useCallback(() => {
    clearNotificationTimeout();
    setSuccess(null);
    setError(null);
  }, [clearNotificationTimeout]);

  // Custom hooks
  const {
    uploadedImage,
    imageDimensions,
    displayScale,
    handleImageUpload,
    removeImage,
    handleImageClick,
    setUploadedImage,
    setModifiedImage
  } = useImageUpload();

  // API integration
  const {
    generateImage,
    isGenerating,
    error: generationError,
    lastGeneration,
    clearError
  } = useImageGeneration();

  const {
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
  } = useViewportControls(imageDimensions, displayScale);

  const {
    transform,
    imageRef
  } = useImageTransform();

  const {
    isMaskingMode,
    isMaskDrawing,
    maskBrushSize,
    lastDrawPoint,
    maskCanvasRef,
    setMaskBrushSize,
    toggleMaskingMode,
    clearMask,
    handleMaskMouseDown,
    handleMaskMouseMove,
    handleMaskMouseUp
  } = useMasking(uploadedImage, imageDimensions, imageContainerRef, transform, viewportZoom);

  const {
    historyIndex,
    historyStack,
    handleUndo,
    handleRedo,
    addToHistory,
    initializeHistory
  } = useImageHistory();

  // Additional state for comparison
  const [comparisonSlider, setComparisonSlider] = useState(50);
  
  // For now, use uploadedImage for both original and modified
  const originalImage = uploadedImage;
  const modifiedImage = uploadedImage;
  
  // Simple download handler
  const handleDownload = () => {
    if (!uploadedImage) return;
    
    const link = document.createElement('a');
    link.href = uploadedImage;
    link.download = 'artmancer-edited-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Handle image generation
  const handleEdit = async (prompt: string) => {
    try {
      clearAllNotifications();
      clearError();
      
      if (!uploadedImage) {
        setNotificationWithTimeout('error', 'Please upload an image first to edit it.');
        return;
      }
      
      const result = await generateImage(prompt, uploadedImage);
      
      if (result && result.image_base64) {
        const imageData = `data:image/png;base64,${result.image_base64}`;
        setUploadedImage(imageData);
        setModifiedImage(imageData);
        
        // Add to history
        addToHistory(imageData);
        
        setNotificationWithTimeout('success', `Image edited successfully! (${result.generation_time.toFixed(1)}s)`);
        
        console.log('Generation successful:', {
          model: result.model_used,
          time: result.generation_time,
          settings: result.settings_used
        });
      }
    } catch (err) {
      console.error('Generation failed:', err);
      setNotificationWithTimeout('error', 'Failed to edit image. Please try again.');
    }
  };

  // Effect to handle generation errors
  useEffect(() => {
    if (generationError) {
      setNotificationWithTimeout('error', generationError);
    }
  }, [generationError, setNotificationWithTimeout]);

  // Check API connectivity on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const { apiService } = await import('@/services/api');
        await apiService.healthCheck();
        console.log('✅ API connection successful');
      } catch (err) {
        console.warn('⚠️ API connection failed:', err);
        // Don't show error immediately - let user try to generate first
      }
    };
    
    checkApiHealth();
  }, []);

  // Cleanup effect to clear notification timeouts
  useEffect(() => {
    return () => {
      clearNotificationTimeout();
    };
  }, [clearNotificationTimeout]);

  // Throttle function for better performance
  const throttle = useCallback((func: Function, delay: number) => {
    let timeoutId: NodeJS.Timeout | null = null;
    let lastExecTime = 0;
    
    return (...args: any[]) => {
      const currentTime = Date.now();
      
      if (currentTime - lastExecTime > delay) {
        func(...args);
        lastExecTime = currentTime;
      } else {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          func(...args);
          lastExecTime = Date.now();
        }, delay - (currentTime - lastExecTime));
      }
    };
  }, []);

  // Resize handlers for the sidebar - optimized for performance
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const updateSidebarWidth = useCallback((e: MouseEvent) => {
    const newWidth = window.innerWidth - e.clientX;
    const minWidth = 280; // Minimum sidebar width
    const maxWidth = Math.min(600, window.innerWidth * 0.5); // Maximum 50% of screen
    
    const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
    setSidebarWidth(clampedWidth);
  }, []);

  // Throttled resize function for smoother performance
  const throttledResize = useMemo(
    () => throttle(updateSidebarWidth, 16), // ~60fps
    [throttle, updateSidebarWidth]
  );

  const handleResizeMove = useCallback((e: MouseEvent) => {
    if (!isResizing) return;
    
    // Use requestAnimationFrame for smooth resizing
    requestAnimationFrame(() => {
      throttledResize(e);
    });
  }, [isResizing, throttledResize]);

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
    // Save the width to localStorage with a slight delay
    setTimeout(() => {
      if (typeof window !== 'undefined') {
        localStorage.setItem('artmancer-sidebar-width', sidebarWidth.toString());
      }
    }, 100);
  }, [sidebarWidth]);

    // Save sidebar width to localStorage when it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('artmancer-sidebar-width', sidebarWidth.toString());
    }
  }, [sidebarWidth]);

  // Mouse event listeners for resizing with passive listeners for better performance
  useEffect(() => {
    if (isResizing) {
      const handleMove = (e: MouseEvent | TouchEvent) => {
        const clientX = 'touches' in e ? e.touches[0]?.clientX : e.clientX;
        if (clientX !== undefined) {
          handleResizeMove({ clientX } as MouseEvent);
        }
      };
      const handleEnd = () => handleResizeEnd();
      
      document.addEventListener('mousemove', handleMove as EventListener, { passive: true });
      document.addEventListener('mouseup', handleEnd);
      document.addEventListener('touchmove', handleMove as EventListener, { passive: true });
      document.addEventListener('touchend', handleEnd);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      // Prevent text selection during resize
      document.body.style.webkitUserSelect = 'none';
      
      return () => {
        document.removeEventListener('mousemove', handleMove as EventListener);
        document.removeEventListener('mouseup', handleEnd);
        document.removeEventListener('touchmove', handleMove as EventListener);
        document.removeEventListener('touchend', handleEnd);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.body.style.webkitUserSelect = '';
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  return (
    <div className="min-h-screen max-h-screen bg-[var(--primary-bg)] text-[var(--text-primary)] flex flex-col dots-pattern-small overflow-hidden">
      {/* Header */}
      <Header 
        onSummon={handleEdit}
        isCustomizeOpen={isCustomizeOpen}
        onToggleCustomize={() => setIsCustomizeOpen(!isCustomizeOpen)}
        isGenerating={isGenerating}
      />

      {/* Main Content - Optimized transitions */}
      <main 
        className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden relative"
        style={{ 
          paddingRight: isCustomizeOpen ? `${sidebarWidth}px` : '0px',
          transition: isResizing ? 'none' : 'padding-right 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          transform: 'translateZ(0)', // Force hardware acceleration
          willChange: isResizing ? 'padding-right' : 'auto'
        }}
      >
        {/* Loading Overlay */}
        {isGenerating && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-40">
            <div className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg p-6 text-center">
              <div className="animate-spin w-8 h-8 border-4 border-[var(--primary-accent)] border-t-transparent rounded-full mx-auto mb-3"></div>
              <p className="text-[var(--text-primary)] font-medium">Generating image...</p>
              <p className="text-[var(--text-secondary)] text-sm mt-1">This may take a few seconds</p>
            </div>
          </div>
        )}

        {/* Success Display */}
        {success && (
          <div className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-green-500/90 text-white px-4 py-2 rounded-lg shadow-lg z-50 max-w-md text-center">
            <div className="flex items-center justify-between">
              <span className="text-sm">{success}</span>
              <button
                onClick={() => {
                  clearNotificationTimeout();
                  setSuccess(null);
                }}
                className="ml-2 text-white/80 hover:text-white"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-red-500/90 text-white px-4 py-2 rounded-lg shadow-lg z-50 max-w-md text-center">
            <div className="flex items-center justify-between">
              <span className="text-sm">{error}</span>
              <button
                onClick={() => {
                  clearNotificationTimeout();
                  setError(null);
                }}
                className="ml-2 text-white/80 hover:text-white"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Left Side - Canvas */}
        <Canvas
          uploadedImage={uploadedImage}
          imageDimensions={imageDimensions}
          displayScale={displayScale}
          transform={transform}
          viewportZoom={viewportZoom}
          isMaskingMode={isMaskingMode}
          isMaskDrawing={isMaskDrawing}
          maskBrushSize={maskBrushSize}
          lastDrawPoint={lastDrawPoint}
          originalImage={originalImage}
          modifiedImage={modifiedImage}
          comparisonSlider={comparisonSlider}
          historyIndex={historyIndex}
          historyStackLength={historyStack.length}
          isHelpOpen={isHelpOpen}
          imageContainerRef={imageContainerRef}
          containerRef={containerRef}
          maskCanvasRef={maskCanvasRef}
          imageRef={imageRef}
          onImageUpload={handleImageUpload}
          onRemoveImage={removeImage}
          onImageClick={handleImageClick}
          onWheel={handleWheel}
          onMaskMouseDown={handleMaskMouseDown}
          onMaskMouseMove={handleMaskMouseMove}
          onMaskMouseUp={handleMaskMouseUp}
          onComparisonSliderChange={setComparisonSlider}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onDownload={handleDownload}
          onZoomViewportIn={zoomViewportIn}
          onZoomViewportOut={zoomViewportOut}
          onResetViewportZoom={resetViewportZoom}
          onToggleHelp={() => setIsHelpOpen(!isHelpOpen)}
        />

        {/* Help Box Component */}
        <HelpBox
          isOpen={isHelpOpen}
          onClose={() => setIsHelpOpen(false)}
        />

        {/* Right Side - Customize Panel */}
        <Sidebar
          isOpen={isCustomizeOpen}
          width={sidebarWidth}
          isResizing={isResizing}
          uploadedImage={uploadedImage}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          onImageUpload={handleImageUpload}
          onRemoveImage={removeImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
          onResizeStart={handleResizeStart}
          onWidthChange={setSidebarWidth}
        />
      </main>
    </div>
  );
}
