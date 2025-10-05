"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import HelpBox from "@/components/HelpBox";
import Canvas from "@/components/MainCanvas";
import NotificationComponent from "@/components/Notification";
import type { NotificationType } from "@/components/Notification";
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
  
  // Enhanced notification state
  const [notificationType, setNotificationType] = useState<NotificationType>('success');
  const [notificationMessage, setNotificationMessage] = useState<string>('');
  const [isNotificationVisible, setIsNotificationVisible] = useState(false);
  
  // AI Task state
  const [aiTask, setAiTask] = useState<'white-balance' | 'object-insert' | 'object-removal'>('object-removal');
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  
  // Advanced options state
  const [negativePrompt, setNegativePrompt] = useState<string>('');
  const [guidanceScale, setGuidanceScale] = useState<number>(3.5);
  const [imageWidth, setImageWidth] = useState<number>(1024);
  const [imageHeight, setImageHeight] = useState<number>(1024);
  const [inferenceSteps, setInferenceSteps] = useState<number>(50);
  const [numImages, setNumImages] = useState<number>(1);
  const [cfgScale, setCfgScale] = useState<number>(1.0);
  
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

  const handleCloseNotification = useCallback(() => {
    setIsNotificationVisible(false);
    clearNotificationTimeout();
    setSuccess(null);
    setError(null);
  }, [clearNotificationTimeout]);

  const setNotificationWithTimeout = useCallback((
    type: NotificationType,
    message: string,
    timeoutMs: number = 5000
  ) => {
    clearNotificationTimeout();
    
    // Set new notification
    setNotificationType(type);
    setNotificationMessage(message);
    setIsNotificationVisible(true);
    
    // Legacy state for backward compatibility
    if (type === 'success') {
      setSuccess(message);
      setError(null);
    } else if (type === 'error') {
      setError(message);
      setSuccess(null);
    }
    
    // Auto-hide timer
    notificationTimeoutRef.current = setTimeout(() => {
      setIsNotificationVisible(false);
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
    setIsNotificationVisible(false);
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

  // Reference image handling for AI tasks
  const handleReferenceImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0]
      const reader = new FileReader()
      reader.onload = (e) => {
        if (e.target && typeof e.target.result === 'string') {
          setReferenceImage(e.target.result)
          setError(null)
        }
      }
      reader.readAsDataURL(file)
    }
  }
  
  const handleRemoveReferenceImage = () => {
    setReferenceImage(null)
  }
  
  const handleAiTaskChange = (task: 'white-balance' | 'object-insert' | 'object-removal') => {
    setAiTask(task)
    // Clear reference image when switching away from object-insert
    if (task !== 'object-insert') {
      setReferenceImage(null)
    }
  }

  const {
    isMaskingMode,
    isMaskDrawing,
    maskBrushSize,
    maskCanvasRef,
    setMaskBrushSize,
    toggleMaskingMode,
    clearMask,
    resetMaskHistory,
    handleMaskMouseDown,
    handleMaskMouseMove,
    handleMaskMouseUp,
    maskHistoryIndex,
    maskHistoryLength,
    undoMask,
    redoMask,
    hasMaskContent
  } = useMasking(uploadedImage, imageDimensions, imageContainerRef, transform, viewportZoom);

  const {
    historyIndex,
    historyStack,
    handleUndo,
    handleRedo,
    addToHistory,
    initializeHistory
  } = useImageHistory();

  const [comparisonSlider, setComparisonSlider] = useState(50);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [modifiedImageForComparison, setModifiedImageForComparison] = useState<string | null>(null);

  // Initialize history when a new image is uploaded
  useEffect(() => {
    if (uploadedImage && !originalImage && !modifiedImageForComparison) {
      // This is a fresh upload, initialize history
      initializeHistory(uploadedImage);
    }
  }, [uploadedImage, originalImage, modifiedImageForComparison, initializeHistory]);
  
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

  // Create wrapped image upload handler that initializes history
  const handleImageUploadWrapper = (event: React.ChangeEvent<HTMLInputElement>) => {
    // First clear any existing state including mask history
    setOriginalImage(null);
    setModifiedImageForComparison(null);
    setComparisonSlider(50);
    resetMaskHistory();
    
    // Handle the upload
    handleImageUpload(event);
    
    // Initialize history for the new image - we'll do this in a useEffect
  };

  // Handle image removal
  const handleRemoveImage = () => {
    removeImage();
    setOriginalImage(null);
    setModifiedImageForComparison(null);
    setComparisonSlider(50);
    resetMaskHistory();
  };

  // Handle return to original image
  const handleReturnToOriginal = () => {
    if (originalImage) {
      setUploadedImage(originalImage);
      setModifiedImage(originalImage);
      setModifiedImageForComparison(null);
      setComparisonSlider(50);
      resetMaskHistory();
      clearMask();
      setNotificationWithTimeout('success', 'Returned to original image');
    }
  };

  // Advanced options handlers
  const handleNegativePromptChange = (value: string) => {
    setNegativePrompt(value);
  };

  const handleGuidanceScaleChange = (value: number) => {
    setGuidanceScale(value);
  };

  const handleImageSizeChange = (width: number, height: number) => {
    setImageWidth(width);
    setImageHeight(height);
  };

  const handleInferenceStepsChange = (value: number) => {
    setInferenceSteps(value);
  };

  const handleNumImagesChange = (value: number) => {
    setNumImages(value);
  };

  const handleCfgScaleChange = (value: number) => {
    setCfgScale(value);
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
      
      // Store the original image before editing
      setOriginalImage(uploadedImage);
      
      const result = await generateImage(prompt, uploadedImage);
      
      if (result && result.image_base64) {
        const imageData = `data:image/png;base64,${result.image_base64}`;
        setUploadedImage(imageData);
        setModifiedImage(imageData);
        setModifiedImageForComparison(imageData); // Set the modified image for comparison
        
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

        {/* Refined Notification Component */}
        <NotificationComponent
          type={notificationType}
          message={notificationMessage}
          isVisible={isNotificationVisible}
          onClose={handleCloseNotification}
          duration={5000}
          position="top"
        />

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
          originalImage={originalImage}
          modifiedImage={modifiedImageForComparison}
          comparisonSlider={comparisonSlider}
          historyIndex={historyIndex}
          historyStackLength={historyStack.length}
          isHelpOpen={isHelpOpen}
          imageContainerRef={imageContainerRef}
          containerRef={containerRef}
          maskCanvasRef={maskCanvasRef}
          imageRef={imageRef}
          onImageUpload={handleImageUploadWrapper}
          onRemoveImage={handleRemoveImage}
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
          referenceImage={referenceImage}
          aiTask={aiTask}
          onImageUpload={handleImageUploadWrapper}
          onRemoveImage={handleRemoveImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
          maskHistoryIndex={maskHistoryIndex}
          maskHistoryLength={maskHistoryLength}
          onMaskUndo={undoMask}
          onMaskRedo={redoMask}
          hasMaskContent={hasMaskContent}
          onReferenceImageUpload={handleReferenceImageUpload}
          onRemoveReferenceImage={handleRemoveReferenceImage}
          onAiTaskChange={handleAiTaskChange}
          onResizeStart={handleResizeStart}
          onWidthChange={setSidebarWidth}
          originalImage={originalImage}
          modifiedImage={modifiedImageForComparison}
          onReturnToOriginal={handleReturnToOriginal}
          negativePrompt={negativePrompt}
          guidanceScale={guidanceScale}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          inferenceSteps={inferenceSteps}
          numImages={numImages}
          cfgScale={cfgScale}
          onNegativePromptChange={handleNegativePromptChange}
          onGuidanceScaleChange={handleGuidanceScaleChange}
          onImageSizeChange={handleImageSizeChange}
          onInferenceStepsChange={handleInferenceStepsChange}
          onNumImagesChange={handleNumImagesChange}
          onCfgScaleChange={handleCfgScaleChange}
        />
      </main>
    </div>
  );
}
