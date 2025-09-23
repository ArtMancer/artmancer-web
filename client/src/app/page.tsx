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
  useImageTransform
} from "@/hooks";

export default function Home() {
  // Basic UI state
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(true);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  
  // Resizable panel state with localStorage persistence
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    // Load saved width from localStorage or use default
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('artmancer-sidebar-width');
      return saved ? parseInt(saved, 10) : 320;
    }
    return 320;
  });
  const [isResizing, setIsResizing] = useState(false);
  const resizeRef = useRef<HTMLDivElement>(null);

  // Custom hooks
  const {
    uploadedImage,
    imageDimensions,
    displayScale,
    handleImageUpload,
    removeImage,
    handleImageClick
  } = useImageUpload();

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

  const handleEdit = () => {
    // TODO: Implement image editing logic
    console.log("Editing image...");
  };

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
