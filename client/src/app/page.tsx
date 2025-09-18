"use client";

import { useState } from "react";
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

  return (
    <div className="min-h-screen max-h-screen bg-[var(--primary-bg)] text-[var(--text-primary)] flex flex-col dots-pattern-small overflow-hidden">
      {/* Header */}
      <Header 
        onSummon={handleEdit}
        isCustomizeOpen={isCustomizeOpen}
        onToggleCustomize={() => setIsCustomizeOpen(!isCustomizeOpen)}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden relative">
        {/* Left Side - Canvas */}
        <Canvas
          uploadedImage={uploadedImage}
          imageDimensions={imageDimensions}
          displayScale={displayScale}
          transform={transform}
          viewportZoom={viewportZoom}
          isDragging={isDragging}
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
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
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
          uploadedImage={uploadedImage}
          isMaskingMode={isMaskingMode}
          maskBrushSize={maskBrushSize}
          onImageUpload={handleImageUpload}
          onRemoveImage={removeImage}
          onToggleMaskingMode={toggleMaskingMode}
          onClearMask={clearMask}
          onMaskBrushSizeChange={setMaskBrushSize}
        />
      </main>
    </div>
  );
}
