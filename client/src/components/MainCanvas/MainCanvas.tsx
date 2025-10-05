import { useRef, useCallback, useEffect, useState } from "react";
import { MdRefresh } from "react-icons/md";
import Toolbox from "../Toolbox";
import StatusBar from "../StatusBar";

interface CanvasProps {
  // Image state
  uploadedImage: string | null;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  transform: { scale: number };
  
  // Viewport state
  viewportZoom: number;
  
  // Masking state
  isMaskingMode: boolean;
  isMaskDrawing: boolean;
  maskBrushSize: number;
  
  // Toolbox state
  originalImage: string | null;
  modifiedImage: string | null;
  comparisonSlider: number;
  historyIndex: number;
  historyStackLength: number;
  isHelpOpen: boolean;
  
  // Refs (passed from parent)
  imageContainerRef: React.RefObject<HTMLDivElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageRef: React.RefObject<HTMLImageElement | null>;
  
  // Event handlers
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onImageClick: (e: React.MouseEvent) => void;
  onWheel: (e: React.WheelEvent) => void;
  
  // Masking handlers
  onMaskMouseDown: (e: React.MouseEvent) => void;
  onMaskMouseMove: (e: React.MouseEvent) => void;
  onMaskMouseUp: () => void;
  
  // Toolbox handlers
  onComparisonSliderChange: (value: number) => void;
  onUndo: () => void;
  onRedo: () => void;
  onDownload: () => void;
  onZoomViewportIn: () => void;
  onZoomViewportOut: () => void;
  onResetViewportZoom: () => void;
  onToggleHelp: () => void;
}

export default function Canvas({
  uploadedImage,
  imageDimensions,
  displayScale,
  transform,
  viewportZoom,
  isMaskingMode,
  isMaskDrawing,
  maskBrushSize,
  originalImage,
  modifiedImage,
  comparisonSlider,
  historyIndex,
  historyStackLength,
  isHelpOpen,
  imageContainerRef,
  containerRef,
  maskCanvasRef,
  imageRef,
  onImageUpload,
  onRemoveImage,
  onImageClick,
  onWheel,
  onMaskMouseDown,
  onMaskMouseMove,
  onMaskMouseUp,
  onComparisonSliderChange,
  onUndo,
  onRedo,
  onDownload,
  onZoomViewportIn,
  onZoomViewportOut,
  onResetViewportZoom,
  onToggleHelp
}: CanvasProps) {
  const [isDraggingSeparator, setIsDraggingSeparator] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const comparisonContainerRef = useRef<HTMLDivElement>(null);

  // Update dimensions when image container changes
  useEffect(() => {
    if (!imageContainerRef.current) return;
    
    const updateDimensions = () => {
      const rect = imageContainerRef.current?.getBoundingClientRect();
      if (rect) {
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    
    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(imageContainerRef.current);
    
    return () => resizeObserver.disconnect();
  }, [imageContainerRef]);

  // Enhanced separator dragging with better pointer handling
  const handleSeparatorMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingSeparator(true);
    
    // Prevent text selection during drag
    document.body.style.userSelect = 'none';
    document.body.style.webkitUserSelect = 'none';
  }, []);

  const handleSeparatorMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingSeparator || !imageContainerRef.current) return;
    
    const container = imageContainerRef.current;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    onComparisonSliderChange(percentage);
  }, [isDraggingSeparator, imageContainerRef, onComparisonSliderChange]);

  const handleSeparatorMouseUp = useCallback(() => {
    setIsDraggingSeparator(false);
    
    // Restore text selection
    document.body.style.userSelect = '';
    document.body.style.webkitUserSelect = '';
  }, []);

  // Touch support for mobile devices
  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    if (!imageContainerRef.current) return;
    
    const container = imageContainerRef.current;
    const rect = container.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    onComparisonSliderChange(percentage);
    setIsDraggingSeparator(true);
  }, [imageContainerRef, onComparisonSliderChange]);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isDraggingSeparator || !imageContainerRef.current) return;
    
    e.preventDefault(); // Prevent scrolling
    
    const container = imageContainerRef.current;
    const rect = container.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    onComparisonSliderChange(percentage);
  }, [isDraggingSeparator, imageContainerRef, onComparisonSliderChange]);

  const handleTouchEnd = useCallback(() => {
    setIsDraggingSeparator(false);
  }, []);

  // Hover support for desktop
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isHovering || isDraggingSeparator || !imageContainerRef.current) return;
    
    const container = imageContainerRef.current;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    onComparisonSliderChange(percentage);
  }, [isHovering, isDraggingSeparator, imageContainerRef, onComparisonSliderChange]);

  // Keyboard support
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!originalImage || !modifiedImage) return;
    
    let increment = 0;
    
    switch (e.key) {
      case 'ArrowLeft':
        increment = -5;
        break;
      case 'ArrowRight':
        increment = 5;
        break;
      default:
        return;
    }
    
    e.preventDefault();
    const newValue = Math.max(0, Math.min(100, comparisonSlider + increment));
    onComparisonSliderChange(newValue);
  }, [originalImage, modifiedImage, comparisonSlider, onComparisonSliderChange]);

  // Add global mouse event listeners for separator dragging
  useEffect(() => {
    if (isDraggingSeparator) {
      document.addEventListener('mousemove', handleSeparatorMouseMove);
      document.addEventListener('mouseup', handleSeparatorMouseUp);
      document.body.style.cursor = 'col-resize';
      
      return () => {
        document.removeEventListener('mousemove', handleSeparatorMouseMove);
        document.removeEventListener('mouseup', handleSeparatorMouseUp);
        document.body.style.cursor = '';
      };
    }
  }, [isDraggingSeparator, handleSeparatorMouseMove, handleSeparatorMouseUp]);

  return (
    <div
      ref={containerRef}
      className="flex-1 relative min-w-0 overflow-hidden flex items-center justify-center"
      onWheel={onWheel}
      style={{ 
        cursor: 'default',
        padding: "2rem" // Move padding to main canvas container
      }}
    >
      {/* Status Bar Component - positioned absolutely */}
      <StatusBar
        imageDimensions={imageDimensions}
        uploadedImage={uploadedImage}
        displayScale={displayScale}
        viewportZoom={viewportZoom}
        isMaskingMode={isMaskingMode}
        transform={transform}
      />
      
      {/* Image Display Area - flexbox centered */}
      <div
        ref={imageContainerRef}
        className={`${
          imageDimensions 
            ? '' 
            : 'w-72 h-72 lg:w-96 lg:h-96'
        } bg-[var(--primary-accent)] rounded-lg flex items-center justify-center shadow-lg overflow-hidden transition-colors select-none`}
        style={{
          userSelect: "none",
          transformOrigin: "center center",
          cursor: uploadedImage ? (isMaskingMode ? 'crosshair' : 'pointer') : 'pointer',
          transform: `scale(${viewportZoom})`,
          ...(imageDimensions && {
            width: `${imageDimensions.width * displayScale}px`,
            height: `${imageDimensions.height * displayScale}px`,
          }),
        }}
        onMouseDown={uploadedImage ? (isMaskingMode ? onMaskMouseDown : undefined) : undefined}
        onMouseMove={uploadedImage ? (isMaskingMode ? onMaskMouseMove : undefined) : undefined}
        onMouseUp={uploadedImage ? (isMaskingMode ? onMaskMouseUp : undefined) : undefined}
        onMouseLeave={uploadedImage ? (isMaskingMode ? onMaskMouseUp : undefined) : undefined}
        onClick={uploadedImage && !isMaskingMode ? onImageClick : undefined}
        onKeyDown={originalImage && modifiedImage ? handleKeyDown : undefined}
        tabIndex={originalImage && modifiedImage ? 0 : undefined}
      >
          {uploadedImage ? (
            <div className="relative w-full h-full overflow-hidden">
              <div
                className="absolute inset-0 transition-transform duration-75 ease-out"
                style={{
                  transform: `scale(${transform.scale})`,
                  transformOrigin: "center center",
                  willChange: "transform",
                }}
              >
                {/* Image display - either single image or comparison view */}
                {originalImage && modifiedImage && originalImage !== modifiedImage ? (
                  /* Comparison view with cropping behavior */
                  <div className="relative w-full h-full">
                    {/* Original Image (left side) - normal color, user's input */}
                    <div className="absolute inset-0">
                      <img
                        ref={imageRef}
                        src={originalImage}
                        alt="Original image"
                        className="w-full h-full object-contain pointer-events-none"
                        draggable={false}
                      />
                      {/* Original image indicator - only show when there's enough space */}
                      {comparisonSlider > 15 && (
                        <div className="absolute top-2 left-2 bg-blue-600 text-white px-2 py-1 rounded text-xs font-medium shadow-lg z-10">
                          ORIGINAL
                        </div>
                      )}
                    </div>

                    {/* Modified Image Container (right side) - grayscale, AI edited */}
                    <div 
                      className="absolute inset-0 overflow-hidden"
                      style={{ 
                        left: `${comparisonSlider}%`,
                        width: `${100 - comparisonSlider}%`
                      }}
                    >
                      {modifiedImage ? (
                        <>
                          <img
                            src={modifiedImage}
                            alt="Modified image"
                            className="w-full h-full object-contain pointer-events-none"
                            style={{ 
                              filter: 'grayscale(100%)',
                              width: `${100 / (100 - comparisonSlider) * 100}%`,
                              marginLeft: `${-comparisonSlider / (100 - comparisonSlider) * 100}%`
                            }}
                            draggable={false}
                          />
                          {/* Modified image indicator - only show when there's enough space */}
                          {(100 - comparisonSlider) > 15 && (
                            <div className="absolute top-2 right-2 bg-gray-500 text-white px-2 py-1 rounded text-xs font-medium shadow-lg z-10">
                              AI EDITED
                            </div>
                          )}
                        </>
                      ) : (
                        /* Placeholder for when modified image is loading or doesn't exist */
                        (100 - comparisonSlider) > 30 && (
                          <div className="absolute inset-0 bg-gray-200 flex items-center justify-center">
                            <div className="text-center text-gray-600">
                              <div className="text-2xl mb-2">🤖</div>
                              <div className="text-sm font-semibold mb-1">AI Processing...</div>
                              <div className="text-xs">Modified image will appear here</div>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                    
                    {/* Separator line with drag functionality */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg z-20 cursor-col-resize"
                      style={{
                        left: `${comparisonSlider}%`,
                        transform: 'translateX(-50%)',
                      }}
                      onMouseDown={handleSeparatorMouseDown}
                    >
                      {/* Separator handle - expanded hit area */}
                      <div 
                        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 bg-white rounded-full shadow-lg flex items-center justify-center cursor-col-resize hover:scale-110 transition-transform"
                        onMouseDown={handleSeparatorMouseDown}
                      >
                        <div className="w-2 h-2 bg-[var(--primary-accent)] rounded-full"></div>
                      </div>
                      
                      {/* Invisible wider hit area for easier dragging */}
                      <div 
                        className="absolute top-0 bottom-0 w-4 left-1/2 transform -translate-x-1/2 cursor-col-resize"
                        onMouseDown={handleSeparatorMouseDown}
                      />
                    </div>
                  </div>
                ) : (
                  /* Single image view */
                  <img
                    ref={imageRef}
                    src={uploadedImage}
                    alt="Uploaded reference"
                    className="w-full h-full object-contain rounded-lg pointer-events-none"
                    draggable={false}
                  />
                )}
                
                {/* Mask overlay - now inside the transform div */}
                {isMaskingMode && (
                  <canvas
                    ref={maskCanvasRef}
                    className="absolute inset-0"
                    style={{ 
                      pointerEvents: "auto",
                      width: "100%",
                      height: "100%",
                      zIndex: 10
                    }}
                  />
                )}
              </div>

              {/* Click hint overlay - doesn't interfere with mouse events */}
              {!isMaskingMode && (
                <div className="absolute inset-0 bg-black/20 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                  <span className="text-white text-sm font-medium bg-black/50 px-3 py-1 rounded">
                    Click to change image
                  </span>
                </div>
              )}
            </div>
          ) : (
            <label
              htmlFor="image-upload"
              className="w-full h-full flex flex-col items-center justify-center text-center text-white/80 cursor-pointer"
            >
              <div className="text-3xl lg:text-4xl mb-2">📷</div>
              <p className="text-xs lg:text-sm px-4 mb-2">
                Click to upload an image
              </p>
              <p className="text-xs px-4 text-white/60 mb-1">
                or your edited image will appear here
              </p>
            </label>
          )}

          {/* Single hidden file input */}
          <input
            type="file"
            accept="image/*"
            onChange={onImageUpload}
            className="hidden"
            id="image-upload"
          />
        </div>

      {/* Toolbox Component */}
      <Toolbox
        uploadedImage={uploadedImage}
        originalImage={originalImage}
        modifiedImage={modifiedImage}
        comparisonSlider={comparisonSlider}
        onComparisonSliderChange={onComparisonSliderChange}
        historyIndex={historyIndex}
        historyStackLength={historyStackLength}
        onUndo={onUndo}
        onRedo={onRedo}
        onDownload={onDownload}
        viewportZoom={viewportZoom}
        onZoomViewportIn={onZoomViewportIn}
        onZoomViewportOut={onZoomViewportOut}
        onResetViewportZoom={onResetViewportZoom}
        isHelpOpen={isHelpOpen}
        onToggleHelp={onToggleHelp}
      />
    </div>
  );
}
