import { useRef, useCallback, useEffect } from "react";
import { MdClose, MdRefresh } from "react-icons/md";
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
  lastDrawPoint: { x: number; y: number } | null;
  
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
  lastDrawPoint,
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
                <img
                  ref={imageRef}
                  src={uploadedImage}
                  alt="Uploaded reference"
                  className="w-full h-full object-contain rounded-lg pointer-events-none"
                  draggable={false}
                />
              </div>

              {/* Control buttons */}
              <div className="absolute top-2 right-2 flex gap-1 z-10">
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onRemoveImage();
                  }}
                  className="bg-red-500/90 hover:bg-red-600 text-white rounded-full w-8 h-8 flex items-center justify-center transition-all duration-200 backdrop-blur-sm"
                  title="Remove image"
                >
                  <MdClose size={14} />
                </button>
              </div>
              
              {/* Mask overlay */}
              {isMaskingMode && (
                <canvas
                  ref={maskCanvasRef}
                  className="absolute inset-0"
                  style={{ 
                    pointerEvents: "none",
                    width: "100%",
                    height: "100%"
                  }}
                />
              )}

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
