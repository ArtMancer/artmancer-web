interface StatusBarProps {
  imageDimensions: { width: number; height: number } | null;
  uploadedImage: string | null;
  displayScale: number;
  viewportZoom: number;
  isMaskingMode: boolean;
  transform: { scale: number };
}

export default function StatusBar({
  imageDimensions,
  uploadedImage,
  displayScale,
  viewportZoom,
  isMaskingMode,
  transform
}: StatusBarProps) {
  // Only show status bar if we have image dimensions or uploaded image
  if (!imageDimensions && !uploadedImage) {
    return null;
  }

  return (
    <div className="absolute top-4 left-4 bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg px-3 py-2 z-30 shadow-lg">
      <div className="text-[var(--text-primary)] text-sm font-medium flex items-center gap-3">
        {/* Image Dimensions */}
        {imageDimensions && (
          <span>
            {imageDimensions.width}×{imageDimensions.height}px
            {displayScale < 1 && (
              <span className="text-[var(--text-secondary)] ml-1">({Math.round(displayScale * 100)}%)</span>
            )}
          </span>
        )}
        
        {/* Separator */}
        {imageDimensions && uploadedImage && (
          <span className="text-[var(--text-secondary)]">•</span>
        )}
        
        {/* Viewport Zoom */}
        {uploadedImage && (
          <span>
            Zoom {Math.round(viewportZoom * 100)}%
          </span>
        )}
        
        {/* Masking Mode Indicator */}
        {isMaskingMode && (
          <>
            <span className="text-[var(--text-secondary)]">•</span>
            <span className="text-[var(--highlight-accent)] font-semibold">✏️ Masking</span>
          </>
        )}
        
        {/* Image Transform Scale (only show if significantly different from viewport) */}
        {(imageDimensions || uploadedImage) && 
         transform.scale !== 1 && 
         Math.abs(transform.scale - viewportZoom) > 0.1 && (
          <>
            <span className="text-[var(--text-secondary)]">•</span>
            <span className="text-[var(--text-secondary)]">
              Scale {Math.round(transform.scale * 100)}%
            </span>
          </>
        )}
      </div>
    </div>
  );
}
