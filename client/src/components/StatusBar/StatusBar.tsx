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
  transform,
}: StatusBarProps) {
  // Only show status bar if we have image dimensions or uploaded image
  if (!imageDimensions && !uploadedImage) {
    return null;
  }

  // Calculate displayed size (actual size on screen)
  const displayedWidth = imageDimensions
    ? Math.round(imageDimensions.width * displayScale * viewportZoom)
    : null;
  const displayedHeight = imageDimensions
    ? Math.round(imageDimensions.height * displayScale * viewportZoom)
    : null;

  return (
    <div className="absolute top-4 left-4 bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg px-3 py-2 z-30 shadow-lg">
      <div className="text-[var(--text-primary)] text-sm font-medium flex items-center gap-3 flex-wrap">
        {/* Image Dimensions - Show original size */}
        {imageDimensions && (
          <span className="font-semibold">
            {imageDimensions.width}×{imageDimensions.height}px
          </span>
        )}

        {/* Displayed Size - Show actual size on screen if different from original */}
        {imageDimensions &&
          displayedWidth &&
          displayedHeight &&
          (displayedWidth !== imageDimensions.width ||
            displayedHeight !== imageDimensions.height) && (
            <>
              <span className="text-[var(--text-secondary)]">•</span>
              <span className="text-[var(--text-secondary)]">
                Display: {displayedWidth}×{displayedHeight}px
              </span>
            </>
          )}

        {/* Separator */}
        {imageDimensions && uploadedImage && (
          <span className="text-[var(--text-secondary)]">•</span>
        )}

        {/* Viewport Zoom */}
        {uploadedImage && viewportZoom !== 1 && (
          <span className="text-[var(--text-secondary)]">
            Zoom {Math.round(viewportZoom * 1000) / 10}%
          </span>
        )}

        {/* Masking Mode Indicator */}
        {isMaskingMode && (
          <>
            <span className="text-[var(--text-secondary)]">•</span>
            <span className="text-[var(--highlight-accent)] font-semibold">
              ✏️ Masking
            </span>
          </>
        )}

        {/* Image Transform Scale (only show if significantly different from viewport) */}
        {(imageDimensions || uploadedImage) &&
          transform.scale !== 1 &&
          Math.abs(transform.scale - viewportZoom) > 0.1 && (
            <>
              <span className="text-[var(--text-secondary)]">•</span>
              <span className="text-[var(--text-secondary)]">
                Scale {Math.round(transform.scale * 1000) / 10}%
              </span>
            </>
          )}
      </div>
    </div>
  );
}
