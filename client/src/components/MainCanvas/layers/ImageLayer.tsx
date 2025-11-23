/**
 * ImageLayer Component
 * Render image (single hoặc comparison mode)
 */

import { getAbsoluteLayerStyle } from '../utils';
import { Z_INDEX } from '../constants';

interface ImageLayerProps {
  mode: 'single' | 'comparison';
  uploadedImage: string | null;
  originalImage: string | null;
  modifiedImage: string | null;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  comparisonSlider: number;
  isDraggingSeparator: boolean;
  imageRef: React.RefObject<HTMLImageElement | null>;
  onSeparatorMouseDown: (e: React.MouseEvent) => void;
  onSeparatorTouchStart: (e: React.TouchEvent) => void;
  onSeparatorTouchMove: (e: React.TouchEvent) => void;
  onSeparatorTouchEnd: (e: React.TouchEvent) => void;
}

export default function ImageLayer({
  mode,
  uploadedImage,
  originalImage,
  modifiedImage,
  imageDimensions,
  displayScale,
  comparisonSlider,
  isDraggingSeparator,
  imageRef,
  onSeparatorMouseDown,
  onSeparatorTouchStart,
  onSeparatorTouchMove,
  onSeparatorTouchEnd,
}: ImageLayerProps) {
  const imageStyle: React.CSSProperties = {
    width: imageDimensions
      ? `${imageDimensions.width * displayScale}px`
      : '100%',
    height: imageDimensions
      ? `${imageDimensions.height * displayScale}px`
      : '100%',
    objectFit: 'contain',
    display: 'block',
    pointerEvents: 'none',
    userSelect: 'none',
  };

  if (mode === 'comparison') {
    return (
      <div
        style={getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.IMAGE)}
      >
        {/* Lớp dưới: Ảnh AI (Modified) - Full Size */}
        <div
          style={getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.IMAGE)}
        >
          {modifiedImage ? (
            <img
              src={modifiedImage}
              alt="AI Result"
              className="pointer-events-none select-none"
              draggable={false}
              style={imageStyle}
            />
          ) : (
            <div
              className="bg-gray-800 animate-pulse flex items-center justify-center"
              style={getAbsoluteLayerStyle(imageDimensions, displayScale)}
            >
              <span className="text-white/50">Processing...</span>
            </div>
          )}
        </div>

        {/* Lớp trên: Ảnh Gốc (Original) - Bị cắt bởi clip-path */}
        <div
          className="absolute overflow-hidden shadow-xl"
          style={{
            ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.IMAGE + 1),
            clipPath: `inset(0 ${100 - comparisonSlider}% 0 0)`,
            transition: isDraggingSeparator
              ? 'none'
              : 'clip-path 0.1s ease-out',
          }}
        >
          <img
            ref={imageRef}
            src={originalImage || undefined}
            alt="Original"
            className="pointer-events-none select-none"
            draggable={false}
            style={imageStyle}
          />

          {/* Label: ORIGINAL */}
          {comparisonSlider > 15 && (
            <div
              className="absolute top-4 left-4 bg-black/60 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg border border-white/20"
              style={{ zIndex: Z_INDEX.UI }}
            >
              ORIGINAL
            </div>
          )}
        </div>

        {/* Label: AI EDITED */}
        {100 - comparisonSlider > 15 && (
          <div
            className="absolute top-4 right-4 bg-[var(--primary)]/80 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg border border-white/20"
            style={{
              zIndex: Z_INDEX.UI,
            }}
          >
            AI PREVIEW
          </div>
        )}
      </div>
    );
  }

  // Single mode
  return (
    <div
      style={getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.IMAGE)}
    >
      <img
        ref={imageRef}
        src={uploadedImage || undefined}
        alt="Reference"
        className="pointer-events-none select-none"
        draggable={false}
        style={imageStyle}
      />
    </div>
  );
}

