/**
 * UIOverlayLayer Component
 * Hints, labels, divider cho comparison mode
 */

import { getAbsoluteLayerStyle } from '../utils';
import { Z_INDEX } from '../constants';

interface UIOverlayLayerProps {
  isComparisonMode: boolean;
  isMaskingMode: boolean;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  comparisonSlider: number;
  isDraggingSeparator: boolean;
  onSeparatorMouseDown: (e: React.MouseEvent) => void;
  onSeparatorTouchStart: (e: React.TouchEvent) => void;
  onSeparatorTouchMove: (e: React.TouchEvent) => void;
  onSeparatorTouchEnd: (e: React.TouchEvent) => void;
}

export default function UIOverlayLayer({
  isComparisonMode,
  isMaskingMode,
  imageDimensions,
  displayScale,
  comparisonSlider,
  isDraggingSeparator,
  onSeparatorMouseDown,
  onSeparatorTouchStart,
  onSeparatorTouchMove,
  onSeparatorTouchEnd,
}: UIOverlayLayerProps) {
  // Divider cho comparison mode
  if (isComparisonMode) {
    return (
      <div
        className="absolute top-0 w-1 bg-white cursor-col-resize hover:bg-blue-400 transition-colors touch-none"
        style={{
          left: `${comparisonSlider}%`,
          top: 0,
          height: imageDimensions
            ? `${imageDimensions.height * displayScale}px`
            : '100%',
          transform: 'translateX(-50%)',
          boxShadow: '0 0 15px rgba(0,0,0,0.7)',
          zIndex: Z_INDEX.DIVIDER,
        }}
        onMouseDown={onSeparatorMouseDown}
        onTouchStart={onSeparatorTouchStart}
        onTouchMove={onSeparatorTouchMove}
        onTouchEnd={onSeparatorTouchEnd}
      >
        {/* Nút tròn ở giữa để dễ cầm nắm */}
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-[0_0_10px_rgba(0,0,0,0.5)] flex items-center justify-center border-4 border-gray-800">
          {/* Icon mũi tên 2 chiều */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-4 h-4 text-gray-800"
          >
            <path d="M14.5 12a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0Z" />
            <path
              fillRule="evenodd"
              d="M7.5 12a5 5 0 1 1 10 0 5 5 0 0 1-10 0Zm-2.06 5.56a8 8 0 1 1 13.12 0 1.5 1.5 0 0 1-2.12-2.12 5 5 0 1 0-8.88 0 1.5 1.5 0 0 1-2.12 2.12Z"
              clipRule="evenodd"
            />
          </svg>
        </div>

        {/* Vùng cảm ứng vô hình rộng hơn để dễ kéo */}
        <div className="absolute inset-y-0 -left-4 -right-4 bg-transparent" />
      </div>
    );
  }

  // Hint overlay cho single mode (khi không ở chế độ mask)
  if (!isMaskingMode && !isComparisonMode) {
    return (
      <div
        className="absolute bg-black/0 hover:bg-black/20 transition-colors flex items-center justify-center pointer-events-none"
        style={{
          ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.UI),
        }}
      >
        <span className="text-white text-sm font-medium bg-black/50 px-3 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
          Click to change image
        </span>
      </div>
    );
  }

  return null;
}

