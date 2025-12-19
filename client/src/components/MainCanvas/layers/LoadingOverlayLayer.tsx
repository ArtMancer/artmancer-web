/**
 * LoadingOverlayLayer Component
 * Hiển thị loading indicator khi FastSAM đang xử lý
 */

import { getAbsoluteLayerStyle } from '../utils';
import { Z_INDEX } from '../constants';

interface LoadingOverlayLayerProps {
  isLoading: boolean;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
}

export default function LoadingOverlayLayer({
  isLoading,
  imageDimensions,
  displayScale,
}: LoadingOverlayLayerProps) {
  if (!isLoading || !imageDimensions) {
    return null;
  }

  return (
    <div
      className="absolute flex items-center justify-center bg-black/40 backdrop-blur-sm"
      style={{
        ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.LOADING),
        pointerEvents: 'auto',
      }}
    >
      <div className="flex flex-col items-center gap-3">
        {/* Spinner */}
        <div className="spinner-smooth w-12 h-12 border-4 border-white/30 border-t-white rounded-full" />
        {/* Text */}
        <span className="text-white text-sm font-medium bg-black/50 px-4 py-2 rounded">
          Đang tạo smart mask...
        </span>
      </div>
    </div>
  );
}

