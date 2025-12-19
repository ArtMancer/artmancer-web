/**
 * ViewportLayer Component
 * Chịu trách nhiệm viewport zoom transform
 */

import { memo } from "react";

interface ViewportLayerProps {
  viewportZoom: number;
  children: React.ReactNode;
}

const ViewportLayer = memo(function ViewportLayer({
  viewportZoom,
  children,
}: ViewportLayerProps) {
  return (
    <div
      style={{
        transform: `scale(${viewportZoom}) translateZ(0)`,
        transformOrigin: 'center center',
        willChange: 'transform',
        transition: 'transform 80ms ease-out',
        backfaceVisibility: 'hidden',
        WebkitBackfaceVisibility: 'hidden',
      }}
    >
      {children}
    </div>
  );
});

export default ViewportLayer;





