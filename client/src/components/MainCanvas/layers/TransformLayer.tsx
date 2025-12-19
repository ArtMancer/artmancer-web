/**
 * TransformLayer Component
 * Pan/zoom transform container
 */

import { getFixedSizeStyle } from '../utils';

interface TransformLayerProps {
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  children: React.ReactNode;
}

export default function TransformLayer({
  imageDimensions,
  displayScale,
  children,
}: TransformLayerProps) {
  const style: React.CSSProperties = {
    ...getFixedSizeStyle(imageDimensions, displayScale),
    position: 'absolute',
    top: 0,
    left: 0,
    // Zoom is now handled by ViewportLayer (viewportZoom).
    // Keep this at scale(1) so image and mask canvases stay perfectly in sync.
    transform: 'scale(1)',
    transformOrigin: 'center center',
    willChange: 'transform',
    transition: 'transform 75ms ease-out',
    margin: 0,
    padding: 0,
    // Children (ImageLayer, MaskCanvasLayer) are absolute positioned relative to this container
  };

  return <div style={style}>{children}</div>;
}

