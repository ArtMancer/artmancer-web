import { getAbsoluteLayerStyle } from '../utils';
import { Z_INDEX } from '../constants';

interface EdgeOverlayLayerProps {
  isMaskingMode: boolean;
  enableEdgeDetection: boolean;
  edgeOverlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
}

export default function EdgeOverlayLayer({
  isMaskingMode,
  enableEdgeDetection,
  edgeOverlayCanvasRef,
  imageDimensions,
  displayScale,
}: EdgeOverlayLayerProps) {
  if (!isMaskingMode || !enableEdgeDetection) {
    return null;
  }

  const style: React.CSSProperties = {
    ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.MASK + 1),
    pointerEvents: 'none',
    mixBlendMode: 'screen', // Makes yellow edges more visible
  };

  return <canvas ref={edgeOverlayCanvasRef} style={style} />;
}



