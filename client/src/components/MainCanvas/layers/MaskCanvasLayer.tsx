/**
 * MaskCanvasLayer Component
 * Canvas overlay cho masking
 */

import { getAbsoluteLayerStyle } from "../utils";
import { Z_INDEX } from "../constants";

interface MaskCanvasLayerProps {
  isMaskingMode: boolean;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  hasMaskContent?: boolean;
}

export default function MaskCanvasLayer({
  isMaskingMode,
  maskCanvasRef,
  imageDimensions,
  displayScale,
  hasMaskContent = false,
}: MaskCanvasLayerProps) {
  // Show mask canvas if in masking mode OR if there's mask content (e.g., after returning to original)
  if (!isMaskingMode && !hasMaskContent) {
    return null;
  }

  const style: React.CSSProperties = {
    ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.MASK),
    pointerEvents: "auto",
    // Ensure mask always displays with opacity (even if individual strokes have opacity)
    // This ensures consistent visual appearance
    opacity: 1.0, // Individual strokes already have rgba(255, 0, 0, 0.5) opacity
    // Transform is already applied by TransformLayer parent, so we don't need it here
  };

  return <canvas ref={maskCanvasRef} style={style} />;
}
