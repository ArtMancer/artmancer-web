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
  maskVisible?: boolean;
  hasMaskContent?: boolean;
}

export default function MaskCanvasLayer({
  isMaskingMode,
  maskCanvasRef,
  imageDimensions,
  displayScale,
  maskVisible = true,
  hasMaskContent = false,
}: MaskCanvasLayerProps) {
  // Show mask canvas if in masking mode OR if there's mask content (e.g., after returning to original)
  if (!isMaskingMode && !hasMaskContent) {
    return null;
  }

  const style: React.CSSProperties = {
    ...getAbsoluteLayerStyle(imageDimensions, displayScale, Z_INDEX.MASK),
    pointerEvents: maskVisible ? "auto" : "none",
    opacity: maskVisible ? 1.0 : 0,
    // Transform is already applied by TransformLayer parent, so we don't need it here
  };

  return <canvas ref={maskCanvasRef} className="transition-opacity duration-200" style={style} />;
}
