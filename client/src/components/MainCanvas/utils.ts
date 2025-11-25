/**
 * Canvas Utility Functions
 * Helper functions cho size calculations và layer styling
 */

interface ImageDimensions {
  width: number;
  height: number;
}

interface DisplaySize {
  width: number;
  height: number;
}

/**
 * Tính toán kích thước hiển thị từ image dimensions và display scale
 */
export function calculateDisplaySize(
  imageDimensions: ImageDimensions | null,
  displayScale: number
): DisplaySize | null {
  if (!imageDimensions) {
    return null;
  }

  return {
    width: imageDimensions.width * displayScale,
    height: imageDimensions.height * displayScale,
  };
}

/**
 * Tạo style object cho layer với kích thước cố định
 */
export function getFixedSizeStyle(
  imageDimensions: ImageDimensions | null,
  displayScale: number
): React.CSSProperties {
  const displaySize = calculateDisplaySize(imageDimensions, displayScale);

  if (!displaySize) {
    return {};
  }

  return {
    width: `${displaySize.width}px`,
    height: `${displaySize.height}px`,
    minWidth: `${displaySize.width}px`,
    minHeight: `${displaySize.height}px`,
    maxWidth: `${displaySize.width}px`,
    maxHeight: `${displaySize.height}px`,
    aspectRatio: `${imageDimensions!.width} / ${imageDimensions!.height}`,
    boxSizing: 'border-box',
  };
}

/**
 * Tạo style object cho absolute positioned layer
 */
export function getAbsoluteLayerStyle(
  imageDimensions: ImageDimensions | null,
  displayScale: number,
  zIndex?: number
): React.CSSProperties {
  const baseStyle = getFixedSizeStyle(imageDimensions, displayScale);

  return {
    ...baseStyle,
    position: 'absolute',
    top: 0,
    left: 0,
    margin: 0,
    padding: 0,
    ...(zIndex !== undefined && { zIndex }),
  };
}





