/**
 * ContentLayer Component
 * Wrapper cho tất cả content, relative positioning
 */

import { getFixedSizeStyle } from '../utils';

interface ContentLayerProps {
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  children: React.ReactNode;
}

export default function ContentLayer({
  imageDimensions,
  displayScale,
  children,
}: ContentLayerProps) {
  const style: React.CSSProperties = {
    ...getFixedSizeStyle(imageDimensions, displayScale),
    position: 'relative',
    display: 'block',
    margin: 0,
    padding: 0,
    lineHeight: 0,
  };

  return <div style={style}>{children}</div>;
}





