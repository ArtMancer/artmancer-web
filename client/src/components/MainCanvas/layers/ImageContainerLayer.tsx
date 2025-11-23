/**
 * ImageContainerLayer Component
 * Container màu tím với kích thước cố định
 */

import { forwardRef } from 'react';
import { getFixedSizeStyle } from '../utils';

interface ImageContainerLayerProps {
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  children: React.ReactNode;
  cursor?: string;
  onMouseDown?: (e: React.MouseEvent) => void;
  onMouseMove?: (e: React.MouseEvent) => void;
  onMouseUp?: (e: React.MouseEvent) => void;
  onMouseLeave?: (e: React.MouseEvent) => void;
  onClick?: (e: React.MouseEvent) => void;
  onKeyDown?: (e: React.KeyboardEvent) => void;
  tabIndex?: number;
}

const ImageContainerLayer = forwardRef<
  HTMLDivElement,
  ImageContainerLayerProps
>(
  (
    {
      imageDimensions,
      displayScale,
      children,
      cursor = 'default',
      onMouseDown,
      onMouseMove,
      onMouseUp,
      onMouseLeave,
      onClick,
      onKeyDown,
      tabIndex,
    },
    ref
  ) => {
    const containerStyle: React.CSSProperties = {
      ...getFixedSizeStyle(imageDimensions, displayScale),
      userSelect: 'none',
      transformOrigin: 'center center',
      display: imageDimensions ? 'block' : 'flex',
      alignItems: imageDimensions ? 'normal' : 'center',
      justifyContent: imageDimensions ? 'normal' : 'center',
      margin: 0,
      padding: 0,
      lineHeight: 0,
      overflow: 'hidden',
      cursor,
    };

    return (
      <div
        ref={ref}
        className={`${
          imageDimensions ? '' : 'w-72 h-72 lg:w-96 lg:h-96'
        } bg-[var(--primary-accent)] rounded-lg shadow-2xl transition-colors select-none ring-1 ring-white/10`}
        style={containerStyle}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
        onKeyDown={onKeyDown}
        tabIndex={tabIndex}
      >
        {children}
      </div>
    );
  }
);

ImageContainerLayer.displayName = 'ImageContainerLayer';

export default ImageContainerLayer;
