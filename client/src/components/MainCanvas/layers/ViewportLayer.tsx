/**
 * ViewportLayer Component
 * Chịu trách nhiệm viewport zoom transform
 */

interface ViewportLayerProps {
  viewportZoom: number;
  children: React.ReactNode;
}

export default function ViewportLayer({
  viewportZoom,
  children,
}: ViewportLayerProps) {
  return (
    <div
      style={{
        transform: `scale(${viewportZoom})`,
        transformOrigin: 'center center',
      }}
    >
      {children}
    </div>
  );
}



