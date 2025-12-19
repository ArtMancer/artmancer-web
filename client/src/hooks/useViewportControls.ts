import { useState, useRef, useCallback, useEffect } from "react";

interface ImageDimensions {
  width: number;
  height: number;
}

interface UseViewportControlsParams {
  imageDimensions: ImageDimensions | null;
  setTransform?: React.Dispatch<React.SetStateAction<{ scale: number }>>;
}

const MIN_ZOOM = 0.1; // 10%
const MAX_ZOOM = 10; // 1000%
const ZOOM_STEP = 0.1; // Button zoom step
const SCROLL_ZOOM_FACTOR = 0.001; // Scroll sensitivity

export function useViewportControls(
  params: UseViewportControlsParams
) {
  const { imageDimensions, setTransform } = params;

  const [viewportZoom, setViewportZoom] = useState(1);
  
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const transformSetterRef = useRef<typeof setTransform>(setTransform);

  useEffect(() => {
    transformSetterRef.current = setTransform;
  }, [setTransform]);

  const clampZoom = useCallback((zoom: number) => {
    return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom));
  }, []);

  // Sync viewportZoom to external transform scale (TransformLayer)
  useEffect(() => {
    if (transformSetterRef.current) {
      transformSetterRef.current(() => ({ scale: viewportZoom }));
    }
  }, [viewportZoom]);

  // Reset zoom when image is cleared
  useEffect(() => {
    if (!imageDimensions) {
      queueMicrotask(() => {
        setViewportZoom(1);
        if (transformSetterRef.current) {
          transformSetterRef.current(() => ({ scale: 1 }));
        }
      });
    }
  }, [imageDimensions]);

  const applyZoom = useCallback(
    (deltaY: number) => {
      setViewportZoom((prev) => {
        const zoomAmount = deltaY * -SCROLL_ZOOM_FACTOR;
        const nextZoom = clampZoom(prev + zoomAmount);
        if (transformSetterRef.current) {
          transformSetterRef.current(() => ({ scale: nextZoom }));
        }
        return nextZoom;
      });
    },
    [clampZoom]
  );

  // React synthetic wheel event listeners có thể bị đánh dấu passive trong một số môi trường,
  // dẫn tới warning "Unable to preventDefault inside passive event listener invocation".
  // Để đảm bảo preventDefault hoạt động hợp lệ, ta gắn native wheel listener với { passive: false }.
  const handleWheel = useCallback((e: React.WheelEvent) => {
    // Giữ handler React làm "no-op" (tránh gọi preventDefault trong listener có thể passive).
    if (!imageDimensions) {
      return;
    }

    if (e.ctrlKey || e.metaKey) {
      return;
    }
    // Không gọi e.preventDefault() ở đây để tránh warning.
  }, [imageDimensions]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !imageDimensions) {
      return;
    }

    const wheelListener = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        return;
      }

      // Native listener với passive:false nên có thể preventDefault hợp lệ
      event.preventDefault();

      const deltaY = event.deltaY;
      applyZoom(deltaY);
    };

    container.addEventListener("wheel", wheelListener, { passive: false });

    return () => {
      container.removeEventListener("wheel", wheelListener);
    };
  }, [imageDimensions, applyZoom]);

  const zoomViewportIn = useCallback(() => {
    if (!imageDimensions) return;
    setViewportZoom((prev) => {
      const next = clampZoom(prev + ZOOM_STEP);
      if (transformSetterRef.current) {
        transformSetterRef.current(() => ({ scale: next }));
      }
      return next;
    });
  }, [clampZoom, imageDimensions]);

  const zoomViewportOut = useCallback(() => {
    if (!imageDimensions) return;
    setViewportZoom((prev) => {
      const next = clampZoom(prev - ZOOM_STEP);
      if (transformSetterRef.current) {
        transformSetterRef.current(() => ({ scale: next }));
      }
      return next;
    });
  }, [clampZoom, imageDimensions]);

  const resetViewportZoom = useCallback(() => {
    if (!imageDimensions) return;
    setViewportZoom(1);
    if (transformSetterRef.current) {
      transformSetterRef.current(() => ({ scale: 1 }));
    }
  }, [imageDimensions]);

  // Dummy mouse handlers preserved for compatibility
  const handleMouseDown = useCallback(() => {}, []);
  const handleMouseMove = useCallback(() => {}, []);
  const handleMouseUp = useCallback(() => {}, []);

  return {
    viewportZoom,
    imageContainerRef,
    containerRef,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    zoomViewportIn,
    zoomViewportOut,
    resetViewportZoom,
  };
}
