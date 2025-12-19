import { useState, useEffect, useCallback, useMemo } from "react";

export interface UseSidebarResizeReturn {
  sidebarWidth: number;
  isResizing: boolean;
  handleResizeStart: (e: React.MouseEvent) => void;
  setSidebarWidth: (width: number) => void;
}

/**
 * Hook to manage sidebar resize functionality
 * Handles mouse/touch events for resizing the sidebar and persists width to localStorage
 */
export function useSidebarResize(
  defaultWidth: number = 320
): UseSidebarResizeReturn {
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    if (typeof window === "undefined") return defaultWidth;
    const saved = localStorage.getItem("artmancer-sidebar-width");
    const parsed = saved ? parseInt(saved, 10) : NaN;
    return Number.isNaN(parsed) ? defaultWidth : parsed;
  });
  const [isResizing, setIsResizing] = useState(false);

  // Throttle function for better performance
  const throttle = useCallback(
    <T extends (...args: never[]) => void>(func: T, delay: number): T => {
      let timeoutId: NodeJS.Timeout | null = null;
      let lastExecTime = 0;

      return ((...args: Parameters<T>) => {
        const currentTime = Date.now();

        if (currentTime - lastExecTime > delay) {
          func(...args);
          lastExecTime = currentTime;
        } else {
          if (timeoutId) clearTimeout(timeoutId);
          timeoutId = setTimeout(() => {
            func(...args);
            lastExecTime = Date.now();
          }, delay - (currentTime - lastExecTime));
        }
      }) as T;
    },
    []
  );

  // Resize handlers for the sidebar - optimized for performance
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const updateSidebarWidth = useCallback((e: MouseEvent) => {
    const newWidth = window.innerWidth - e.clientX;
    const minWidth = 280; // Minimum sidebar width
    const maxWidth = Math.min(600, window.innerWidth * 0.5); // Maximum 50% of screen

    const clampedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
    setSidebarWidth(clampedWidth);
  }, []);

  // Throttled resize function for smoother performance
  const throttledResize = useMemo(
    () => throttle<(e: MouseEvent) => void>(updateSidebarWidth, 16), // ~60fps
    [throttle, updateSidebarWidth]
  );

  const handleResizeMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;

      // Use requestAnimationFrame for smooth resizing
      requestAnimationFrame(() => {
        throttledResize(e);
      });
    },
    [isResizing, throttledResize]
  );

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
    // Save the width to localStorage with a slight delay
    setTimeout(() => {
      if (typeof window !== "undefined") {
        localStorage.setItem(
          "artmancer-sidebar-width",
          sidebarWidth.toString()
        );
      }
    }, 100);
  }, [sidebarWidth]);

  // Save sidebar width to localStorage when it changes
  useEffect(() => {
    if (typeof window !== "undefined" && !isResizing) {
      localStorage.setItem("artmancer-sidebar-width", sidebarWidth.toString());
    }
  }, [sidebarWidth, isResizing]);

  // Mouse event listeners for resizing with passive listeners for better performance
  useEffect(() => {
    if (isResizing) {
      const handleMove = (e: MouseEvent | TouchEvent) => {
        const clientX = "touches" in e ? e.touches[0]?.clientX : e.clientX;
        if (clientX !== undefined) {
          handleResizeMove({ clientX } as MouseEvent);
        }
      };
      const handleEnd = () => handleResizeEnd();

      document.addEventListener("mousemove", handleMove as EventListener, {
        passive: true,
      });
      document.addEventListener("mouseup", handleEnd);
      document.addEventListener("touchmove", handleMove as EventListener, {
        passive: true,
      });
      document.addEventListener("touchend", handleEnd);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      // Prevent text selection during resize
      document.body.style.webkitUserSelect = "none";

      return () => {
        document.removeEventListener("mousemove", handleMove as EventListener);
        document.removeEventListener("mouseup", handleEnd);
        document.removeEventListener("touchmove", handleMove as EventListener);
        document.removeEventListener("touchend", handleEnd);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        document.body.style.webkitUserSelect = "";
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);

  return {
    sidebarWidth,
    isResizing,
    handleResizeStart,
    setSidebarWidth,
  };
}

