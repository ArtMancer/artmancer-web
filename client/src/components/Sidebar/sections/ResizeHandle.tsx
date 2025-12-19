"use client";

import React from "react";

interface ResizeHandleProps {
  /** Whether sidebar is open */
  isOpen: boolean;
  /** Whether currently resizing */
  isResizing?: boolean;
  /** Current sidebar width */
  width: number;
  /** Callback when resize starts */
  onResizeStart?: (e: React.MouseEvent) => void;
  /** Callback when width changes (for keyboard accessibility) */
  onWidthChange?: (width: number) => void;
}

/**
 * Resize Handle Component
 * 
 * Provides a draggable handle on the left edge of the sidebar for resizing.
 * Supports mouse, touch, and keyboard interactions.
 * 
 * User Interaction Flow:
 * 1. User clicks/touches handle → onResizeStart triggers → parent handles drag
 * 2. User presses ArrowLeft/ArrowRight → width changes → saved to localStorage
 * 3. While resizing → handle highlights with accent color
 * 
 * State Changes:
 * - isResizing change → visual feedback (highlight color)
 * - width change → persisted to localStorage for next session
 * 
 * Why this component:
 * - Isolated resize logic for better maintainability
 * - Reusable if needed elsewhere
 * - Clear separation of concerns
 */
export default function ResizeHandle({
  isOpen,
  isResizing = false,
  width,
  onResizeStart,
  onWidthChange,
}: ResizeHandleProps) {
  // Early return if sidebar is closed or no resize handler
  if (!isOpen || !onResizeStart) {
    return null;
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Keyboard accessibility: Arrow keys to resize
    if (e.key === "ArrowLeft" && onWidthChange) {
      e.preventDefault();
      const newWidth = Math.max(280, width - 10);
      onWidthChange(newWidth);
      // Persist to localStorage for next session
      localStorage.setItem("sidebarWidth", String(newWidth));
    } else if (e.key === "ArrowRight" && onWidthChange) {
      e.preventDefault();
      const newWidth = Math.min(600, width + 10);
      onWidthChange(newWidth);
      // Persist to localStorage for next session
      localStorage.setItem("sidebarWidth", String(newWidth));
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    // Touch support for mobile devices
    const touch = e.touches[0];
    if (touch && onResizeStart) {
      // Convert touch event to mouse event for compatibility
      onResizeStart(e as unknown as React.MouseEvent);
    }
  };

  return (
    <div
      className={`absolute top-0 left-0 w-3 h-full cursor-col-resize z-10 transition-all duration-150 ${
        isResizing
          ? "bg-[var(--primary-accent)] opacity-100"
          : "bg-transparent hover:bg-[var(--primary-accent)] hover:opacity-60"
      }`}
      style={{
        transform: "translateX(-50%)", // Center on the left edge
        willChange: isResizing ? "background-color" : "auto",
      }}
      onMouseDown={onResizeStart}
      onTouchStart={handleTouchStart}
      role="separator"
      aria-label="Resize sidebar"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    />
  );
}

