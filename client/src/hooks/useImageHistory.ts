import { useState, useCallback } from "react";

/**
 * Simple history hook with Base State pattern.
 * - Index 0: base/clean state.
 * - First stroke -> index 1, so canUndo becomes true immediately.
 */
export function useImageHistory() {
  const [historyStack, setHistoryStack] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex >= 0 && historyIndex < historyStack.length - 1;

  const handleUndo = useCallback(() => {
    if (!canUndo) return null;
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      return historyStack[newIndex];
  }, [canUndo, historyIndex, historyStack]);

  const handleRedo = useCallback(() => {
    if (!canRedo) return null;
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      return historyStack[newIndex];
  }, [canRedo, historyIndex, historyStack]);

  const addToHistory = useCallback(
    (imageData: string) => {
      // Truncate any redo branch and push new state
    const newStack = historyStack.slice(0, historyIndex + 1);
    newStack.push(imageData);
    setHistoryStack(newStack);
    setHistoryIndex(newStack.length - 1);
    return imageData;
    },
    [historyStack, historyIndex]
  );

  const initializeHistory = useCallback((imageData: string) => {
    setHistoryStack([imageData]); // Base state at index 0
    setHistoryIndex(0);
  }, []);

  return {
    historyStack,
    historyIndex,
    canUndo,
    canRedo,
    handleUndo,
    handleRedo,
    addToHistory,
    initializeHistory,
  };
}
