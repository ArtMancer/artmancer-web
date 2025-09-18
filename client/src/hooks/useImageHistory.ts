import { useState, useCallback } from "react";

export function useImageHistory() {
  const [historyStack, setHistoryStack] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      return historyStack[newIndex];
    }
    return null;
  }, [historyIndex, historyStack]);

  const handleRedo = useCallback(() => {
    if (historyIndex < historyStack.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      return historyStack[newIndex];
    }
    return null;
  }, [historyIndex, historyStack]);

  const addToHistory = useCallback((imageData: string) => {
    const newStack = historyStack.slice(0, historyIndex + 1);
    newStack.push(imageData);
    setHistoryStack(newStack);
    setHistoryIndex(newStack.length - 1);
    return imageData;
  }, [historyStack, historyIndex]);

  const initializeHistory = useCallback((imageData: string) => {
    setHistoryStack([imageData]);
    setHistoryIndex(0);
  }, []);

  return {
    historyStack,
    historyIndex,
    handleUndo,
    handleRedo,
    addToHistory,
    initializeHistory
  };
}
