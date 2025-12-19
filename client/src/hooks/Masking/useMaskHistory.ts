import { useState, useRef, useCallback } from "react";
import type { CanvasState } from "./useMaskDrawing";

// Command Pattern Interface
interface Command {
  execute(): void;
  undo(): void;
  redo(): void;
  hasContent(): boolean;
  isEmpty(): boolean; // Added for optimization
}

// Concrete Command: Canvas Draw Operation
class CanvasDrawCommand implements Command {
  private canvas: HTMLCanvasElement;
  private previousState: CanvasState;
  private currentState: CanvasState;

  constructor(
    canvas: HTMLCanvasElement,
    previousState: CanvasState,
    currentState: CanvasState
  ) {
    this.canvas = canvas;
    this.previousState = previousState;
    this.currentState = currentState;
  }

  private restoreCanvasState(state: CanvasState): void {
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (state.imageData && state.hasContent) {
        ctx.putImageData(state.imageData, 0, 0);
      }
    } catch (error) {
      console.error("Failed to restore canvas state:", error);
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  execute(): void {
    this.restoreCanvasState(this.currentState);
  }

  undo(): void {
    this.restoreCanvasState(this.previousState);
  }

  redo(): void {
    this.execute();
  }

  hasContent(): boolean {
    return this.currentState.hasContent;
  }

  isEmpty(): boolean {
    return !this.currentState.hasContent;
  }
}

// Concrete Command: Canvas Clear Operation
class CanvasClearCommand implements Command {
  private canvas: HTMLCanvasElement;
  private previousState: CanvasState;

  constructor(canvas: HTMLCanvasElement, previousState: CanvasState) {
    this.canvas = canvas;
    this.previousState = previousState;
  }

  execute(): void {
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (ctx) {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  undo(): void {
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    try {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (this.previousState.imageData) {
        ctx.putImageData(this.previousState.imageData, 0, 0);
      }
    } catch (error) {
      console.error("Failed to restore canvas state during clear undo:", error);
    }
  }

  redo(): void {
    this.execute();
  }

  hasContent(): boolean {
    return false; // Clear command always results in no content
  }

  isEmpty(): boolean {
    // Nếu trước đó canvas đã trống thì clear là no-op, không cần đưa vào history
    return !this.previousState.hasContent;
  }
}

// Command History Manager
class CommandHistory {
  public commands: Command[] = [];
  public currentIndex: number = -1;
  private maxHistorySize: number;

  constructor(maxHistorySize: number = 20) {
    this.maxHistorySize = maxHistorySize;
  }

  executeCommand(command: Command): void {
    // Bỏ qua command không thay đổi nội dung canvas
    if (command.isEmpty()) {
      return;
    }

    // Remove any future commands if we're not at the end
    if (this.currentIndex < this.commands.length - 1) {
      this.commands = this.commands.slice(0, this.currentIndex + 1);
    }

    // Add new command
    this.commands.push(command);
    this.currentIndex = this.commands.length - 1;

    // Execute the command after adding to history
    command.execute();

    // Limit history size
    if (this.commands.length > this.maxHistorySize) {
      this.commands.shift();
      this.currentIndex = Math.max(0, this.currentIndex - 1);
    }
  }

  undo(): boolean {
    if (this.canUndo()) {
      const command = this.commands[this.currentIndex];
      command.undo();
      this.currentIndex--;
      return true;
    }
    return false;
  }

  redo(): boolean {
    if (this.canRedo()) {
      this.currentIndex++;
      const command = this.commands[this.currentIndex];
      command.redo();
      return true;
    }
    return false;
  }

  canUndo(): boolean {
    return this.currentIndex >= 0;
  }

  canRedo(): boolean {
    return this.currentIndex < this.commands.length - 1;
  }

  hasContent(): boolean {
    if (this.currentIndex >= 0 && this.currentIndex < this.commands.length) {
      const command = this.commands[this.currentIndex];
      return command.hasContent();
    }
    return false;
  }

  getHistoryLength(): number {
    return this.commands.length;
  }

  getCurrentIndex(): number {
    return this.currentIndex;
  }

  clear(): void {
    this.commands = [];
    this.currentIndex = -1;
  }

  getHistoryInfo() {
    return {
      totalCommands: this.commands.length,
      currentIndex: this.currentIndex,
      canUndo: this.canUndo(),
      canRedo: this.canRedo(),
    };
  }
}

export interface UseMaskHistoryParams {
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  captureCanvasState: () => CanvasState;
  maxHistorySize?: number;
}

export interface UseMaskHistoryReturn {
  historyState: {
    canUndo: boolean;
    canRedo: boolean;
    hasContent: boolean;
    historyLength: number;
    currentIndex: number;
  };
  updateHistoryState: () => void;
  executeDrawCommand: (previousState: CanvasState, currentState: CanvasState) => void;
  executeClearCommand: (previousState: CanvasState) => void;
  ensureBaseState: () => void;
  deleteLastCommand: () => void;
  undoMask: () => void;
  redoMask: () => void;
  clearMask: () => void;
  resetMaskHistory: () => void;
  syncHistoryWithCanvas: () => void;
  maskHistoryIndex: number;
  maskHistoryLength: number;
  canUndo: boolean;
  canRedo: boolean;
}

/**
 * Hook to manage mask drawing history using Command Pattern
 * Handles undo/redo operations and command execution
 */
export function useMaskHistory(
  params: UseMaskHistoryParams
): UseMaskHistoryReturn {
  const { maskCanvasRef, captureCanvasState, maxHistorySize = 20 } = params;

  const commandHistory = useRef(new CommandHistory(maxHistorySize));
  const [historyState, setHistoryState] = useState({
    canUndo: false,
    canRedo: false,
    hasContent: false,
    historyLength: 0,
    currentIndex: -1,
  });

  // Helper function to update history state
  const updateHistoryState = useCallback(() => {
    const history = commandHistory.current;
    const newState = {
      canUndo: history.canUndo(),
      canRedo: history.canRedo(),
      hasContent: history.hasContent(),
      historyLength: history.getHistoryLength(),
      currentIndex: history.getCurrentIndex(),
    };

    // Force UI sync even on first write so Undo enables immediately
    setHistoryState((prev) => {
      if (
        prev.canUndo === newState.canUndo &&
        prev.canRedo === newState.canRedo &&
        prev.hasContent === newState.hasContent &&
        prev.historyLength === newState.historyLength &&
        prev.currentIndex === newState.currentIndex
      ) {
        return prev;
      }
      return newState;
    });
  }, []);

  // Execute draw command
  const executeDrawCommand = useCallback(
    (previousState: CanvasState, currentState: CanvasState) => {
      const canvas = maskCanvasRef.current;
      if (!canvas) return;

      const drawCommand = new CanvasDrawCommand(canvas, previousState, currentState);
      commandHistory.current.executeCommand(drawCommand);
      updateHistoryState();
    },
    [maskCanvasRef, updateHistoryState]
  );

  // Ensure base state exists at index 0 (empty or current snapshot)
  const ensureBaseState = useCallback(() => {
    if (historyState.historyLength > 0) return;
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    const baseState = captureCanvasState();
    const baseCommand = new CanvasDrawCommand(canvas, baseState, baseState);
    commandHistory.current.executeCommand(baseCommand);
    updateHistoryState();
  }, [captureCanvasState, historyState.historyLength, maskCanvasRef, updateHistoryState]);

  // Execute clear command
  const executeClearCommand = useCallback(
    (previousState: CanvasState) => {
      const canvas = maskCanvasRef.current;
      if (!canvas) return;

      // Optimization: Don't create clear command if canvas is already empty
      if (!previousState.hasContent) {
        return;
      }

      const clearCommand = new CanvasClearCommand(canvas, previousState);
      commandHistory.current.executeCommand(clearCommand);
      updateHistoryState();
    },
    [maskCanvasRef, updateHistoryState]
  );

  // Undo mask
  const undoMask = useCallback(() => {
    if (commandHistory.current.undo()) {
      updateHistoryState();
    }
  }, [updateHistoryState]);

  // Delete last command (pop without redo ability)
  const deleteLastCommand = useCallback(() => {
    const history = commandHistory.current;
    if (history.getHistoryLength() === 0) return;
    history.commands = history.commands.slice(0, -1);
    history.currentIndex = history.commands.length - 1;
    // Apply the new top state to canvas
    const canvas = maskCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const topCommand = history.commands[history.currentIndex];
        if (topCommand && !topCommand.isEmpty()) {
          topCommand.execute();
        }
      }
    }
    updateHistoryState();
  }, [maskCanvasRef, updateHistoryState]);

  // Redo mask
  const redoMask = useCallback(() => {
    if (commandHistory.current.redo()) {
      updateHistoryState();
    }
  }, [updateHistoryState]);

  // Clear mask
  const clearMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    // Capture current state before clearing
    const currentState = captureCanvasState();

    // Execute clear command
    executeClearCommand(currentState);
  }, [captureCanvasState, executeClearCommand, maskCanvasRef]);

  // Reset mask history
  const resetMaskHistory = () => {
    commandHistory.current.clear();
    updateHistoryState();
  };

  // Sync history với canvas state hiện tại (dùng sau khi restore canvas)
  const syncHistoryWithCanvas = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;

    // Capture current canvas state
    const currentState = captureCanvasState();
    
    // Reset history
    commandHistory.current.clear();
    
    // Add current state như base state
    const baseCommand = new CanvasDrawCommand(canvas, currentState, currentState);
    commandHistory.current.commands = [baseCommand];
    commandHistory.current.currentIndex = 0;
    
    updateHistoryState();
    
    console.log("✅ [useMaskHistory] Synced history with current canvas state", {
      hasContent: currentState.hasContent,
    });
  }, [maskCanvasRef, captureCanvasState, updateHistoryState]);

  return {
    historyState,
    updateHistoryState,
    executeDrawCommand,
    executeClearCommand,
    ensureBaseState,
    deleteLastCommand,
    undoMask,
    redoMask,
    clearMask,
    resetMaskHistory,
    syncHistoryWithCanvas,
    maskHistoryIndex: historyState.currentIndex,
    maskHistoryLength: historyState.historyLength,
    canUndo: historyState.canUndo,
    canRedo: historyState.canRedo,
  };
}

