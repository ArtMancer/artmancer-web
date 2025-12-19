import { useCallback, useRef, useState } from "react";

export type StreamEvent =
  | {
      status: "processing";
      message?: string;
      progress?: number;
      current_step?: number | null;
      total_steps?: number | null;
      task_id?: string;
    }
  | {
      status: "completed";
      task_id?: string;
      image?: string;
    }
  | {
      status: "error";
      message: string;
      task_id?: string;
    };

interface UseModalStreamResult<TEvent = StreamEvent> {
  isStreaming: boolean;
  error: string | null;
  logs: string[];
  start: (
    body: unknown,
    options?: {
      signal?: AbortSignal;
      onEvent?: (event: TEvent) => void;
    }
  ) => Promise<void>;
  cancel: () => void;
}

/**
 * Generic SSE stream hook using fetch + ReadableStream reader.
 * - Không dùng EventSource (hỗ trợ POST body).
 * - Không ràng buộc vào schema backend cụ thể, onEvent nhận raw JSON event.
 */
export function useModalStream<TEvent = StreamEvent>(
  url: string
): UseModalStreamResult<TEvent> {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const start = useCallback<
    UseModalStreamResult<TEvent>["start"]
  >(async (body, options) => {
    if (isStreaming) {
      return;
    }

    setIsStreaming(true);
    setError(null);
    setLogs([]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const combinedSignal = options?.signal
      ? new AbortSignalProxy([controller.signal, options.signal])
      : controller.signal;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal: combinedSignal as AbortSignal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Persistent buffer-based SSE parsing
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let separatorIndex = buffer.indexOf("\n\n");
        while (separatorIndex !== -1) {
          const rawEvent = buffer.slice(0, separatorIndex).trim();
          buffer = buffer.slice(separatorIndex + 2);

          if (rawEvent.startsWith("data:")) {
            const jsonStr = rawEvent.replace(/^data:\s*/, "");
            try {
              const parsed = JSON.parse(jsonStr) as TEvent;
              options?.onEvent?.(parsed);

              // Simple log helper for debug
              let message: string;
              if (typeof parsed === "object" && parsed !== null) {
                const obj = parsed as {
                  loading_message?: unknown;
                  status?: unknown;
                };
                if (typeof obj.loading_message === "string") {
                  message = obj.loading_message;
                } else if (typeof obj.status === "string") {
                  message = obj.status;
                } else {
                  message = jsonStr;
                }
              } else {
                message = jsonStr;
              }
              setLogs((prev) => [...prev, String(message)]);
            } catch (e) {
              console.error("Failed to parse SSE event:", e, jsonStr);
            }
          }

          separatorIndex = buffer.indexOf("\n\n");
        }
      }
    } catch (err) {
      // AbortError: coi như cancel, không set error user-facing
      if (
        err instanceof DOMException &&
        err.name === "AbortError"
      ) {
        return;
      }
      setError((err as Error).message || String(err));
      console.error("useModalStream error:", err);
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  }, [isStreaming, url]);

  return {
    isStreaming,
    error,
    logs,
    start,
    cancel,
  };
}

/**
 * Simple proxy cho nhiều AbortSignal cùng lúc.
 * Abort nếu bất kỳ signal nào bị abort.
 */
class AbortSignalProxy implements AbortSignal {
  private _aborted: boolean;
  readonly reason: unknown;
  onabort: ((this: AbortSignal, ev: Event) => void) | null = null;

  constructor(signals: AbortSignal[]) {
    this._aborted = signals.some((s) => s.aborted);
    this.reason = undefined;

    signals.forEach((signal) => {
      signal.addEventListener(
        "abort",
        (ev) => {
          if (this._aborted) return;
          this._aborted = true;
          this.onabort?.(ev);
        },
        { once: true }
      );
    });
  }

  get aborted() {
    return this._aborted;
  }

  addEventListener: AbortSignal["addEventListener"] = () => {};
  removeEventListener: AbortSignal["removeEventListener"] = () => {};
  dispatchEvent: AbortSignal["dispatchEvent"] = () => false;
}


