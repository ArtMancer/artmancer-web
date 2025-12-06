"use client";

import {
  createContext,
  useContext,
  useState,
  ReactNode,
  useEffect,
  useRef,
} from "react";

type ServerStatus = "offline" | "online";

interface ServiceStatus {
  generation?: boolean;
  segmentation?: boolean;
  image_utils?: boolean;
  job_manager?: boolean;
}

interface ServerContextType {
  status: ServerStatus;
  serviceStatus: ServiceStatus;
  isReady: boolean;
  checkStatus: (showNotification?: boolean) => Promise<void>;
}

const ServerContext = createContext<ServerContextType | undefined>(undefined);

export function ServerProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ServerStatus>("offline");
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>({});
  const isCheckingRef = useRef(false);
  const hasInitialCheckRef = useRef(false);

  /**
   * Check status từ API Gateway.
   * @param showNotification - Nếu true, hiện notification khi có thay đổi status
   */
  const checkStatus = async (showNotification: boolean = true) => {
    // Tránh multiple concurrent checks
    if (isCheckingRef.current) {
      return;
    }

    isCheckingRef.current = true;

    try {
      const response = await fetch("/api/server-status", {
        method: "GET",
        cache: "no-store",
        signal: AbortSignal.timeout(15000), // 15s timeout (API Gateway checks 4 services, mỗi service 3s)
      });

      if (response.ok) {
        const data = await response.json();
        const actualStatus = data.status === "online" ? "online" : "offline";

        // Update service status từ API Gateway response
        if (data.services) {
          setServiceStatus({
            generation: data.services.generation?.status === "healthy",
            segmentation: data.services.segmentation?.status === "healthy",
            image_utils: data.services.image_utils?.status === "healthy",
            job_manager: data.services.job_manager?.status === "healthy",
          });
        }

        let notify: { type: "success" | "error"; message: string } | null =
          null;

        // Update status
        setStatus((currentStatus) => {
          if (currentStatus !== actualStatus) {
            if (showNotification) {
              notify =
                actualStatus === "online"
                  ? {
                      type: "success",
                      message: "Server is online and ready",
                    }
                  : {
                      type: "error",
                      message: "Server connection lost",
                    };
            }
            return actualStatus;
          }
          return currentStatus;
        });

        if (notify) {
          window.dispatchEvent(
            new CustomEvent("server-notification", {
              detail: notify,
            })
          );
        }
      } else {
        // Failed - set offline
        let notify: { type: "error"; message: string } | null = null;
        setStatus((currentStatus) => {
          if (currentStatus === "online") {
            if (showNotification) {
              notify = {
                type: "error",
                message: "Failed to connect to server",
              };
            }
            return "offline";
          }
          return currentStatus;
        });
        if (notify) {
          window.dispatchEvent(
            new CustomEvent("server-notification", {
              detail: notify,
            })
          );
        }
      }
    } catch (error) {
      // Failed - set offline
      let notify: { type: "error"; message: string } | null = null;
      if (showNotification) {
        console.error("Error checking server status:", error);
      }
      setStatus((currentStatus) => {
        if (currentStatus === "online") {
          const errorMessage =
            error instanceof Error && error.name === "TimeoutError"
              ? "Server connection timeout"
              : "Failed to connect to server";
          if (showNotification) {
            notify = { type: "error", message: errorMessage };
          }
          return "offline";
        }
        return currentStatus;
      });
      if (notify) {
        window.dispatchEvent(
          new CustomEvent("server-notification", {
            detail: notify,
          })
        );
      }
    } finally {
      isCheckingRef.current = false;
    }
  };

  // Initial check khi app boot
  useEffect(() => {
    if (hasInitialCheckRef.current) return;
    hasInitialCheckRef.current = true;

    // Đợi một chút để đảm bảo app đã load xong
    setTimeout(() => {
      checkStatus(false); // Không hiện notification cho initial check
    }, 1000);
  }, []);

  return (
    <ServerContext.Provider
      value={{
        status,
        serviceStatus,
        isReady: status === "online",
        checkStatus,
      }}
    >
      {children}
    </ServerContext.Provider>
  );
}

export function useServer() {
  const context = useContext(ServerContext);
  if (!context) throw new Error("useServer must be used within ServerProvider");
  return context;
}
