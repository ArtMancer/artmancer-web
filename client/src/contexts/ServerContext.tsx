'use client';

import { createContext, useContext, useState, ReactNode, useEffect, useRef } from 'react';

type ServerStatus = 'offline' | 'booting' | 'online';

interface ServiceStatus {
  light: boolean;
  heavy: boolean;
}

interface ServerContextType {
  status: ServerStatus;
  serviceStatus: ServiceStatus;
  toggleServer: () => void;
  isReady: boolean;
  checkStatus: (showNotification?: boolean) => Promise<void>;
  isUserShutDown: boolean;
}

const ServerContext = createContext<ServerContextType | undefined>(undefined);

export function ServerProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ServerStatus>('offline');
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>({ light: false, heavy: false });
  const [isUserShutDown, setIsUserShutDown] = useState(false); // Track nếu user đã tắt server
  const checkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isCheckingRef = useRef(false);

  /**
   * Check status thực tế từ backend.
   * Không retry - failed là failed luôn, timeout 5s.
   * Chỉ nên gọi khi local state = online hoặc booting để tối ưu chi phí.
   * @param showNotification - Nếu true, hiện notification khi có thay đổi status
   */
  const checkStatus = async (showNotification: boolean = true) => {
    // Tránh multiple concurrent checks
    if (isCheckingRef.current) {
      return;
    }

    isCheckingRef.current = true;
    
    try {
      const response = await fetch('/api/server-status', {
        method: 'GET',
        cache: 'no-store',
        // Không set timeout: gửi 1 request và đợi backend trả về
      });

      if (response.ok) {
        const data = await response.json();
        const actualStatus = data.status === 'online' ? 'online' : 'offline';
        
        // Update service status
        setServiceStatus({
          light: data.light || false,
          heavy: data.heavy || false,
        });
        
        let notify: { type: 'success' | 'error'; message: string } | null = null;

        // Update status dựa trên kết quả thực tế
        setStatus((currentStatus) => {
          // Nếu đang booting và check thấy online, chuyển sang online
          if (currentStatus === 'booting' && actualStatus === 'online') {
            if (showNotification) {
              notify = { type: 'success', message: 'Server is online and ready' };
            }
            return 'online';
          }
          // Nếu đang offline và check thấy online, chuyển sang online
          if (currentStatus === 'offline' && actualStatus === 'online') {
            if (showNotification) {
              notify = { type: 'success', message: 'Server is online and ready' };
            }
            return 'online';
          }
          // Nếu đang online nhưng check thấy offline, chuyển sang offline
          if (currentStatus === 'online' && actualStatus === 'offline') {
            if (showNotification) {
              notify = { type: 'error', message: 'Server connection lost' };
            }
            return 'offline';
          }
          return currentStatus;
        });
        
        if (notify) {
          window.dispatchEvent(new CustomEvent('server-notification', {
            detail: notify
          }));
        }
        
        // Reset isUserShutDown nếu server online
        if (actualStatus === 'online') {
          setIsUserShutDown(false);
        }
      } else {
        let notify: { type: 'error'; message: string } | null = null;
        // Failed - set offline if needed
        setStatus((currentStatus) => {
          if (currentStatus === 'online' || currentStatus === 'booting') {
            if (showNotification) {
              notify = { type: 'error', message: 'Failed to connect to server' };
            }
            return 'offline';
          }
          return currentStatus;
        });
        if (notify) {
          window.dispatchEvent(new CustomEvent('server-notification', {
            detail: notify
          }));
        }
      }
    } catch (error) {
      let notify: { type: 'error'; message: string } | null = null;
      // Failed - set offline if needed
      console.error('Error checking server status:', error);
      setStatus((currentStatus) => {
        if (currentStatus === 'online' || currentStatus === 'booting') {
          const errorMessage = error instanceof Error && error.name === 'TimeoutError'
            ? 'Server connection timeout'
            : 'Failed to connect to server';
          if (showNotification) {
            notify = { type: 'error', message: errorMessage };
          }
          return 'offline';
        }
        return currentStatus;
      });
      if (notify) {
        window.dispatchEvent(new CustomEvent('server-notification', {
          detail: notify
        }));
      }
    } finally {
      isCheckingRef.current = false;
    }
  };

  const toggleServer = async () => {
    // Lấy current status trước
    const currentStatus = status;
    
    if (currentStatus === 'online') {
      // Tắt server
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
        checkIntervalRef.current = null;
      }
      setIsUserShutDown(true);
      setStatus('offline');
      window.dispatchEvent(new CustomEvent('server-notification', {
        detail: { type: 'info', message: 'Server shut down' }
      }));
      return;
    }

    // Bật server
    setIsUserShutDown(false);
    setStatus('booting');

    try {
      // 1. Gọi API đánh thức backend
      await fetch('/api/wake-up', { method: 'POST' });

      // 2. Đợi một chút rồi verify status thực tế (chỉ check 1 lần, không retry)
      // Không ép set offline nếu chưa kịp online; user sẽ tự bấm manual check
      setTimeout(async () => {
        await checkStatus(false); // false = không hiện notification
        // Nếu vẫn booting, giữ nguyên trạng thái để user tự refresh
      }, 2000);
    } catch (error) {
      console.error('Lỗi bật server', error);
      setStatus('offline');
      // Trigger notification via custom event
      window.dispatchEvent(new CustomEvent('server-notification', {
        detail: { type: 'error', message: 'Cannot connect to Cloud Server' }
      }));
    }
  };

  // Không auto check để tối ưu chi phí
  // User sẽ manual check qua nút trong ServerControl component

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, []);

  return (
    <ServerContext.Provider
      value={{ status, serviceStatus, toggleServer, isReady: status === 'online', checkStatus, isUserShutDown }}
    >
      {children}
    </ServerContext.Provider>
  );
}

export function useServer() {
  const context = useContext(ServerContext);
  if (!context)
    throw new Error('useServer must be used within ServerProvider');
  return context;
}

