import { useState, useRef, useCallback } from "react";
import type { NotificationType } from "@/components/Notification";

/**
 * Custom hook for managing notification state and auto-hide functionality
 * 
 * Provides:
 * - Notification state (type, message, visibility)
 * - Auto-hide timer management
 * - Helper functions for showing/hiding notifications
 * 
 * Usage:
 * ```tsx
 * const {
 *   notificationType,
 *   notificationMessage,
 *   isNotificationVisible,
 *   showNotification,
 *   hideNotification,
 *   clearNotification,
 * } = useNotification();
 * ```
 * 
 * State Changes:
 * - showNotification() → sets type, message, visibility=true, starts auto-hide timer
 * - hideNotification() → sets visibility=false, clears timer
 * - clearNotification() → resets all state, clears timer
 * - Auto-hide timer → automatically hides notification after timeout
 */
export function useNotification() {
  // Notification state
  const [notificationType, setNotificationType] =
    useState<NotificationType>("success");
  const [notificationMessage, setNotificationMessage] = useState<string>("");
  const [isNotificationVisible, setIsNotificationVisible] = useState(false);

  // Notification timeout ref for auto-hide functionality
  const notificationTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Clear any existing notification timeout
   * Called before setting a new notification or hiding current one
   */
  const clearNotificationTimeout = useCallback(() => {
    if (notificationTimeoutRef.current) {
      clearTimeout(notificationTimeoutRef.current);
      notificationTimeoutRef.current = null;
    }
  }, []);

  /**
   * Hide notification and clear timeout
   * Called when user manually closes notification or auto-hide triggers
   */
  const hideNotification = useCallback(() => {
    setIsNotificationVisible(false);
    clearNotificationTimeout();
  }, [clearNotificationTimeout]);

  /**
   * Show notification with auto-hide after specified timeout
   * 
   * @param type - Notification type (success, error, info, warning)
   * @param message - Notification message to display
   * @param timeoutMs - Auto-hide timeout in milliseconds (default: 5000ms)
   * 
   * State Changes:
   * - Sets notification type and message
   * - Shows notification (visibility = true)
   * - Starts auto-hide timer
   * - Clears any existing timeout before setting new one
   */
  const showNotification = useCallback(
    (type: NotificationType, message: string, timeoutMs: number = 5000) => {
      // Clear any existing timeout
      clearNotificationTimeout();

      // Set new notification
      setNotificationType(type);
      setNotificationMessage(message);
      setIsNotificationVisible(true);

      // Auto-hide timer
      notificationTimeoutRef.current = setTimeout(() => {
        setIsNotificationVisible(false);
      }, timeoutMs);
    },
    [clearNotificationTimeout]
  );

  /**
   * Clear all notification state and timeout
   * Useful for cleanup or resetting notification system
   */
  const clearNotification = useCallback(() => {
    clearNotificationTimeout();
    setIsNotificationVisible(false);
    setNotificationMessage("");
    setNotificationType("success");
  }, [clearNotificationTimeout]);

  return {
    // State
    notificationType,
    notificationMessage,
    isNotificationVisible,
    // Actions
    showNotification,
    hideNotification,
    clearNotification,
  };
}

