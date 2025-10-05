import React, { useEffect, useState } from 'react';
import { MdCheckCircle, MdError, MdInfo, MdWarning, MdClose } from 'react-icons/md';

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

interface NotificationProps {
  type: NotificationType;
  message: string;
  isVisible: boolean;
  duration?: number; // Auto-hide duration in milliseconds
  onClose: () => void;
  showCloseButton?: boolean;
  position?: 'top' | 'bottom';
}

const notificationConfig = {
  success: {
    icon: MdCheckCircle,
    bgColor: 'bg-emerald-500',
    borderColor: 'border-emerald-600',
    iconColor: 'text-emerald-100',
  },
  error: {
    icon: MdError,
    bgColor: 'bg-red-500',
    borderColor: 'border-red-600',
    iconColor: 'text-red-100',
  },
  warning: {
    icon: MdWarning,
    bgColor: 'bg-amber-500',
    borderColor: 'border-amber-600',
    iconColor: 'text-amber-100',
  },
  info: {
    icon: MdInfo,
    bgColor: 'bg-blue-500',
    borderColor: 'border-blue-600',
    iconColor: 'text-blue-100',
  },
};

export default function Notification({
  type,
  message,
  isVisible,
  duration = 5000,
  onClose,
  showCloseButton = true,
  position = 'top',
}: NotificationProps) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [timeLeft, setTimeLeft] = useState(duration);

  const config = notificationConfig[type];
  const Icon = config.icon;

  // Handle auto-hide timer
  useEffect(() => {
    if (isVisible && duration > 0) {
      const interval = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 100) {
            return 0;
          }
          return prev - 100;
        });
      }, 100);

      return () => clearInterval(interval);
    }
  }, [isVisible, duration]);

  // Handle auto-close when timer reaches zero
  useEffect(() => {
    if (timeLeft <= 0 && isVisible && duration > 0) {
      onClose();
    }
  }, [timeLeft, isVisible, duration, onClose]);

  // Handle animations
  useEffect(() => {
    if (isVisible) {
      setIsAnimating(true);
      setTimeLeft(duration);
    } else {
      const timer = setTimeout(() => setIsAnimating(false), 300);
      return () => clearTimeout(timer);
    }
  }, [isVisible, duration]);

  // Reset timer when message changes
  useEffect(() => {
    if (isVisible) {
      setTimeLeft(duration);
    }
  }, [message, duration, isVisible]);

  if (!isAnimating && !isVisible) return null;

  const progressPercentage = duration > 0 ? (timeLeft / duration) * 100 : 0;

  return (
    <div
      className={`fixed left-1/2 transform -translate-x-1/2 z-50 transition-all duration-300 ease-out ${
        position === 'top' ? 'top-6' : 'bottom-6'
      } ${
        isVisible
          ? 'opacity-100 translate-y-0 scale-100'
          : position === 'top'
          ? 'opacity-0 -translate-y-4 scale-95'
          : 'opacity-0 translate-y-4 scale-95'
      }`}
      style={{
        maxWidth: 'calc(100vw - 2rem)',
      }}
    >
      <div
        className={`${config.bgColor} ${config.borderColor} border-l-4 text-white rounded-lg shadow-2xl backdrop-blur-sm relative overflow-hidden`}
        style={{
          background: `linear-gradient(135deg, ${config.bgColor.replace('bg-', 'rgb(var(--')} / 0.95), ${config.bgColor.replace('bg-', 'rgb(var(--')} / 0.85))`,
        }}
      >
        {/* Progress bar */}
        {duration > 0 && (
          <div
            className="absolute top-0 left-0 h-1 bg-white/30 transition-all duration-100 ease-linear"
            style={{ width: `${progressPercentage}%` }}
          />
        )}

        <div className="px-4 py-3 flex items-start gap-3">
          {/* Icon */}
          <Icon className={`${config.iconColor} flex-shrink-0 mt-0.5`} size={20} />

          {/* Message */}
          <div className="flex-1 min-w-0">
            <p className="text-white font-medium text-sm leading-relaxed break-words">
              {message}
            </p>
          </div>

          {/* Close button */}
          {showCloseButton && (
            <button
              onClick={onClose}
              className="flex-shrink-0 text-white/80 hover:text-white transition-colors p-1 rounded hover:bg-white/10"
              aria-label="Close notification"
            >
              <MdClose size={18} />
            </button>
          )}
        </div>

        {/* Subtle glow effect */}
        <div className="absolute inset-0 rounded-lg shadow-lg pointer-events-none" />
      </div>
    </div>
  );
}