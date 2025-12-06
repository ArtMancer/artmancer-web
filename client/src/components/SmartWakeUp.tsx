'use client';

import { useState, ReactNode } from 'react';

interface SmartWakeUpProps {
  children: ReactNode;
}

/**
 * SmartWakeUp - Component wrapper để đánh thức backend khi user có intent.
 * 
 * Trigger wake-up khi:
 * - User hover vào khu vực (onMouseEnter)
 * - User focus vào input/button (onFocus)
 * - User touch trên mobile (onTouchStart)
 * 
 * Chỉ trigger 1 lần duy nhất để tránh spam requests.
 */
export default function SmartWakeUp({ children }: SmartWakeUpProps) {
  const [hasTriggered, setHasTriggered] = useState(false);

  const triggerWakeUp = () => {
    if (hasTriggered) return; // Đã gọi rồi thì thôi

    setHasTriggered(true);
    console.log('🚀 User intent detected. Waking up backend...');

    // Gọi API route của Next.js
    fetch('/api/wake-up', {
      method: 'POST',
      keepalive: true, // Đảm bảo request vẫn gửi đi dù user chuyển trang ngay
    }).catch((err) => {
      console.error('Wake up request failed:', err);
    });
  };

  return (
    <div
      // Sự kiện rê chuột vào
      onMouseEnter={triggerWakeUp}
      // Sự kiện focus vào (keyboard navigation hoặc click)
      onFocus={triggerWakeUp}
      // Sự kiện chạm trên mobile
      onTouchStart={triggerWakeUp}
      className="w-full h-full" // Giữ layout không bị vỡ
    >
      {children}
    </div>
  );
}

