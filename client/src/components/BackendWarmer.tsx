'use client';

import { useEffect, useRef } from 'react';

/**
 * BackendWarmer - Component vô hình để đánh thức Modal containers khi web load.
 * 
 * Gửi ping tới cả LightService và HeavyService để warm up containers,
 * giảm cold start time khi user thực sự cần sử dụng các tính năng.
 */
export default function BackendWarmer() {
  const hasWarmedUp = useRef(false);

  useEffect(() => {
    if (hasWarmedUp.current) return;

    const warmUp = async () => {
      try {
        const LIGHT_URL =
          process.env.NEXT_PUBLIC_API_URL ||
          'https://nxan2911--artmancer-lightservice-serve.modal.run';
        const HEAVY_URL =
          process.env.NEXT_PUBLIC_RUNPOD_GENERATE_URL ||
          'https://nxan2911--artmancer-heavyservice-serve.modal.run';

        // Fire-and-forget ping to both services
        // Sử dụng keepalive để request vẫn gửi được ngay cả khi network chập chờn
        fetch(`${LIGHT_URL}/ping`, { method: 'GET', keepalive: true }).catch(
          () => {}
        );
        fetch(`${HEAVY_URL}/ping`, { method: 'GET', keepalive: true }).catch(
          () => {}
        );

        console.log('🚀 Backend warmer: ping sent to Modal containers');
        hasWarmedUp.current = true;
      } catch (error) {
        console.error('⚠️ Backend warmer failed:', error);
      }
    };

    warmUp();
  }, []);

  return null; // Component này không render gì
}

