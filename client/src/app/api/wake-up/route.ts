import { NextResponse } from 'next/server';

/**
 * API route để đánh thức Modal containers khi user có intent sử dụng.
 * 
 * Route này gọi tới cả LightService và HeavyService /ping endpoints
 * để warm up containers trước khi user thực sự submit request.
 */
export async function POST() {
  try {
    const LIGHT_URL =
      process.env.NEXT_PUBLIC_API_URL ||
      'https://nxan2911--artmancer-lightservice-serve.modal.run';
    const HEAVY_URL =
      process.env.NEXT_PUBLIC_RUNPOD_GENERATE_URL ||
      'https://nxan2911--artmancer-heavyservice-serve.modal.run';

    // Fire-and-forget ping to both services
    // Không cần await để tránh block response
    fetch(`${LIGHT_URL}/ping`, {
      method: 'GET',
      cache: 'no-store',
    }).catch((err) => {
      console.error('Wake up LightService failed silently:', err);
    });

    fetch(`${HEAVY_URL}/ping`, {
      method: 'GET',
      cache: 'no-store',
    }).catch((err) => {
      console.error('Wake up HeavyService failed silently:', err);
    });

    return NextResponse.json({ status: 'sent' });
  } catch (error) {
    console.error('Wake up API route error:', error);
    return NextResponse.json({ status: 'error' }, { status: 500 });
  }
}

