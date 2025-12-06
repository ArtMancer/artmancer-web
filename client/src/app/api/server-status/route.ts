import { NextResponse } from 'next/server';

/**
 * API route để check status thực tế của backend servers.
 * Chỉ gọi khi cần thiết để tối ưu chi phí serverless.
 * Sử dụng /api/health endpoint giống notification system để đảm bảo consistency.
 */
export async function GET() {
  try {
    const LIGHT_URL =
      process.env.NEXT_PUBLIC_API_URL ||
      'https://nxan2911--artmancer-lightservice-serve.modal.run';
    const HEAVY_URL =
      process.env.NEXT_PUBLIC_RUNPOD_GENERATE_URL ||
      'https://nxan2911--artmancer-heavyservice-serve.modal.run';

    // Check cả 2 services bằng /ping endpoint (nhẹ nhất, đủ để check status)
    // Sử dụng Promise.allSettled để không fail nếu 1 service down
    const [lightResult, heavyResult] = await Promise.allSettled([
      fetch(`${LIGHT_URL}/ping`, {
        method: 'GET',
        cache: 'no-store',
        signal: AbortSignal.timeout(5000), // 5s timeout
      }),
      fetch(`${HEAVY_URL}/ping`, {
        method: 'GET',
        cache: 'no-store',
        signal: AbortSignal.timeout(5000),
      }),
    ]);

    // Check if any service is online
    const lightOnline =
      lightResult.status === 'fulfilled' &&
      lightResult.value.status === 200;
    const heavyOnline =
      heavyResult.status === 'fulfilled' &&
      heavyResult.value.status === 200;

    // Nếu ít nhất 1 service trả về 200, coi như online
    const isOnline = lightOnline || heavyOnline;

    return NextResponse.json({
      status: isOnline ? 'online' : 'offline',
      light: lightOnline,
      heavy: heavyOnline,
    });
  } catch (error) {
    console.error('Server status check error:', error);
    return NextResponse.json(
      { status: 'offline', error: 'Failed to check status' },
      { status: 500 }
    );
  }
}

