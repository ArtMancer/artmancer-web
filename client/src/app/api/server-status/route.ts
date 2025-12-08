import { NextResponse } from 'next/server';

/**
 * API route để check status thực tế của backend servers thông qua API Gateway.
 * API Gateway sẽ aggregate health checks từ tất cả services.
 */
export async function GET() {
  try {
    const GATEWAY_URL =
      process.env.NEXT_PUBLIC_API_GATEWAY_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      'https://nxan2911--api-gateway.modal.run';

    // Gọi API Gateway để lấy aggregated health status
    // Tăng timeout lên 15s để đảm bảo đủ thời gian cho health checks (mỗi service timeout 2s, chạy song song)
    const response = await fetch(`${GATEWAY_URL}/api/system/health`, {
      method: 'GET',
      cache: 'no-store',
      signal: AbortSignal.timeout(15000), // 15s timeout (API Gateway checks services concurrently with 2s timeout each)
    });

    if (response.ok) {
      const data = await response.json();
      
      // API Gateway trả về:
      // {
      //   "status": "healthy" | "degraded",
      //   "services": {
      //     "generation": {"status": "healthy", ...},
      //     "segmentation": {"status": "healthy", ...},
      //     ...
      //   }
      // }
      
      return NextResponse.json({
        status: data.status === 'healthy' ? 'online' : 'offline',
        services: data.services || {},
      });
    } else {
      // API Gateway không available
      return NextResponse.json({
        status: 'offline',
        services: {},
      });
    }
  } catch (error) {
    console.error('Server status check error:', error);
    return NextResponse.json(
      { status: 'offline', services: {}, error: 'Failed to check status' },
      { status: 500 }
    );
  }
}
