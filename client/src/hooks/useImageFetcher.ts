import { useState, useRef, useCallback, useEffect } from "react";

interface FetchImageParams {
  url: string;
  options?: RequestInit;
}

interface UseImageFetcherResult {
  imageUrl: string | null;
  isLoading: boolean;
  error: string | null;
  fetchImage: (params: FetchImageParams) => Promise<string | null>;
}

/**
 * Generic hook để fetch ảnh dạng blob và quản lý Object URL an toàn bộ nhớ.
 *
 * - Luôn dùng response.blob()
 * - Tự động createObjectURL + revokeObjectURL khi URL thay đổi hoặc unmount
 */
export default function useImageFetcher(): UseImageFetcherResult {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentUrlRef = useRef<string | null>(null);

  const revokeCurrentUrl = useCallback(() => {
    if (currentUrlRef.current) {
      URL.revokeObjectURL(currentUrlRef.current);
      currentUrlRef.current = null;
    }
  }, []);

  const fetchImage = useCallback(
    async ({ url, options }: FetchImageParams): Promise<string | null> => {
      setIsLoading(true);
      setError(null);

      try {
        // Hủy URL cũ (nếu có) trước khi tạo URL mới
        revokeCurrentUrl();

        const response = await fetch(url, {
          ...options,
        });

        if (!response.ok) {
          let message = `HTTP ${response.status}`;
          try {
            const text = await response.text();
            if (text) {
              message = `${message}: ${text}`;
            }
          } catch {
            // ignore parse error, giữ message mặc định
          }
          throw new Error(message);
        }

        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);

        currentUrlRef.current = objectUrl;
        setImageUrl(objectUrl);

        return objectUrl;
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Không thể tải ảnh (blob).";
        setError(msg);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [revokeCurrentUrl]
  );

  // Cleanup khi unmount
  useEffect(() => {
    return () => {
      revokeCurrentUrl();
    };
  }, [revokeCurrentUrl]);

  return {
    imageUrl,
    isLoading,
    error,
    fetchImage,
  };
}


