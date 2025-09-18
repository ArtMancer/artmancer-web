import { useState, useRef, useCallback } from "react";

export function useImageTransform() {
  const [transform, setTransform] = useState({ scale: 1 });
  const imageRef = useRef<HTMLImageElement>(null);

  // Reset transform
  const resetView = useCallback(() => {
    setTransform({ scale: 1 });
  }, []);

  // Download function
  const handleDownload = useCallback((uploadedImage: string | null) => {
    if (!uploadedImage) return;
    
    const link = document.createElement('a');
    link.href = uploadedImage;
    link.download = 'artmancer-generated-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  return {
    transform,
    imageRef,
    setTransform,
    resetView,
    handleDownload
  };
}
