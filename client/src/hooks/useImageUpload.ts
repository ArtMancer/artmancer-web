import { useState, useCallback, useEffect } from "react";

interface ImageDimensions {
  width: number;
  height: number;
}

export function useImageUpload() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<ImageDimensions | null>(null);
  const [displayScale, setDisplayScale] = useState(1);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [modifiedImage, setModifiedImage] = useState<string | null>(null);

  // Calculate display scale for large images
  const calculateDisplayScale = useCallback((dimensions: ImageDimensions) => {
    // Calculate viewport constraints (leaving some padding)
    const maxViewportWidth = window.innerWidth * 0.7; // 70% of viewport width
    const maxViewportHeight = window.innerHeight * 0.7; // 70% of viewport height
    
    // Calculate scale factor to fit within viewport
    const scaleX = maxViewportWidth / dimensions.width;
    const scaleY = maxViewportHeight / dimensions.height;
    const scaleFactor = Math.min(scaleX, scaleY, 1); // Don't scale up, only down
    
    return scaleFactor;
  }, []);

  // Update display scale when image dimensions or window size changes
  useEffect(() => {
    if (!imageDimensions) {
      queueMicrotask(() => setDisplayScale(1));
      return;
    }

    const updateScale = () => {
      const newScale = calculateDisplayScale(imageDimensions);
      queueMicrotask(() => setDisplayScale(newScale));
    };

    updateScale();
    window.addEventListener("resize", updateScale);

    return () => {
      window.removeEventListener("resize", updateScale);
    };
  }, [imageDimensions, calculateDisplayScale]);

  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageData = e.target?.result as string;
        
        // Create an image element to get dimensions
        const img = document.createElement('img');
        img.onload = () => {
          setImageDimensions({ width: img.width, height: img.height });
        };
        img.src = imageData;
        
        setUploadedImage(imageData);
        setOriginalImage(imageData);
        setModifiedImage(imageData);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const removeImage = useCallback(() => {
    setUploadedImage(null);
    setImageDimensions(null);
    setDisplayScale(1);
    setOriginalImage(null);
    setModifiedImage(null);
  }, []);

  // Simple click handler for image upload
  const handleImageClick = useCallback((e: React.MouseEvent) => {
    if (!uploadedImage) return;
    
    e.preventDefault();
    const fileInput = document.getElementById('image-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  }, [uploadedImage]);

  return {
    uploadedImage,
    imageDimensions,
    displayScale,
    originalImage,
    modifiedImage,
    handleImageUpload,
    removeImage,
    handleImageClick,
    setUploadedImage,
    setModifiedImage,
    setImageDimensions
  };
}
