import { useState, useCallback } from "react";

export interface UseReferenceImageReturn {
  referenceImage: string | null;
  referenceMaskR: string | null;
  isRefEditorOpen: boolean;
  pendingRefImage: string | null;
  handleReferenceImageUpload: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  handleRefEditorSubmit: (
    processedImage: string,
    maskData?: string | null
  ) => Promise<void>;
  handleRefEditorClose: () => void;
  handleRemoveReferenceImage: () => void;
  handleEditReferenceImage: () => void;
  setReferenceImage: (image: string | null) => void;
  setReferenceMaskR: (mask: string | null) => void;
}

/**
 * Hook to manage reference image state and handlers for object insertion tasks
 * Handles reference image upload, editing, and mask management
 */
export function useReferenceImage(): UseReferenceImageReturn {
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  const [referenceMaskR, setReferenceMaskR] = useState<string | null>(null);
  const [isRefEditorOpen, setIsRefEditorOpen] = useState(false);
  const [pendingRefImage, setPendingRefImage] = useState<string | null>(null);

  const handleReferenceImageUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files && event.target.files[0]) {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target && typeof e.target.result === "string") {
            // Open editor instead of setting directly
            setPendingRefImage(e.target.result);
            setIsRefEditorOpen(true);
          }
        };
        reader.readAsDataURL(file);
      }
    },
    []
  );

  const handleRefEditorSubmit = useCallback(
    async (processedImage: string, maskData?: string | null) => {
      // Store both reference image and Mask R (for two-source mask workflow)
      setReferenceImage(processedImage);
      setReferenceMaskR(maskData || null);
      setPendingRefImage(null);
      setIsRefEditorOpen(false);
    },
    []
  );

  const handleRefEditorClose = useCallback(() => {
    setPendingRefImage(null);
    setIsRefEditorOpen(false);
  }, []);

  const handleRemoveReferenceImage = useCallback(() => {
    setReferenceImage(null);
    setReferenceMaskR(null);
  }, []);

  const handleEditReferenceImage = useCallback(() => {
    if (referenceImage) {
      // Open editor with current reference image
      setPendingRefImage(referenceImage);
      setIsRefEditorOpen(true);
    }
  }, [referenceImage]);

  return {
    referenceImage,
    referenceMaskR,
    isRefEditorOpen,
    pendingRefImage,
    handleReferenceImageUpload,
    handleRefEditorSubmit,
    handleRefEditorClose,
    handleRemoveReferenceImage,
    handleEditReferenceImage,
    setReferenceImage,
    setReferenceMaskR,
  };
}

