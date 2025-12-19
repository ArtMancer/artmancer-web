"use client";

import React from "react";
import { useLanguage } from "@/contexts/LanguageContext";

interface ImageUploadSectionProps {
  /** Image data URL from parent (source of truth in page.tsx) */
  uploadedImage: string | null;
  /** Whether editing is done (computed in parent from originalImage/modifiedImage) */
  isEditingDone: boolean;
  /** Callback to remove image (handler from page.tsx) */
  onRemoveImage: () => void;
}

/**
 * Image Upload Section Component
 * 
 * Displays the "Remove Image" button when an image is uploaded and editing is not done.
 * This section is hidden when:
 * - No image is uploaded (uploadedImage is null)
 * - Editing is complete (isEditingDone is true, showing comparison view)
 * 
 * User Interaction Flow:
 * 1. User uploads image → section appears
 * 2. User clicks "Remove Image" → onRemoveImage callback triggers cleanup in page.tsx
 * 3. After generation → isEditingDone becomes true → section hides
 */
export default function ImageUploadSection({
  uploadedImage,
  isEditingDone,
  onRemoveImage,
}: ImageUploadSectionProps) {
  const { t } = useLanguage();

  // Early return if section should not be visible
  // Matches original condition: !isEditingDone && uploadedImage
  if (isEditingDone || !uploadedImage) {
    return null;
  }

  return (
    <div className="pb-2">
      <h4 className="mb-2 text-text-primary font-semibold text-sm sm:text-base">
        {t("sidebar.imageUpload")}
      </h4>
      <div className="flex flex-col gap-2">
        <button
          onClick={onRemoveImage}
          className="w-full rounded bg-red-600 text-white py-3 text-sm font-medium hover:bg-red-700 transition-colors"
        >
          {t("sidebar.removeImage")}
        </button>
      </div>
    </div>
  );
}

