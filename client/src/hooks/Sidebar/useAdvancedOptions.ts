import { useMemo, useState } from "react";

interface UseAdvancedOptionsProps {
  /** Current negative prompt value */
  negativePrompt: string;
  /** Callback when negative prompt changes */
  onNegativePromptChange?: (value: string) => void;
  /** Current AI task (UI level) */
  taskType?: "white-balance" | "object-insert" | "object-removal";
  /** Callback when guidance scale changes */
  onGuidanceScaleChange?: (value: number) => void;
}

const NEGATIVE_PROMPT_DEFAULTS: Record<
  "removal" | "insertion" | "white_balance",
  { negative_prompt: string; guidance_scale: number }
> = {
  removal: {
    negative_prompt:
      "low quality, ghosting, dark stains, messy background, leftover outlines, blurry patch, replacing object",
    guidance_scale: 4.0,
  },
  insertion: {
    negative_prompt:
      "low quality, floating object, no shadow, wrong perspective, cartoon, unnatural lighting, cut and paste look",
    guidance_scale: 5.5,
  },
  white_balance: {
    negative_prompt:
      "low quality, altering geometry, changing shape, structural changes, face deformation, new elements, overexposed",
    guidance_scale: 2.5,
  },
};

const getDefaultsForTask = (
  taskType?: "white-balance" | "object-insert" | "object-removal"
) => {
  switch (taskType) {
    case "object-removal":
      return NEGATIVE_PROMPT_DEFAULTS.removal;
    case "object-insert":
      return NEGATIVE_PROMPT_DEFAULTS.insertion;
    case "white-balance":
      return NEGATIVE_PROMPT_DEFAULTS.white_balance;
    default:
      return null;
  }
};

/**
 * Hook for managing advanced options state and validation
 * 
 * Manages the negative prompt toggle state and ensures it stays
 * in sync with the actual negative prompt value.
 * 
 * State Changes:
 * - When negativePrompt gets a value → auto-enable toggle
 * - When toggle is disabled → clear negativePrompt
 * 
 * Why useEffect: Sync toggle state with negativePrompt value from parent
 * to handle external changes (e.g., reset from page.tsx)
 */
export function useAdvancedOptions({
  negativePrompt,
  onNegativePromptChange,
  taskType,
  onGuidanceScaleChange,
}: UseAdvancedOptionsProps) {
  const hasPrompt = useMemo(
    () => negativePrompt.trim().length > 0,
    [negativePrompt]
  );

  // Local toggle state; actual enabled state is derived with hasPrompt
  const [localToggle, setLocalToggle] = useState(() => hasPrompt);
  const isNegativePromptEnabled = hasPrompt || localToggle;

  /**
   * Toggle negative prompt on/off
   * When disabled, clears the negative prompt value
   */
  const handleToggleNegativePrompt = () => {
    const next = !isNegativePromptEnabled;
    setLocalToggle(next);

    if (!next) {
      // Tắt negative prompt → xoá giá trị hiện tại
      onNegativePromptChange?.("");
      return;
    }

    // Bật negative prompt:
    // - Nếu user chưa nhập gì → áp dụng default theo task (cả negative_prompt & guidance_scale)
    // - Nếu đã có prompt → giữ nguyên, không override
    if (!negativePrompt.trim()) {
      const defaults = getDefaultsForTask(taskType);
      if (defaults) {
        onNegativePromptChange?.(defaults.negative_prompt);
        onGuidanceScaleChange?.(defaults.guidance_scale);
      }
    }
  };

  return {
    isNegativePromptEnabled,
    handleToggleNegativePrompt,
  };
}

