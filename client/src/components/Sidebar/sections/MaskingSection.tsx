"use client";

import React from "react";
import { Slider, Checkbox, Separator, Tooltip } from "radix-ui";
import { Paintbrush, Square, Eraser, Check, X, Trash2, Eye, EyeOff, Wand2 } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

interface MaskingSectionProps {
  /** Whether masking mode is active */
  isMaskingMode: boolean;
  /** Current brush size (1-100) */
  maskBrushSize: number;
  /** Current tool type (brush or box) */
  maskToolType?: "brush" | "box" | "eraser";
  /** Mask visibility state from parent */
  isMaskVisible?: boolean;
  /** Whether smart masking is enabled */
  enableSmartMasking?: boolean;
  /** Whether smart mask is currently loading */
  isSmartMaskLoading?: boolean;
  /** Smart mask model type */
  smartMaskModelType?: "segmentation" | "birefnet";
  /** Border adjustment value (-10 to 10) */
  borderAdjustment?: number;
  /** Whether mask has content */
  hasMaskContent?: boolean;
  /** Whether image is uploaded */
  uploadedImage: string | null;
  /** Current AI task type */
  aiTask: "white-balance" | "object-insert" | "object-removal";
  /** Whether editing is done (comparison mode) */
  isEditingDone: boolean;

  // Callbacks from parent
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
  onMaskToolTypeChange?: (type: "brush" | "box" | "eraser") => void;
  onToggleMaskVisible?: () => void;
  onSmartMaskingChange?: (enabled: boolean) => void;
  onSmartMaskModelTypeChange?: (modelType: "segmentation" | "birefnet") => void;
  onBorderAdjustmentChange?: (value: number) => void;
  onDetectSmartMask?: () => void;
}

/**
 * Masking Section Component
 * 
 * Provides controls for mask creation and editing:
 * - Toggle masking mode on/off
 * - Clear mask
 * - Toggle mask visibility
 * - Undo/Redo mask strokes
 * - Smart masking settings (FastSAM/BiRefNet)
 * - Tool selection (brush/box)
 * - Brush size control
 * - Border adjustment for smart masks
 * 
 * User Interaction Flow:
 * 1. User clicks "Start Masking" → enters masking mode → can draw on canvas
 * 2. User draws mask → mask history updated → undo/redo available
 * 3. User enables smart masking → selects model (FastSAM/BiRefNet) → draws stroke/box → AI generates mask
 * 4. User adjusts border → smart mask border expands/shrinks
 * 5. User clicks "Exit Masking" → exits mode, mask preserved
 * 
 * State Changes:
 * - isMaskingMode toggle → enables/disables canvas drawing
 * - maskToolType change → switches between brush and box tools
 * - enableSmartMasking toggle → enables AI-assisted mask generation
 * - smartMaskModelType change → switches between FastSAM and BiRefNet (auto-switches tool if needed)
 */
export default function MaskingSection({
  isMaskingMode,
  maskBrushSize,
  maskToolType = "brush",
  enableSmartMasking = true,
  isSmartMaskLoading = false,
  smartMaskModelType = "segmentation",
  borderAdjustment = 0,
  hasMaskContent = false,
  uploadedImage,
  aiTask,
  isEditingDone,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onMaskToolTypeChange,
  onToggleMaskVisible,
  onSmartMaskingChange,
  onSmartMaskModelTypeChange,
  onBorderAdjustmentChange,
  onDetectSmartMask,
  isMaskVisible = true,
}: MaskingSectionProps) {
  const { t } = useLanguage();

  // Early return if section should not be visible
  // Only show for object-insert and object-removal tasks
  if (
    isEditingDone ||
    (aiTask !== "object-insert" && aiTask !== "object-removal")
  ) {
    return null;
  }

  return (
    <>
      {/* Divider trước vùng điều khiển masking cho các task insert/removal */}
      <Separator.Root className="my-2 bg-border-color h-px w-full" />
      <div className="pb-4">
        <h3 className="text-text-primary font-medium mb-3 text-sm lg:text-base">
          {aiTask === "object-insert"
            ? t("sidebar.markInsertArea")
            : t("sidebar.markRemovalArea")}
        </h3>
        <div className="space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <Tooltip.Root delayDuration={300}>
              <Tooltip.Trigger asChild>
                <button
                  onClick={onToggleMaskingMode}
                  disabled={!uploadedImage}
                  className={`btn-interactive btn-primary-hover px-3 py-2 rounded text-sm font-medium transition-colors inline-flex items-center gap-1.5 sm:gap-2 shrink-0 ${
                    isMaskingMode
                      ? "bg-highlight-accent text-white"
                      : "bg-primary-accent hover:bg-highlight-accent text-white disabled:bg-secondary-bg disabled:text-text-secondary disabled:cursor-not-allowed"
                  }`}
                >
                  {isMaskingMode ? <X size={16} className="sm:w-[18px] sm:h-[18px]" /> : <Paintbrush size={16} className="sm:w-[18px] sm:h-[18px]" />}
                  <span className="text-xs sm:text-sm whitespace-nowrap">
                    {isMaskingMode ? t("sidebar.exitMasking") : t("sidebar.startMasking")}
                  </span>
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="bg-secondary-bg border border-border-color text-text-primary px-2 py-1 rounded text-xs shadow-lg" sideOffset={5}>
                  {isMaskingMode
                    ? t("sidebar.exitMasking")
                    : t("sidebar.startMasking")}
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
            {isMaskingMode && enableSmartMasking && onDetectSmartMask && (
              <Tooltip.Root delayDuration={300}>
                <Tooltip.Trigger asChild>
                  <button
                    onClick={onDetectSmartMask}
                    disabled={isSmartMaskLoading || !hasMaskContent}
                    className={`btn-interactive px-2.5 sm:px-3 py-2 rounded text-sm font-medium transition-colors inline-flex items-center gap-1.5 sm:gap-2 shrink-0 ${
                      isSmartMaskLoading || !hasMaskContent
                        ? "bg-secondary-bg text-text-secondary cursor-not-allowed opacity-60"
                        : "bg-primary-accent hover:bg-highlight-accent text-white"
                    }`}
                  >
                    <Wand2 size={16} className="sm:w-[18px] sm:h-[18px]" />
                    <span className="text-xs sm:text-sm whitespace-nowrap">Detect</span>
                  </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content className="bg-secondary-bg border border-border-color text-text-primary px-2 py-1 rounded text-xs shadow-lg" sideOffset={5}>
                    {!hasMaskContent
                      ? "Draw a stroke or box first"
                      : isSmartMaskLoading
                      ? "Detecting..."
                      : "Detect object from mask"}
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            )}
            {isMaskingMode && (
              <Tooltip.Root delayDuration={300}>
                <Tooltip.Trigger asChild>
                  <button
                    onClick={onClearMask}
                    className="btn-interactive p-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors shrink-0"
                  >
                    <Trash2 size={16} className="sm:w-[18px] sm:h-[18px]" />
                  </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content className="bg-secondary-bg border border-border-color text-text-primary px-2 py-1 rounded text-xs shadow-lg" sideOffset={5}>
                    {t("sidebar.clearMask")}
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            )}
            {(hasMaskContent || isMaskingMode) && (
              <Tooltip.Root delayDuration={300}>
                <Tooltip.Trigger asChild>
                  <button
                    onClick={onToggleMaskVisible}
                    disabled={!onToggleMaskVisible}
                    className={`btn-interactive p-2 rounded text-sm font-medium transition-colors shrink-0 ${
                      isMaskVisible
                        ? "bg-secondary-bg hover:bg-primary-accent text-text-primary hover:text-white"
                        : "bg-primary-accent hover:bg-highlight-accent text-white"
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isMaskVisible ? <EyeOff size={16} className="sm:w-[18px] sm:h-[18px]" /> : <Eye size={16} className="sm:w-[18px] sm:h-[18px]" />}
                  </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content className="bg-secondary-bg border border-border-color text-text-primary px-2 py-1 rounded text-xs shadow-lg" sideOffset={5}>
                    {isMaskVisible ? "Hide mask" : "Show mask"}
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            )}
          </div>

          {/* Mask History Controls removed (undo/redo not used) */}

          {isMaskingMode && (
            <div className="space-y-3">
              {/* Smart Masking Settings */}
              <div className="space-y-2">
                <label className="block text-text-secondary text-sm mb-2">
                  {t("sidebar.maskSettings")}
                </label>
                {onSmartMaskingChange && (
                  <label className="flex items-center gap-2 text-text-primary text-sm">
                    <Checkbox.Root
                      checked={enableSmartMasking}
                      onCheckedChange={(checked: boolean | "indeterminate") =>
                        onSmartMaskingChange(!!checked)
                      }
                      disabled={isSmartMaskLoading}
                      className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
                        enableSmartMasking
                          ? "bg-primary-accent border-primary-accent"
                          : "bg-transparent border-border-color"
                      } ${isSmartMaskLoading ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      <Checkbox.Indicator>
                        <Check size={12} strokeWidth={3} className="text-white" />
                      </Checkbox.Indicator>
                    </Checkbox.Root>
                    <span className="flex items-center gap-2">
                      {t("sidebar.smartMasking")}
                      {isSmartMaskLoading && (
                        <span className="inline-block w-4 h-4 border-2 border-primary-accent border-t-transparent rounded-full animate-spin" />
                      )}
                    </span>
                  </label>
                )}

                {/* Model Type Selection */}
                {enableSmartMasking && onSmartMaskModelTypeChange && (
                  <div className="mt-3 space-y-2">
                    <label className="text-text-secondary text-sm block">
                      {t("sidebar.smartMaskModel")}
                    </label>
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          onSmartMaskModelTypeChange("segmentation");
                        }}
                    className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                          smartMaskModelType === "segmentation"
                            ? "bg-primary-accent text-white"
                        : "bg-secondary-bg text-text-primary hover:bg-primary-bg"
                        }`}
                        disabled={isSmartMaskLoading}
                      >
                        FastSAM
                      </button>
                      <button
                        onClick={() => {
                          onSmartMaskModelTypeChange("birefnet");
                        }}
                        className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                          smartMaskModelType === "birefnet"
                            ? "bg-primary-accent text-white"
                            : "bg-secondary-bg text-text-primary hover:bg-primary-bg"
                        }`}
                        disabled={isSmartMaskLoading}
                      >
                        BiRefNet
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Border Adjustment Control */}
              {enableSmartMasking && onBorderAdjustmentChange && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-text-secondary text-sm">
                      {t("sidebar.borderAdjustment")}
                    </label>
                    <input
                      type="number"
                      min={-10}
                      max={10}
                      value={borderAdjustment}
                      onChange={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (!Number.isNaN(value)) {
                          onBorderAdjustmentChange(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (Number.isNaN(value) || value < -10) {
                          onBorderAdjustmentChange(-10);
                        } else if (value > 10) {
                          onBorderAdjustmentChange(10);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-primary-bg border border-border-color rounded text-text-primary text-center focus:outline-none focus:border-primary-accent"
                    />
                  </div>
                  <div className="px-2">
                  <Slider.Root
                      min={-10}
                      max={10}
                      step={1}
                      value={[borderAdjustment]}
                      onValueChange={([value]: number[]) =>
                        onBorderAdjustmentChange(value)
                      }
                      className="relative flex h-5 w-full touch-none select-none items-center"
                    >
                      <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                        <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
                      </Slider.Track>
                      <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
                    </Slider.Root>
                    <div className="flex justify-between items-center text-xs text-text-secondary mt-2 px-1">
                      <span>{t("sidebar.shrink")}</span>
                      <span>{t("sidebar.grow")}</span>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <label className="flex items-center gap-2 text-text-secondary text-sm mb-2">
                  <span>{t("sidebar.maskTool")}</span>
                  <span className="flex items-center gap-1 text-xs text-text-secondary">
                    <Paintbrush size={14} />
                    <Eraser size={14} />
                    <Square size={14} />
                  </span>
                </label>
                <div className="flex gap-2 mb-3">
                  <button
                    onClick={() => onMaskToolTypeChange?.("brush")}
                    className={`tool-item flex-1 px-3 py-2 text-sm rounded border transition-all duration-150 ${
                      maskToolType === "brush"
                        ? "tool-item-active bg-primary-accent text-white border-primary-accent"
                        : "bg-secondary-bg text-text-primary border-border-color hover:bg-secondary-bg"
                    }`}
                  >
                    <span className="inline-flex items-center gap-1">
                      <Paintbrush size={14} className={maskToolType === "brush" ? "tool-icon-rotate" : ""} data-active={maskToolType === "brush"} />
                      <span>{t("sidebar.brush")}</span>
                    </span>
                  </button>
                  <button
                    onClick={() => onMaskToolTypeChange?.("box")}
                    className={`tool-item flex-1 px-3 py-2 text-sm rounded border transition-all duration-150 ${
                      maskToolType === "box"
                        ? "tool-item-active bg-primary-accent text-white border-primary-accent"
                        : "bg-secondary-bg text-text-primary border-border-color hover:bg-secondary-bg"
                    }`}
                  >
                    <span className="inline-flex items-center gap-1">
                      <Square size={14} />
                      <span>{t("sidebar.box")}</span>
                    </span>
                  </button>
                  <button
                    onClick={() => onMaskToolTypeChange?.("eraser")}
                    className={`tool-item flex-1 px-3 py-2 text-sm rounded border transition-all duration-150 ${
                      maskToolType === "eraser"
                        ? "tool-item-active bg-primary-accent text-white border-primary-accent"
                        : "bg-secondary-bg text-text-primary border-border-color hover:bg-secondary-bg"
                    }`}
                  >
                    <span className="inline-flex items-center gap-1">
                      <Eraser size={14} className={maskToolType === "eraser" ? "tool-icon-rotate" : ""} data-active={maskToolType === "eraser"} />
                      <span>{t("sidebar.eraser")}</span>
                    </span>
                  </button>
                </div>
              </div>

              {maskToolType === "brush" && (
                <div>
                  <label className="block text-text-secondary text-sm mb-2">
                    {t("sidebar.brushSize")}
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={maskBrushSize}
                      onChange={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (!Number.isNaN(value)) {
                          onMaskBrushSizeChange(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (Number.isNaN(value) || value < 1) {
                          onMaskBrushSizeChange(1);
                        } else if (value > 100) {
                          onMaskBrushSizeChange(100);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-secondary-bg border border-border-color rounded text-text-primary text-center focus:outline-none focus:border-primary-accent"
                    />
                    <span className="text-xs text-text-secondary mr-1">px</span>
                    <div className="flex-1 px-2">
                      <Slider.Root
                        min={1}
                        max={100}
                        step={1}
                        value={[maskBrushSize]}
                        onValueChange={([value]: number[]) =>
                          onMaskBrushSizeChange(value)
                        }
                        className="relative flex h-5 w-full touch-none select-none items-center"
                      >
                        <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                          <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
                        </Slider.Track>
                        <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
                      </Slider.Root>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          <p className="text-xs text-text-secondary">
            {!uploadedImage
              ? aiTask === "object-insert"
                ? t("sidebar.uploadFirstObjectInsert")
                : t("sidebar.uploadFirstObjectRemove")
              : isMaskingMode
              ? aiTask === "object-insert"
                ? t("sidebar.drawForInsertion")
                : t("sidebar.drawForRemoval")
              : aiTask === "object-insert"
              ? t("sidebar.enableMaskingInsertion")
              : t("sidebar.enableMaskingRemoval")}
          </p>
        </div>
      </div>
    </>
  );
}

