import React, { useMemo, memo } from "react";
import {
  Undo2,
  Image as ImageIcon,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Check,
  Eye,
  EyeOff,
} from "lucide-react";
import { Select, Separator, Tooltip } from "radix-ui";
import { useLanguage } from "@/contexts/LanguageContext";
import type { InputQualityPreset } from "@/services/api";
import ImageUploadSection from "./sections/ImageUploadSection";
import ImageResolutionSection from "./sections/ImageResolutionSection";
import MaskingSection from "./sections/MaskingSection";
import AdvancedOptionsSection from "./sections/AdvancedOptionsSection";
import ResizeHandle from "./sections/ResizeHandle";

interface SidebarProps {
  isOpen: boolean;
  width?: number; // Optional width for resizable functionality
  uploadedImage: string | null;
  referenceImage: string | null;
  aiTask: "white-balance" | "object-insert" | "object-removal";
  isMaskingMode: boolean;
  maskBrushSize: number;
  maskToolType?: "brush" | "box" | "eraser";
  isMaskVisible?: boolean;
  isResizing?: boolean; // For resize handle styling
  imageDimensions?: { width: number; height: number } | null;
  inputQuality: InputQualityPreset;
  customSquareSize?: number; // Custom size for 1:1 ratio (e.g., 512, 768, 1024)
  isApplyingQuality?: boolean;
  onReferenceImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onRemoveReferenceImage: () => void;
  onEditReferenceImage?: () => void;
  onAiTaskChange: (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => void;
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
  onMaskToolTypeChange?: (type: "brush" | "box" | "eraser") => void;
  onToggleMaskVisible?: () => void;
  // Smart masking props
  enableSmartMasking?: boolean;
  isSmartMaskLoading?: boolean;
  onSmartMaskingChange?: (enabled: boolean) => void;
  smartMaskModelType?: "segmentation" | "birefnet";
  onSmartMaskModelTypeChange?: (modelType: "segmentation" | "birefnet") => void;
  borderAdjustment?: number;
  onBorderAdjustmentChange?: (value: number) => void;
  onDetectSmartMask?: () => void;
  onResizeStart?: (e: React.MouseEvent) => void; // Resize handle handler
  onWidthChange?: (width: number) => void; // For keyboard resize
  // Mask state
  hasMaskContent?: boolean; // To determine if mask has been drawn
  // New props for comparison state
  originalImage?: string | null;
  modifiedImage?: string | null;
  onReturnToOriginal?: () => void; // New prop for returning to original image
  // Advanced options props
  negativePrompt?: string;
  guidanceScale?: number;
  inferenceSteps?: number;
  seed?: number;
  onNegativePromptChange?: (value: string) => void;
  onGuidanceScaleChange?: (value: number) => void;
  onInferenceStepsChange?: (value: number) => void;
  onSeedChange?: (value: number) => void;
  enableMaeRefinement?: boolean;
  onEnableMaeRefinementChange?: (value: boolean) => void;
  onInputQualityChange: (value: InputQualityPreset) => void;
  onCustomSquareSizeChange?: (size: number) => void; // Handler for custom square size
}

function Sidebar({
  isOpen,
  width = 320, // Default width
  uploadedImage,
  referenceImage,
  aiTask,
  isMaskingMode,
  maskBrushSize,
  maskToolType = "brush",
  isMaskVisible = true,
  isResizing = false,
  imageDimensions = null,
  inputQuality,
  customSquareSize = 1024,
  isApplyingQuality = false,
  onReferenceImageUpload,
  onRemoveImage,
  onRemoveReferenceImage,
  onEditReferenceImage,
  onAiTaskChange,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onMaskToolTypeChange,
  onToggleMaskVisible,
  // Smart masking props
  enableSmartMasking = true,
  isSmartMaskLoading = false,
  onSmartMaskingChange,
  smartMaskModelType = "segmentation",
  onSmartMaskModelTypeChange,
  borderAdjustment = 0,
  onBorderAdjustmentChange,
  onDetectSmartMask,
  onResizeStart,
  onWidthChange,
  // Mask state with defaults
  hasMaskContent = false,
  // New props for comparison state
  originalImage = null,
  modifiedImage = null,
  onReturnToOriginal,
  // Advanced options with defaults
  negativePrompt = "",
  guidanceScale = 2.0, // Default 2.0
  inferenceSteps = 15, // Default 15 steps
  seed = 42, // Default seed: 42 (famous default)
  onNegativePromptChange,
  onGuidanceScaleChange,
  onInferenceStepsChange,
  onSeedChange,
  enableMaeRefinement = true,
  onEnableMaeRefinementChange,
  onInputQualityChange,
  onCustomSquareSizeChange,
}: SidebarProps) {
  // Translation hook
  const { t } = useLanguage();

  // Memoized: Determine if we're in "editing done" state (comparison mode)
  const isEditingDone = useMemo(
    () =>
      Boolean(originalImage && modifiedImage && originalImage !== modifiedImage),
    [originalImage, modifiedImage]
  );

  return (
    <div
      className={`sidebar-transition sidebar-content shrink-0 flex flex-col fixed right-0 z-30 sidebar-scrollable ${
        isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
      style={{
        backgroundColor: "var(--panel-bg)",
        width: `${width}px`,
        height: "100vh",
        transform: isOpen ? "translateX(0)" : `translateX(${width}px)`,
        transformOrigin: "right center",
        willChange: "transform, opacity",
        overflowX: "hidden",
        overflowY: "hidden",
        paddingRight: "4px",
      }}
    >
      {/* Customize Content - Scrollable content */}
      {isOpen && (
        <div
          className="flex-1 px-4 pt-2 pb-8 space-y-6 overflow-y-auto"
          style={{
            height: "100%", // Use full height instead of minHeight
            paddingBottom: "8rem", // Extra padding at bottom for better scroll experience
          }}
        >
          {/* Image Upload Section (only remove image) */}
          <ImageUploadSection
            uploadedImage={uploadedImage}
            isEditingDone={isEditingDone}
            onRemoveImage={onRemoveImage}
          />

          {/* Image Resolution Section */}
          {!isEditingDone && uploadedImage && (
            <ImageResolutionSection
              inputQuality={inputQuality}
              customSquareSize={customSquareSize}
              isApplyingQuality={isApplyingQuality}
              imageDimensions={imageDimensions}
              onInputQualityChange={onInputQualityChange}
              onCustomSquareSizeChange={onCustomSquareSizeChange}
            />
          )}

          {/* Editing Done Message */}
          {isEditingDone && (
            <div className="bg-secondary-bg border border-success rounded-lg p-4 text-center">
              <div className="flex items-center justify-center gap-2 mb-4">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "var(--success)" }} />
                <span className="text-success font-medium text-sm inline-flex items-center gap-1">
                  <Sparkles size={12} />
                  {t("sidebar.editingComplete")}
                </span>
              </div>
              <p className="text-text-secondary text-xs mb-4">
                {t("sidebar.editingCompleteDesc")}
              </p>
              <div className="space-y-2">
                <div className="flex gap-2">
                  <button
                    className="btn-interactive flex-1 inline-flex items-center justify-center gap-2 rounded border border-border-color bg-secondary-bg text-text-primary px-3 py-2 text-xs font-medium hover:bg-primary-accent hover:text-white hover:border-primary-accent transition-colors"
                    onClick={() => {
                      if (onReturnToOriginal) {
                        onReturnToOriginal();
                      }
                    }}
                  >
                    <Undo2 size={12} />
                    {t("sidebar.returnToOriginal")}
                  </button>
                </div>
                {hasMaskContent && onToggleMaskVisible && (
                  <Tooltip.Root delayDuration={300}>
                    <Tooltip.Trigger asChild>
                      <button
                        className={`btn-interactive w-full rounded p-2 text-xs font-medium transition-colors flex items-center justify-center ${
                          isMaskVisible
                            ? "bg-secondary-bg text-text-primary border border-border-color hover:bg-primary-accent hover:text-white hover:border-primary-accent"
                            : "bg-primary-accent text-white border border-primary-accent hover:bg-highlight-accent"
                        }`}
                        onClick={onToggleMaskVisible}
                      >
                        {isMaskVisible ? <EyeOff size={18} /> : <Eye size={18} />}
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
            </div>
          )}

          {/* AI Task Selection */}
          {!isEditingDone && (
            <>
              {(uploadedImage || isEditingDone) && (
                <Separator.Root className="my-2 bg-border-color h-px w-full" />
              )}
              <div className="pb-4">
                <h3 className="text-text-primary font-medium mb-3 text-sm lg:text-base">
                  {t("sidebar.aiTask")}
                </h3>
                <Select.Root
                  value={aiTask}
                  onValueChange={(value) =>
                    onAiTaskChange(
                      value as "white-balance" | "object-insert" | "object-removal"
                    )
                  }
                >
                  <Select.Trigger
                    className="w-full inline-flex items-center justify-between rounded border border-primary-accent bg-primary-bg text-text-primary px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-accent transition-colors hover:bg-secondary-bg data-[state=open]:bg-secondary-bg data-placeholder:text-text-secondary shadow-sm"
                    aria-label={t("sidebar.aiTask")}
                  >
                    <Select.Value />
                    <Select.Icon>
                      <ChevronDown size={16} />
                    </Select.Icon>
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content className="z-50 overflow-hidden rounded border border-border-color bg-primary-bg shadow-xl">
                      <Select.ScrollUpButton className="flex items-center justify-center py-1 text-text-secondary">
                        <ChevronUp size={16} />
                      </Select.ScrollUpButton>
                      <Select.Viewport className="p-1">
                        <Select.Item
                          value="white-balance"
                          className="relative flex items-center gap-2 rounded px-2 py-2 text-sm text-text-primary cursor-pointer hover:bg-secondary-bg data-highlighted:bg-secondary-bg data-[state=checked]:bg-primary-accent data-[state=checked]:text-white outline-none"
                        >
                          <Select.ItemText>{t("sidebar.whiteBalance")}</Select.ItemText>
                          <Select.ItemIndicator className="ml-auto">
                            <Check size={14} />
                          </Select.ItemIndicator>
                        </Select.Item>
                        <Select.Item
                          value="object-insert"
                          className="relative flex items-center gap-2 rounded px-2 py-2 text-sm text-text-primary cursor-pointer hover:bg-secondary-bg data-highlighted:bg-secondary-bg data-[state=checked]:bg-primary-accent data-[state=checked]:text-white outline-none"
                        >
                          <Select.ItemText>{t("sidebar.objectInsert")}</Select.ItemText>
                          <Select.ItemIndicator className="ml-auto">
                            <Check size={14} />
                          </Select.ItemIndicator>
                        </Select.Item>
                        <Select.Item
                          value="object-removal"
                          className="relative flex items-center gap-2 rounded px-2 py-2 text-sm text-text-primary cursor-pointer hover:bg-secondary-bg data-highlighted:bg-secondary-bg data-[state=checked]:bg-primary-accent data-[state=checked]:text-white outline-none"
                        >
                          <Select.ItemText>{t("sidebar.objectRemoval")}</Select.ItemText>
                          <Select.ItemIndicator className="ml-auto">
                            <Check size={14} />
                          </Select.ItemIndicator>
                        </Select.Item>
                      </Select.Viewport>
                      <Select.ScrollDownButton className="flex items-center justify-center py-1 text-text-secondary">
                        <ChevronDown size={16} />
                      </Select.ScrollDownButton>
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              </div>
            </>
          )}

          {/* Reference Image Upload (only for object insert) */}
          {!isEditingDone && aiTask === "object-insert" && (
            <>
              <Separator.Root className="my-2 bg-border-color h-px w-full" />
              <div className="pb-2">
                <h4 className="mb-2 text-text-primary font-semibold text-sm sm:text-base">
                  {t("sidebar.referenceImage")}
                </h4>
                <div className="space-y-3">
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    onChange={onReferenceImageUpload}
                    onClick={(e) => {
                      (e.target as HTMLInputElement).value = "";
                    }}
                    className="hidden"
                    id="reference-image-upload"
                  />
                  <label htmlFor="reference-image-upload" className="block">
                    <span className="flex w-full items-center justify-center gap-2 rounded border-2 border-dashed border-primary-accent bg-secondary-bg px-3 py-3 text-sm text-text-secondary hover:bg-primary-accent hover:text-white hover:border-solid transition-colors cursor-pointer">
                      <ImageIcon />
                      {t("sidebar.chooseReference")}
                    </span>
                  </label>
                  {referenceImage && (
                    <div className="space-y-2">
                      <div className="rounded-lg border-2 border-primary-accent bg-secondary-bg p-1 overflow-hidden">
                        <div className="relative w-full aspect-square bg-primary-bg rounded flex items-center justify-center">
                          
                          <img
                            src={referenceImage}
                            alt="Reference"
                            className="max-w-full max-h-full object-contain rounded"
                          />
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {onEditReferenceImage && (
                          <button
                            className="flex-1 rounded bg-primary-accent text-white px-3 py-3 text-sm font-medium hover:bg-highlight-accent transition-colors"
                            onClick={onEditReferenceImage}
                          >
                            {t("sidebar.editReference")}
                          </button>
                        )}
                        <button
                          className="flex-1 rounded bg-red-600 text-white px-3 py-3 text-sm font-medium hover:bg-red-700 transition-colors"
                          onClick={onRemoveReferenceImage}
                        >
                          {t("sidebar.removeReference")}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Masking Tool Section - Only for object insert and removal */}
          <MaskingSection
            isMaskingMode={isMaskingMode}
            maskBrushSize={maskBrushSize}
            maskToolType={maskToolType}
            isMaskVisible={isMaskVisible}
            enableSmartMasking={enableSmartMasking}
            isSmartMaskLoading={isSmartMaskLoading}
            smartMaskModelType={smartMaskModelType}
            borderAdjustment={borderAdjustment}
            hasMaskContent={hasMaskContent}
            uploadedImage={uploadedImage}
            aiTask={aiTask}
            isEditingDone={isEditingDone}
            onToggleMaskingMode={onToggleMaskingMode}
            onClearMask={onClearMask}
            onMaskBrushSizeChange={onMaskBrushSizeChange}
            onMaskToolTypeChange={onMaskToolTypeChange}
            onToggleMaskVisible={onToggleMaskVisible}
            onSmartMaskingChange={onSmartMaskingChange}
            onSmartMaskModelTypeChange={onSmartMaskModelTypeChange}
            onBorderAdjustmentChange={onBorderAdjustmentChange}
            onDetectSmartMask={onDetectSmartMask}
          />

          {/* Advanced Settings */}
          <AdvancedOptionsSection
            negativePrompt={negativePrompt}
            taskType={aiTask}
            guidanceScale={guidanceScale}
            inferenceSteps={inferenceSteps}
            seed={seed}
            isEditingDone={isEditingDone}
            enableMaeRefinement={enableMaeRefinement}
            onNegativePromptChange={onNegativePromptChange}
            onGuidanceScaleChange={onGuidanceScaleChange}
            onInferenceStepsChange={onInferenceStepsChange}
            onSeedChange={onSeedChange}
            onEnableMaeRefinementChange={onEnableMaeRefinementChange}
          />
        </div>
      )}

      {/* Resize Handle - Attached to sidebar for smooth movement */}
      <ResizeHandle
        isOpen={isOpen}
        isResizing={isResizing}
        width={width}
        onResizeStart={onResizeStart}
        onWidthChange={onWidthChange}
      />
    </div>
  );
}

// Memoize Sidebar component to prevent unnecessary re-renders
// Only re-render when props actually change
export default memo(Sidebar);
