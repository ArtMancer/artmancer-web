import React from "react";
import { 
  FaCamera, 
  FaDownload, 
  FaUpload, 
  FaImage, 
  FaTrash,
  FaPlus,
  FaBalanceScale,
  FaUndo
} from "react-icons/fa";
import { HiSparkles } from "react-icons/hi2";
import { useLanguage } from "@/contexts/LanguageContext";

interface SidebarProps {
  isOpen: boolean;
  width?: number; // Optional width for resizable functionality
  uploadedImage: string | null;
  referenceImage: string | null;
  aiTask: 'white-balance' | 'object-insert' | 'object-removal';
  isMaskingMode: boolean;
  maskBrushSize: number;
  isResizing?: boolean; // For resize handle styling
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onReferenceImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onRemoveReferenceImage: () => void;
  onAiTaskChange: (task: 'white-balance' | 'object-insert' | 'object-removal') => void;
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
  onResizeStart?: (e: React.MouseEvent) => void; // Resize handle handler
  onWidthChange?: (width: number) => void; // For keyboard resize
  // Mask history props
  maskHistoryIndex?: number;
  maskHistoryLength?: number;
  onMaskUndo?: () => void;
  onMaskRedo?: () => void;
  hasMaskContent?: boolean; // To determine if mask has been drawn
  // New props for comparison state
  originalImage?: string | null;
  modifiedImage?: string | null;
  onReturnToOriginal?: () => void; // New prop for returning to original image
  // Advanced options props
  negativePrompt?: string;
  guidanceScale?: number;
  imageWidth?: number;
  imageHeight?: number;
  inferenceSteps?: number;
  numImages?: number;
  cfgScale?: number;
  onNegativePromptChange?: (value: string) => void;
  onGuidanceScaleChange?: (value: number) => void;
  onImageSizeChange?: (width: number, height: number) => void;
  onInferenceStepsChange?: (value: number) => void;
  onNumImagesChange?: (value: number) => void;
  onCfgScaleChange?: (value: number) => void;
}

export default function Sidebar({
  isOpen,
  width = 320, // Default width
  uploadedImage,
  referenceImage,
  aiTask,
  isMaskingMode,
  maskBrushSize,
  isResizing = false,
  onImageUpload,
  onReferenceImageUpload,
  onRemoveImage,
  onRemoveReferenceImage,
  onAiTaskChange,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onResizeStart,
  onWidthChange,
  // Mask history props with defaults
  maskHistoryIndex = -1,
  maskHistoryLength = 0,
  onMaskUndo,
  onMaskRedo,
  hasMaskContent = false,
  // New props for comparison state
  originalImage = null,
  modifiedImage = null,
  onReturnToOriginal,
  // Advanced options with defaults
  negativePrompt = '',
  guidanceScale = 3.5,
  imageWidth = 1024,
  imageHeight = 1024,
  inferenceSteps = 50,
  numImages = 1,
  cfgScale = 1.0,
  onNegativePromptChange,
  onGuidanceScaleChange,
  onImageSizeChange,
  onInferenceStepsChange,
  onNumImagesChange,
  onCfgScaleChange,
}: SidebarProps) {
  // Translation hook
  const { t } = useLanguage();
  
  // Determine if we're in "editing done" state (comparison mode)
  const isEditingDone = originalImage && modifiedImage && originalImage !== modifiedImage;
  return (
    <div
      className={`bg-[var(--secondary-bg)] flex-shrink-0 flex flex-col lg:flex-col fixed right-0 z-30 sidebar-scrollable ${
        isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
      style={{
        width: `${width}px`,
        height: "100vh",
        transform: isOpen ? "translateX(0)" : `translateX(${width}px)`,
        transition:
          "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease",
        transformOrigin: "right center",
        willChange: "transform, opacity",
        overflowX: "hidden", // Prevent horizontal overflow
        overflowY: "hidden", // Remove scrolling from main container, delegate to content
        paddingRight: "4px", // Space for scrollbar
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
          {/* Image Upload Section */}
          {!isEditingDone && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t('sidebar.imageUpload')}
              </h3>
              <div className="space-y-3">
                <label
                  htmlFor="image-upload-panel"
                  className="w-full px-4 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white rounded-lg cursor-pointer transition-colors text-sm font-medium flex items-center justify-center gap-2"
                >
                  <FaCamera className="w-4 h-4" />
                  {t('sidebar.chooseImage')}
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={onImageUpload}
                  className="hidden"
                  id="image-upload-panel"
                />
                {uploadedImage && (
                  <button
                    onClick={onRemoveImage}
                    className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
                  >
                    {t('sidebar.removeImage')}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Editing Done Message */}
          {isEditingDone && (
            <div className="bg-[var(--secondary-bg)] border border-[var(--success)] rounded-lg p-4 text-center">
              <div className="flex items-center justify-center gap-2 mb-2">
                <div className="w-2 h-2 bg-[var(--success)] rounded-full"></div>
                <h3 className="text-[var(--success)] font-medium text-sm flex items-center gap-1">
                  <HiSparkles className="w-3 h-3" />
                  {t('sidebar.editingComplete')}
                </h3>
              </div>
              <p className="text-[var(--text-secondary)] text-xs mb-3">
                {t('sidebar.editingCompleteDesc')}
              </p>
              <div className="space-y-2">
                <button
                  onClick={() => {
                    if (modifiedImage) {
                      const link = document.createElement("a");
                      link.href = modifiedImage;
                      link.download = `artmancer-edited-${Date.now()}.png`;
                      document.body.appendChild(link);
                      link.click();
                      document.body.removeChild(link);
                    }
                  }}
                  className="w-full px-3 py-2 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white rounded text-xs font-medium transition-colors flex items-center justify-center gap-2"
                >
                  <FaDownload className="w-3 h-3" />
                  {t('sidebar.downloadImage')}
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => {
                      if (onReturnToOriginal) {
                        onReturnToOriginal();
                      }
                    }}
                    className="px-3 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded text-xs font-medium transition-colors flex items-center justify-center gap-1 border border-[var(--border-color)]"
                  >
                    <FaUndo className="w-3 h-3" />
                    {t('sidebar.returnToOriginal')}
                  </button>
                  <button
                    onClick={() => {
                      // Reset to original state and trigger file input
                      if (onReturnToOriginal) {
                        onReturnToOriginal();
                      }
                      // Small delay to ensure state is reset before opening file dialog
                      setTimeout(() => {
                        const fileInput = document.getElementById(
                          "image-upload-panel"
                        ) as HTMLInputElement;
                        if (fileInput) {
                          fileInput.value = ""; // Clear previous selection
                          fileInput.click();
                        }
                      }, 100);
                    }}
                    className="px-3 py-2 bg-[var(--success)] hover:bg-[var(--success)] text-white rounded text-xs font-medium transition-colors opacity-90 hover:opacity-100 flex items-center justify-center gap-1"
                  >
                    <FaCamera className="w-3 h-3" />
                    {t('sidebar.newImage')}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* AI Task Selection */}
          {!isEditingDone && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t('sidebar.aiTask')}
              </h3>
              <select
                value={aiTask}
                onChange={(e) =>
                  onAiTaskChange(
                    e.target.value as
                      | "white-balance"
                      | "object-insert"
                      | "object-removal"
                  )
                }
                className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)] focus:border-transparent"
              >
                <option value="white-balance">{t('sidebar.whiteBalance')}</option>
                <option value="object-insert">{t('sidebar.objectInsert')}</option>
                <option value="object-removal">{t('sidebar.objectRemoval')}</option>
              </select>
            </div>
          )}

          {/* Reference Image Upload (only for object insert) */}
          {!isEditingDone && aiTask === "object-insert" && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t('sidebar.referenceImage')}
              </h3>
              <div className="space-y-3">
                <label
                  htmlFor="reference-image-upload"
                  className="w-full px-4 py-3 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white rounded-lg cursor-pointer transition-colors text-sm font-medium flex items-center justify-center gap-2 border-2 border-dashed border-[var(--primary-accent)]"
                >
                  <FaImage className="w-4 h-4" />
                  {t('sidebar.chooseReference')}
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={onReferenceImageUpload}
                  className="hidden"
                  id="reference-image-upload"
                />
                {referenceImage && (
                  <div className="space-y-2">
                    <div className="relative w-full h-24 bg-[var(--secondary-bg)] rounded-lg overflow-hidden">
                      <img
                        src={referenceImage}
                        alt="Reference"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <button
                      onClick={onRemoveReferenceImage}
                      className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
                    >
                      {t('sidebar.removeReference')}
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Quality Section */}
          {!isEditingDone && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t('sidebar.quality')}
              </h3>
              <select className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base">
                <option>{t('sidebar.standard')}</option>
                <option>{t('sidebar.high')}</option>
                <option>{t('sidebar.ultra')}</option>
              </select>
            </div>
          )}

          {/* Masking Tool Section - Only for object insert and removal */}
          {!isEditingDone &&
            (aiTask === "object-insert" || aiTask === "object-removal") && (
              <div className="pb-4 border-b border-[var(--border-color)]">
                <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                  {aiTask === "object-insert"
                    ? t('sidebar.markInsertArea')
                    : t('sidebar.markRemovalArea')}
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={onToggleMaskingMode}
                      disabled={!uploadedImage}
                      className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                        isMaskingMode
                          ? "bg-[var(--highlight-accent)] text-white"
                          : "bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white disabled:bg-gray-600 disabled:cursor-not-allowed"
                      }`}
                    >
                      {isMaskingMode ? t('sidebar.exitMasking') : t('sidebar.startMasking')}
                    </button>
                    {isMaskingMode && (
                      <button
                        onClick={onClearMask}
                        className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors"
                      >
                        {t('sidebar.clearMask')}
                      </button>
                    )}
                  </div>

                  {/* Mask History Controls - Show when there's actionable history */}
                  {isMaskingMode && maskHistoryLength > 0 && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={onMaskUndo}
                        disabled={!onMaskUndo || maskHistoryIndex <= 0}
                        className="px-3 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        title={t('sidebar.undoMaskStroke')}
                      >
                        ↶ {t('sidebar.undo')}
                      </button>
                      <button
                        onClick={onMaskRedo}
                        disabled={
                          !onMaskRedo ||
                          maskHistoryIndex >= maskHistoryLength - 1
                        }
                        className="px-3 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        title={t('sidebar.redoMaskStroke')}
                      >
                        ↷ {t('sidebar.redo')}
                      </button>
                    </div>
                  )}

                  {isMaskingMode && (
                    <div className="space-y-3">
                      <div>
                        <label className="block text-[var(--text-secondary)] text-sm mb-2">
                          {t('sidebar.brushSize')}
                        </label>
                        <div className="flex items-center gap-2">
                          <input
                            type="number"
                            min="1"
                            max="50"
                            value={maskBrushSize}
                            onChange={(e) => {
                              const value = parseInt(e.target.value);
                              if (!isNaN(value) && value >= 1 && value <= 50) {
                                onMaskBrushSizeChange(value);
                              }
                            }}
                            onBlur={(e) => {
                              const value = parseInt(e.target.value);
                              if (isNaN(value) || value < 1) {
                                onMaskBrushSizeChange(1);
                              } else if (value > 50) {
                                onMaskBrushSizeChange(50);
                              }
                            }}
                            className="w-16 px-2 py-1 text-sm bg-[var(--secondary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                          />
                          <span className="text-xs text-[var(--text-secondary)] mr-1">
                            px
                          </span>
                          <input
                            type="range"
                            min="1"
                            max="50"
                            value={maskBrushSize}
                            onChange={(e) =>
                              onMaskBrushSizeChange(parseInt(e.target.value))
                            }
                            className="flex-1 accent-[var(--primary-accent)]"
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-[var(--text-secondary)]">
                    {!uploadedImage
                      ? aiTask === "object-insert"
                        ? t('sidebar.uploadFirstObjectInsert')
                        : t('sidebar.uploadFirstObjectRemove')
                      : isMaskingMode
                      ? aiTask === "object-insert"
                        ? t('sidebar.drawForInsertion')
                        : t('sidebar.drawForRemoval')
                      : aiTask === "object-insert"
                      ? t('sidebar.enableMaskingInsertion')
                      : t('sidebar.enableMaskingRemoval')}
                  </p>
                </div>
              </div>
            )}

          {/* White Balance Settings (only for white balance task) */}
          {!isEditingDone && aiTask === "white-balance" && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t('sidebar.whiteBalanceSettings')}
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t('sidebar.autoCorrectionStrength')}: 80%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    defaultValue="80"
                    className="w-full accent-[var(--primary-accent)]"
                  />
                </div>
                <p className="text-xs text-[var(--text-secondary)]">
                  {t('sidebar.aiAutoAdjust')}
                </p>
              </div>
            </div>
          )}

          {/* Advanced Settings */}
          {!isEditingDone && (
            <div className="pt-2">
              <h3 className="text-[var(--text-primary)] font-medium mb-4">
                {t('sidebar.advanced')}
              </h3>
              <div className="space-y-6">
                {/* Negative Prompt */}
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t('sidebar.negativePrompt')}
                  </label>
                  <textarea
                    placeholder={t('sidebar.negativePromptPlaceholder')}
                    rows={2}
                    value={negativePrompt}
                    onChange={(e) => onNegativePromptChange?.(e.target.value)}
                    className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)] focus:border-transparent"
                  />
                </div>

                {/* Guidance Scale */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t('sidebar.guidanceScale')}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      step="0.5"
                      value={guidanceScale}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value >= 1 && value <= 10) {
                          onGuidanceScaleChange?.(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseFloat(e.target.value);
                        if (isNaN(value) || value < 1) {
                          onGuidanceScaleChange?.(1);
                        } else if (value > 10) {
                          onGuidanceScaleChange?.(10);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                    />
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="0.5"
                    value={guidanceScale}
                    onChange={(e) => onGuidanceScaleChange?.(parseFloat(e.target.value))}
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                    <span>{t('sidebar.moreCreative')}</span>
                    <span>{t('sidebar.followPromptStrictly')}</span>
                  </div>
                </div>

                {/* Image Dimensions */}
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-3">
                    {t('sidebar.imageSize')}
                  </label>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-[var(--text-secondary)] mb-2 block">
                        {t('sidebar.width')}
                      </label>
                      <select 
                        value={imageWidth}
                        onChange={(e) => onImageSizeChange?.(parseInt(e.target.value), imageHeight)}
                        className="w-full px-2 py-1 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)]"
                      >
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024">1024px</option>
                        <option value="1536">1536px</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-[var(--text-secondary)] mb-2 block">
                        {t('sidebar.height')}
                      </label>
                      <select 
                        value={imageHeight}
                        onChange={(e) => onImageSizeChange?.(imageWidth, parseInt(e.target.value))}
                        className="w-full px-2 py-1 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)]"
                      >
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024">1024px</option>
                        <option value="1536">1536px</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Steps (Updated from existing) */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t('sidebar.inferenceSteps')}
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="100"
                      value={inferenceSteps}
                      onChange={(e) => {
                        const value = parseInt(e.target.value);
                        if (!isNaN(value) && value >= 10 && value <= 100) {
                          onInferenceStepsChange?.(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value);
                        if (isNaN(value) || value < 10) {
                          onInferenceStepsChange?.(10);
                        } else if (value > 100) {
                          onInferenceStepsChange?.(100);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                    />
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={inferenceSteps}
                    onChange={(e) => onInferenceStepsChange?.(parseInt(e.target.value))}
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                    <span>{t('sidebar.faster')}</span>
                    <span>{t('sidebar.higherQuality')}</span>
                  </div>
                </div>

                {/* Number of Images */}
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t('sidebar.numberOfImages')}
                  </label>
                  <select 
                    value={numImages}
                    onChange={(e) => onNumImagesChange?.(parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)]"
                  >
                    <option value="1">{t('sidebar.oneImage')}</option>
                    <option value="2">{t('sidebar.twoImages')}</option>
                    <option value="3">{t('sidebar.threeImages')}</option>
                    <option value="4">{t('sidebar.fourImages')}</option>
                  </select>
                </div>

                {/* True CFG Scale */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t('sidebar.cfgScale')}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="5"
                      step="0.1"
                      value={cfgScale.toFixed(1)}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value >= 1 && value <= 5) {
                          onCfgScaleChange?.(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseFloat(e.target.value);
                        if (isNaN(value) || value < 1) {
                          onCfgScaleChange?.(1);
                        } else if (value > 5) {
                          onCfgScaleChange?.(5);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                    />
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="0.1"
                    value={cfgScale}
                    onChange={(e) => onCfgScaleChange?.(parseFloat(e.target.value))}
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <p className="text-xs text-[var(--text-secondary)] mt-2">
                    {t('sidebar.cfgGuidanceDescription')}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Resize Handle - Attached to sidebar for smooth movement */}
      {isOpen && onResizeStart && (
        <div
          className={`absolute top-0 left-0 w-3 h-full cursor-col-resize z-10 transition-all duration-150 ${
            isResizing
              ? "bg-[var(--primary-accent)] opacity-100"
              : "bg-transparent hover:bg-[var(--primary-accent)] hover:opacity-60"
          }`}
          style={{
            transform: "translateX(-50%)", // Center on the left edge
            willChange: isResizing ? "background-color" : "auto",
          }}
          onMouseDown={onResizeStart}
          onTouchStart={(e) => {
            // Touch support for mobile
            const touch = e.touches[0];
            if (touch) {
              onResizeStart(e as any);
            }
          }}
          role="separator"
          aria-label="Resize sidebar"
          tabIndex={0}
          onKeyDown={(e) => {
            // Keyboard accessibility
            if (e.key === "ArrowLeft" && onWidthChange) {
              e.preventDefault();
              onWidthChange(Math.max(280, width - 10));
              // Save to localStorage
              localStorage.setItem(
                "sidebarWidth",
                String(Math.max(280, width - 10))
              );
            } else if (e.key === "ArrowRight" && onWidthChange) {
              e.preventDefault();
              onWidthChange(Math.min(600, width + 10));
              // Save to localStorage
              localStorage.setItem(
                "sidebarWidth",
                String(Math.min(600, width + 10))
              );
            }
          }}
        >
        </div>
      )}
    </div>
  );
}
