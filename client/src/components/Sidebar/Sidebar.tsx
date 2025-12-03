import React from "react";
import {
  FaCamera,
  FaDownload,
  FaUpload,
  FaImage,
  FaImages,
  FaTrash,
  FaPlus,
  FaBalanceScale,
  FaUndo,
  FaQuestionCircle,
} from "react-icons/fa";
import { HiSparkles } from "react-icons/hi2";
import { useLanguage } from "@/contexts/LanguageContext";
import {
  Button,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  Checkbox,
  FormControlLabel,
  Box,
  Typography,
  Chip,
} from "@mui/material";
import {
  CloudUpload as CloudUploadIcon,
  Image as ImageIcon,
  CheckCircle as CheckCircleIcon,
} from "@mui/icons-material";
import type { InputQualityPreset } from "@/services/api";

interface SidebarProps {
  isOpen: boolean;
  width?: number; // Optional width for resizable functionality
  uploadedImage: string | null;
  referenceImage: string | null;
  aiTask: "white-balance" | "object-insert" | "object-removal" | "evaluation";
  appMode?: "inference" | "benchmark"; // App mode: inference or benchmark
  isMaskingMode: boolean;
  maskBrushSize: number;
  maskToolType?: "brush" | "box";
  isResizing?: boolean; // For resize handle styling
  imageDimensions?: { width: number; height: number } | null;
  inputQuality: InputQualityPreset;
  isApplyingQuality?: boolean;
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onReferenceImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onRemoveReferenceImage: () => void;
  onAiTaskChange: (
    task: "white-balance" | "object-insert" | "object-removal" | "evaluation"
  ) => void;
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
  onMaskToolTypeChange?: (type: "brush" | "box") => void;
  // Smart masking props
  enableSmartMasking?: boolean;
  isSmartMaskLoading?: boolean;
  onSmartMaskingChange?: (enabled: boolean) => void;
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
  lastRequestId?: string | null; // Request ID for visualization download
  // Advanced options props
  negativePrompt?: string;
  guidanceScale?: number;
  inferenceSteps?: number;
  cfgScale?: number;
  onNegativePromptChange?: (value: string) => void;
  onGuidanceScaleChange?: (value: number) => void;
  onInferenceStepsChange?: (value: number) => void;
  onCfgScaleChange?: (value: number) => void;
  // White balance props
  onWhiteBalance?: (
    method: "auto" | "manual" | "ai",
    temperature?: number,
    tint?: number
  ) => void;
  whiteBalanceTemperature?: number;
  whiteBalanceTint?: number;
  onWhiteBalanceTemperatureChange?: (value: number) => void;
  onWhiteBalanceTintChange?: (value: number) => void;
  // Evaluation mode props
  evaluationMode?: "single" | "multiple";
  evaluationTask?: "white-balance" | "object-insert" | "object-removal";
  evaluationSingleOriginal?: string | null;
  evaluationSingleTarget?: string | null;
  evaluationImagePairs?: Array<{
    original: string | null;
    target: string | null;
    filename: string;
  }>;
  evaluationConditionalImages?: string[];
  evaluationReferenceImage?: string | null;
  evaluationDisplayLimit?: number;
  allowMultipleFolders?: boolean;
  onEvaluationModeChange?: (mode: "single" | "multiple") => void;
  onEvaluationTaskChange?: (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => void;
  onEvaluationSingleOriginalUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationSingleTargetUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationMultipleUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationOriginalFolderUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationTargetFolderUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationConditionalImagesUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onEvaluationReferenceImageUpload?: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  onAllowMultipleFoldersChange?: (value: boolean) => void;
  onEvaluationDisplayLimitChange?: (value: number) => void;
  evaluationResults?: any[];
  evaluationResponse?: any;
  onExportEvaluationJSON?: () => void;
  onExportEvaluationCSV?: () => void;
  onInputQualityChange: (value: InputQualityPreset) => void;
  // Low-end optimization props
  enable4BitTextEncoder?: boolean;
  enableCpuOffload?: boolean;
  enableMemoryOptimizations?: boolean;
  enableFlowmatchScheduler?: boolean;
  onEnable4BitTextEncoderChange?: (enabled: boolean) => void;
  onEnableCpuOffloadChange?: (enabled: boolean) => void;
  onEnableMemoryOptimizationsChange?: (enabled: boolean) => void;
  onEnableFlowmatchSchedulerChange?: (enabled: boolean) => void;
  // Benchmark mode props
  benchmarkFolder?: string;
  benchmarkValidation?: {
    success: boolean;
    message: string;
    image_count?: number;
    details?: Record<string, number>;
  } | null;
  isValidatingBenchmark?: boolean;
  benchmarkSampleCount?: number;
  benchmarkTask?: "white-balance" | "object-insert" | "object-removal";
  isRunningBenchmark?: boolean;
  benchmarkProgress?: {
    current: number;
    total: number;
    currentImage?: string;
  } | null;
  benchmarkResults?: any;
  benchmarkPrompt?: string;
  onBenchmarkFolderChange?: (path: string) => void;
  onBenchmarkFolderValidate?: (file: File | File[]) => void;
  onBenchmarkSampleCountChange?: (count: number) => void;
  onBenchmarkTaskChange?: (
    task: "white-balance" | "object-insert" | "object-removal"
  ) => void;
  onBenchmarkPromptChange?: (prompt: string) => void;
  onRunBenchmark?: () => void;
}

export default function Sidebar({
  isOpen,
  width = 320, // Default width
  uploadedImage,
  referenceImage,
  aiTask,
  appMode = "inference", // Default to inference mode
  isMaskingMode,
  maskBrushSize,
  maskToolType = "brush",
  isResizing = false,
  imageDimensions = null,
  inputQuality,
  isApplyingQuality = false,
  onImageUpload,
  onReferenceImageUpload,
  onRemoveImage,
  onRemoveReferenceImage,
  onAiTaskChange,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onMaskToolTypeChange,
  // Smart masking props
  enableSmartMasking = true,
  isSmartMaskLoading = false,
  onSmartMaskingChange,
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
  lastRequestId = null,
  // Advanced options with defaults
  negativePrompt = "",
  guidanceScale = 1.0,
  inferenceSteps = 40,
  cfgScale = 4.0,
  onNegativePromptChange,
  onGuidanceScaleChange,
  onInferenceStepsChange,
  onCfgScaleChange,
  // White balance props with defaults
  onWhiteBalance,
  whiteBalanceTemperature = 0,
  whiteBalanceTint = 0,
  onWhiteBalanceTemperatureChange,
  onWhiteBalanceTintChange,
  // Evaluation mode props with defaults
  evaluationMode = "single",
  evaluationTask = "white-balance",
  evaluationSingleOriginal = null,
  evaluationSingleTarget = null,
  evaluationImagePairs = [],
  evaluationConditionalImages = [],
  evaluationReferenceImage = null,
  evaluationDisplayLimit = 10,
  allowMultipleFolders = false,
  onEvaluationModeChange,
  onEvaluationTaskChange,
  onEvaluationSingleOriginalUpload,
  onEvaluationSingleTargetUpload,
  onEvaluationMultipleUpload,
  onEvaluationOriginalFolderUpload,
  onEvaluationTargetFolderUpload,
  onEvaluationConditionalImagesUpload,
  onEvaluationReferenceImageUpload,
  onAllowMultipleFoldersChange,
  onEvaluationDisplayLimitChange,
  evaluationResults = [],
  evaluationResponse = null,
  onExportEvaluationJSON,
  onExportEvaluationCSV,
  onInputQualityChange,
  // Low-end optimization props with defaults
  enable4BitTextEncoder = false,
  enableCpuOffload = false,
  enableMemoryOptimizations = false,
  enableFlowmatchScheduler = false,
  onEnable4BitTextEncoderChange,
  onEnableCpuOffloadChange,
  onEnableMemoryOptimizationsChange,
  onEnableFlowmatchSchedulerChange,
}: SidebarProps) {
  // Translation hook
  const { t } = useLanguage();

  // Determine if we're in "editing done" state (comparison mode)
  const isEditingDone =
    originalImage && modifiedImage && originalImage !== modifiedImage;

  const inputQualityOptions: {
    value: InputQualityPreset;
    label: string;
    ratio: string;
    description: string;
  }[] = [
    {
      value: "resized",
      label: "Resize 1:1",
      ratio: "1:1",
      description:
        "Resize về aspect ratio vuông (512x512, 1024x1024,...), tối ưu hiệu năng.",
    },
    {
      value: "original",
      label: "Ảnh gốc",
      ratio: "Gốc",
      description: "Giữ nguyên kích thước và tỷ lệ gốc, chi tiết cao nhất.",
    },
  ];

  const selectedQuality = inputQualityOptions.find(
    (option) => option.value === inputQuality
  );

  const showOriginalWarning =
    inputQuality === "original" &&
    imageDimensions &&
    Math.max(imageDimensions.width, imageDimensions.height) >= 2048;
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
                {t("sidebar.imageUpload")}
              </h3>
              <div className="space-y-3">
                <label
                  htmlFor="image-upload-panel"
                  className="w-full px-4 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white rounded-lg cursor-pointer transition-colors text-sm font-medium flex items-center justify-center gap-2"
                >
                  <FaCamera className="w-4 h-4" />
                  {t("sidebar.chooseImage")}
                </label>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  onChange={onImageUpload}
                  className="hidden"
                  id="image-upload-panel"
                />
                {uploadedImage && (
                  <button
                    onClick={onRemoveImage}
                    className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
                  >
                    {t("sidebar.removeImage")}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Input Quality Section */}
          {!isEditingDone && uploadedImage && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[var(--text-primary)] font-medium text-sm lg:text-base">
                  Chất lượng xử lý
                </h3>
                <button
                  type="button"
                  className="text-[var(--text-secondary)] hover:text-[var(--primary-accent)] transition-colors"
                  title="Giảm kích thước ảnh đầu vào để tiết kiệm VRAM. Mức thấp hơn giúp chạy nhanh hơn nhưng ít chi tiết hơn."
                >
                  <FaQuestionCircle className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {inputQualityOptions.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => onInputQualityChange(option.value)}
                      disabled={isApplyingQuality}
                      className={`px-3 py-2 rounded text-sm border transition-colors text-left ${
                        inputQuality === option.value
                          ? "bg-[var(--primary-accent)] text-white border-[var(--primary-accent)]"
                          : "bg-[var(--secondary-bg)] text-[var(--text-primary)] border-[var(--border-color)] hover:bg-[var(--hover-bg)]"
                      } ${
                        isApplyingQuality ? "opacity-60 cursor-not-allowed" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between font-medium">
                        <span>{option.label}</span>
                        <span className="text-xs opacity-80">
                          {option.ratio}
                        </span>
                      </div>
                      <p className="text-[10px] mt-1 opacity-80">
                        {option.description}
                      </p>
                    </button>
                  ))}
                </div>
                {selectedQuality && (
                  <p className="text-xs text-[var(--text-secondary)]">
                    Đang chọn: {selectedQuality.label} ({selectedQuality.ratio}{" "}
                    kích thước gốc)
                  </p>
                )}
                {isApplyingQuality && (
                  <p className="text-xs text-[var(--text-secondary)]">
                    Đang áp dụng chất lượng mới, vui lòng chờ...
                  </p>
                )}
                {showOriginalWarning && (
                  <p className="text-xs text-yellow-400">
                    Ảnh lớn ({imageDimensions?.width}×{imageDimensions?.height}
                    ). Mức Original có thể tốn rất nhiều VRAM, cân nhắc chọn
                    High hoặc Medium nếu gặp lỗi OOM.
                  </p>
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
                  {t("sidebar.editingComplete")}
                </h3>
              </div>
              <p className="text-[var(--text-secondary)] text-xs mb-3">
                {t("sidebar.editingCompleteDesc")}
              </p>
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
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
                    className="px-3 py-2 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white rounded text-xs font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <FaDownload className="w-3 h-3" />
                    {t("sidebar.downloadImage")}
                  </button>
                  {lastRequestId && (
                    <button
                      onClick={() => {
                        const apiUrl =
                          process.env.NEXT_PUBLIC_API_URL ||
                          process.env.NEXT_PUBLIC_RUNPOD_GENERATE_URL ||
                          "http://localhost:8003";
                        window.open(
                          `${apiUrl}/api/visualization/${lastRequestId}/download?format=zip`,
                          "_blank"
                        );
                      }}
                      className="px-3 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded text-xs font-medium transition-colors flex items-center justify-center gap-2 border border-[var(--border-color)]"
                      title="Download visualization (original + generated images)"
                    >
                      <FaImages className="w-3 h-3" />
                      {t("sidebar.downloadVisualization")}
                    </button>
                  )}
                </div>
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
                    {t("sidebar.returnToOriginal")}
                  </button>
                  <button
                    onClick={() => {
                      // Remove current image first
                      if (onRemoveImage) {
                        onRemoveImage();
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
                    {t("sidebar.newImage")}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* AI Task Selection */}
          {!isEditingDone && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t("sidebar.aiTask")}
              </h3>
              <select
                value={aiTask}
                onChange={(e) =>
                  onAiTaskChange(
                    e.target.value as
                      | "white-balance"
                      | "object-insert"
                      | "object-removal"
                      | "evaluation"
                  )
                }
                className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)] focus:border-transparent"
              >
                <option value="white-balance">
                  {t("sidebar.whiteBalance")}
                </option>
                <option value="object-insert">
                  {t("sidebar.objectInsert")}
                </option>
                <option value="object-removal">
                  {t("sidebar.objectRemoval")}
                </option>
              </select>
            </div>
          )}

          {/* White Balance Controls */}
          {!isEditingDone && aiTask === "white-balance" && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t("sidebar.whiteBalanceSettings")}
              </h3>
              <div className="space-y-4">
                {/* Method Selection */}
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t("sidebar.method")}
                  </label>
                  <select
                    onChange={(e) => {
                      const method = e.target.value as "auto" | "manual" | "ai";
                      onWhiteBalance?.(method);
                    }}
                    className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)] focus:border-transparent"
                  >
                    <option value="auto">
                      {t("sidebar.autoWhiteBalance")}
                    </option>
                    <option value="manual">
                      {t("sidebar.manualWhiteBalance")}
                    </option>
                    <option value="ai">{t("sidebar.aiWhiteBalance")}</option>
                  </select>
                </div>

                {/* Manual Controls */}
                <div className="space-y-3">
                  <div>
                    <label className="block text-[var(--text-secondary)] text-sm mb-2">
                      {t("sidebar.temperature")}: {whiteBalanceTemperature}
                    </label>
                    <input
                      type="range"
                      min="-100"
                      max="100"
                      value={whiteBalanceTemperature}
                      onChange={(e) =>
                        onWhiteBalanceTemperatureChange?.(
                          parseInt(e.target.value)
                        )
                      }
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-[var(--text-secondary)] text-sm mb-2">
                      {t("sidebar.tint")}: {whiteBalanceTint}
                    </label>
                    <input
                      type="range"
                      min="-100"
                      max="100"
                      value={whiteBalanceTint}
                      onChange={(e) =>
                        onWhiteBalanceTintChange?.(parseInt(e.target.value))
                      }
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Reference Image Upload (only for object insert) */}
          {!isEditingDone && aiTask === "object-insert" && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t("sidebar.referenceImage")}
              </h3>
              <div className="space-y-3">
                <label
                  htmlFor="reference-image-upload"
                  className="w-full px-4 py-3 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white rounded-lg cursor-pointer transition-colors text-sm font-medium flex items-center justify-center gap-2 border-2 border-dashed border-[var(--primary-accent)]"
                >
                  <FaImage className="w-4 h-4" />
                  {t("sidebar.chooseReference")}
                </label>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
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
                      {t("sidebar.removeReference")}
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Benchmark Mode Section */}
          {!isEditingDone && appMode === "benchmark" && (
            <Box
              sx={{
                pb: 3,
                borderBottom: 1,
                borderColor: "divider",
              }}
            >
              <Typography
                variant="subtitle2"
                sx={{
                  mb: 2,
                  fontWeight: 600,
                  color: "text.primary",
                }}
              >
                🎯 Benchmark Mode
              </Typography>

              {/* Task Selection */}
              <Box sx={{ mb: 2 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", mb: 1, display: "block" }}
                >
                  Task Type
                </Typography>
                <ToggleButtonGroup
                  value={benchmarkTask}
                  exclusive
                  onChange={(_, value) => {
                    if (value !== null) {
                      onBenchmarkTaskChange?.(value);
                    }
                  }}
                  fullWidth
                  size="small"
                >
                  <ToggleButton value="object-removal">
                    Object Removal ✅
                  </ToggleButton>
                  <ToggleButton value="white-balance" disabled>
                    White Balancing 🚧
                  </ToggleButton>
                  <ToggleButton value="object-insert" disabled>
                    Object Insertion 🚧
                  </ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {/* Benchmark Dataset Upload */}
              <Box sx={{ mb: 2 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", mb: 1, display: "block" }}
                >
                  📁 Benchmark Dataset
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: "text.secondary",
                    fontSize: "0.7rem",
                    display: "block",
                    mb: 1,
                  }}
                >
                  Upload ZIP file or folder containing: input/, mask/,
                  groundtruth/
                </Typography>

                {/* ZIP Upload */}
                <input
                  type="file"
                  accept=".zip"
                  id="benchmark-zip-upload"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      console.log(
                        "📤 ZIP file selected:",
                        file.name,
                        file.size,
                        "bytes"
                      );
                      // Validate first (this will also set the file and folder name)
                      if (onBenchmarkFolderValidate) {
                        onBenchmarkFolderValidate(file);
                      } else {
                        // Fallback: just set folder name if validate handler not provided
                        onBenchmarkFolderChange?.(file.name);
                      }
                    } else {
                      console.warn("⚠️ No file selected");
                    }
                  }}
                />
                <label htmlFor="benchmark-zip-upload">
                  <Button
                    component="span"
                    variant="outlined"
                    fullWidth
                    size="small"
                    startIcon={<CloudUploadIcon />}
                    sx={{ textTransform: "none", mb: 1 }}
                  >
                    📦 Upload ZIP File
                  </Button>
                </label>

                {/* Folder Upload */}
                <input
                  type="file"
                  id="benchmark-folder-upload"
                  style={{ display: "none" }}
                  {...({ webkitdirectory: "", directory: "" } as any)}
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length > 0) {
                      console.log("📁 Folder selected:", files.length, "files");
                      // Create a FileList-like object or pass files array
                      // For now, we'll create a ZIP-like structure on frontend
                      // Or pass to backend as folder structure
                      if (onBenchmarkFolderValidate) {
                        // Pass files as a special marker - backend will handle folder structure
                        // We'll need to update handler to accept FileList
                        const folderName =
                          files[0]?.webkitRelativePath?.split("/")[0] ||
                          "folder";
                        onBenchmarkFolderChange?.(folderName);
                        // Create a FormData with all files
                        const formData = new FormData();
                        files.forEach((file) => {
                          const relativePath =
                            file.webkitRelativePath || file.name;
                          formData.append("files", file, relativePath);
                        });
                        // We'll need to update the validation handler to accept folder uploads
                        // For now, show a message that folder upload needs backend support
                        console.log("📁 Folder upload:", formData);
                        // Call a new handler for folder upload
                        if (
                          (onBenchmarkFolderValidate as any).handleFolderUpload
                        ) {
                          (onBenchmarkFolderValidate as any).handleFolderUpload(
                            files
                          );
                        } else {
                          // Fallback: try to validate as folder structure
                          onBenchmarkFolderValidate(files as any);
                        }
                      }
                    } else {
                      console.warn("⚠️ No files selected");
                    }
                  }}
                />
                <label htmlFor="benchmark-folder-upload">
                  <Button
                    component="span"
                    variant="outlined"
                    fullWidth
                    size="small"
                    startIcon={<CloudUploadIcon />}
                    disabled={isValidatingBenchmark}
                    sx={{ textTransform: "none" }}
                  >
                    {isValidatingBenchmark
                      ? "Đang kiểm tra..."
                      : "📁 Upload Folder"}
                  </Button>
                </label>
                {benchmarkFolder && (
                  <Box
                    sx={{
                      mt: 1,
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                    }}
                  >
                    <Chip
                      label={benchmarkFolder}
                      size="small"
                      sx={{ fontSize: "0.7rem" }}
                    />
                    <Button
                      size="small"
                      onClick={() => {
                        onBenchmarkFolderChange?.("");
                        // Reset validation state
                        if (onBenchmarkFolderValidate) {
                          // Clear validation by passing empty string
                        }
                        // Reset file input
                        const input = document.getElementById(
                          "benchmark-zip-upload"
                        ) as HTMLInputElement;
                        if (input) input.value = "";
                      }}
                      sx={{ minWidth: "auto", px: 1 }}
                    >
                      ✕
                    </Button>
                  </Box>
                )}
              </Box>

              {/* Validation Status */}
              {isValidatingBenchmark && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: "info.light",
                    borderRadius: 1,
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                  }}
                >
                  <Box
                    sx={{
                      width: 16,
                      height: 16,
                      border: "2px solid",
                      borderColor: "info.main",
                      borderTopColor: "transparent",
                      borderRadius: "50%",
                      animation: "spin 1s linear infinite",
                      "@keyframes spin": {
                        "0%": { transform: "rotate(0deg)" },
                        "100%": { transform: "rotate(360deg)" },
                      },
                    }}
                  />
                  <Typography
                    variant="caption"
                    sx={{
                      color: "info.dark",
                      fontSize: "0.75rem",
                    }}
                  >
                    Đang kiểm tra cấu trúc folder/file...
                  </Typography>
                </Box>
              )}
              {!isValidatingBenchmark && benchmarkValidation && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: benchmarkValidation.success
                      ? "success.light"
                      : "error.light",
                    borderRadius: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: benchmarkValidation.success
                        ? "success.dark"
                        : "error.dark",
                      fontSize: "0.75rem",
                      display: "block",
                    }}
                  >
                    {benchmarkValidation.message}
                  </Typography>
                  {benchmarkValidation.success &&
                    benchmarkValidation.details && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: "text.secondary",
                          fontSize: "0.7rem",
                          display: "block",
                          mt: 0.5,
                        }}
                      >
                        Input: {benchmarkValidation.details.input || 0} | Mask:{" "}
                        {benchmarkValidation.details.mask || 0} | Groundtruth:{" "}
                        {benchmarkValidation.details.groundtruth || 0}
                      </Typography>
                    )}
                </Box>
              )}

              {/* Sample Count Selection */}
              {benchmarkValidation?.success && (
                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", mb: 1, display: "block" }}
                  >
                    🔢 Sample Count
                  </Typography>
                  <TextField
                    type="number"
                    value={benchmarkSampleCount || ""}
                    onChange={(e) => {
                      const inputValue = e.target.value.trim();

                      // Handle empty input
                      if (inputValue === "") {
                        onBenchmarkSampleCountChange?.(0);
                        return;
                      }

                      const value = parseInt(inputValue, 10);
                      const totalImages = benchmarkValidation?.image_count || 0;

                      // Validation logic
                      if (isNaN(value)) {
                        // Invalid input, reset to 0
                        onBenchmarkSampleCountChange?.(0);
                        return;
                      }

                      if (value < 0) {
                        // Negative value, fallback to 0 (all images)
                        onBenchmarkSampleCountChange?.(0);
                        return;
                      }

                      if (value > totalImages && totalImages > 0) {
                        // Value exceeds total, fallback to total (all images)
                        onBenchmarkSampleCountChange?.(totalImages);
                        return;
                      }

                      // Valid value
                      onBenchmarkSampleCountChange?.(value);
                    }}
                    onBlur={(e) => {
                      // Ensure value is valid on blur
                      const value = parseInt(e.target.value, 10) || 0;
                      const totalImages = benchmarkValidation?.image_count || 0;

                      if (value < 0) {
                        onBenchmarkSampleCountChange?.(0);
                      } else if (value > totalImages && totalImages > 0) {
                        onBenchmarkSampleCountChange?.(totalImages);
                      } else {
                        onBenchmarkSampleCountChange?.(value);
                      }
                    }}
                    inputProps={{
                      min: 0,
                      max: benchmarkValidation.image_count || 0,
                      step: 1,
                    }}
                    fullWidth
                    size="small"
                    placeholder="0 = all images"
                    helperText={(() => {
                      const total = benchmarkValidation.image_count || 0;
                      const current = benchmarkSampleCount || 0;

                      if (current === 0) {
                        return `Process all ${total} images`;
                      } else if (current > total) {
                        return `⚠️ Value exceeds total. Will process all ${total} images.`;
                      } else if (current < 0) {
                        return `⚠️ Invalid value. Will process all ${total} images.`;
                      } else {
                        return `Process ${current} randomly selected images (from ${total} total)`;
                      }
                    })()}
                    error={
                      benchmarkSampleCount !== undefined &&
                      benchmarkSampleCount !== null &&
                      (benchmarkSampleCount < 0 ||
                        (benchmarkValidation?.image_count !== undefined &&
                          benchmarkValidation.image_count > 0 &&
                          benchmarkSampleCount >
                            benchmarkValidation.image_count))
                    }
                    sx={{ mb: 1 }}
                  />

                  {/* Validation messages */}
                  {benchmarkSampleCount !== undefined &&
                    benchmarkSampleCount !== null && (
                      <>
                        {benchmarkSampleCount < 0 && (
                          <Typography
                            variant="caption"
                            sx={{
                              color: "error.main",
                              fontSize: "0.7rem",
                              display: "block",
                              mt: 0.5,
                            }}
                          >
                            ❌ Invalid: Negative values not allowed. Using 0
                            (all images).
                          </Typography>
                        )}
                        {benchmarkValidation?.image_count &&
                          benchmarkSampleCount >
                            benchmarkValidation.image_count && (
                            <Typography
                              variant="caption"
                              sx={{
                                color: "warning.main",
                                fontSize: "0.7rem",
                                display: "block",
                                mt: 0.5,
                              }}
                            >
                              ⚠️ Only {benchmarkValidation.image_count} images
                              available. Will process all{" "}
                              {benchmarkValidation.image_count} images.
                            </Typography>
                          )}
                        {benchmarkSampleCount > 0 &&
                          benchmarkSampleCount <=
                            (benchmarkValidation?.image_count || 0) && (
                            <Typography
                              variant="caption"
                              sx={{
                                color: "success.main",
                                fontSize: "0.7rem",
                                display: "block",
                                mt: 0.5,
                              }}
                            >
                              ✅ Will randomly select {benchmarkSampleCount}{" "}
                              images from{" "}
                              {benchmarkValidation?.image_count || 0} total.
                            </Typography>
                          )}
                      </>
                    )}

                  {/* Info box explaining sample selection */}
                  <Box
                    sx={{
                      mt: 1,
                      p: 1,
                      bgcolor: "background.paper",
                      borderRadius: 1,
                      border: "1px solid",
                      borderColor: "divider",
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{
                        color: "text.secondary",
                        fontSize: "0.7rem",
                        display: "block",
                      }}
                    >
                      <strong>Sample Selection Rules:</strong>
                    </Typography>
                    <ul
                      style={{
                        margin: "4px 0",
                        paddingLeft: "20px",
                        fontSize: "0.7rem",
                        color: "var(--text-secondary)",
                      }}
                    >
                      <li>
                        <strong>0</strong> = Process all images
                      </li>
                      <li>
                        <strong>1-N</strong> = Randomly select N images
                      </li>
                      <li>
                        <strong>&gt; Total</strong> = Fallback to all images
                        (with warning)
                      </li>
                      <li>
                        <strong>&lt; 0</strong> = Fallback to all images (with
                        warning)
                      </li>
                    </ul>
                  </Box>
                </Box>
              )}

              {/* Progress Indicator */}
              {isRunningBenchmark && benchmarkProgress && (
                <Box
                  sx={{ mb: 2, p: 1.5, bgcolor: "info.light", borderRadius: 1 }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: "info.dark",
                      fontWeight: 500,
                      display: "block",
                    }}
                  >
                    Processing Benchmark...
                  </Typography>
                  <Box sx={{ mt: 1, mb: 1 }}>
                    <Box
                      sx={{
                        width: "100%",
                        height: 8,
                        bgcolor: "grey.300",
                        borderRadius: 1,
                        overflow: "hidden",
                      }}
                    >
                      <Box
                        sx={{
                          width: `${
                            (benchmarkProgress.current /
                              benchmarkProgress.total) *
                            100
                          }%`,
                          height: "100%",
                          bgcolor: "primary.main",
                          transition: "width 0.3s ease",
                        }}
                      />
                    </Box>
                  </Box>
                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", fontSize: "0.7rem" }}
                  >
                    {benchmarkProgress.current} / {benchmarkProgress.total} (
                    {Math.round(
                      (benchmarkProgress.current / benchmarkProgress.total) *
                        100
                    )}
                    %)
                  </Typography>
                  {benchmarkProgress.currentImage && (
                    <Typography
                      variant="caption"
                      sx={{
                        color: "text.secondary",
                        fontSize: "0.7rem",
                        display: "block",
                        mt: 0.5,
                      }}
                    >
                      Current: {benchmarkProgress.currentImage}
                    </Typography>
                  )}
                </Box>
              )}

              {/* Info: Prompt is entered in Header */}
              <Box sx={{ mb: 2, p: 1, bgcolor: "info.light", borderRadius: 1 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: "info.dark",
                    fontSize: "0.7rem",
                    display: "block",
                  }}
                >
                  ℹ️ Enter prompt in the header above. It will be used for ALL
                  images in the benchmark.
                </Typography>
              </Box>

              {/* Task-specific Settings */}
              {benchmarkTask === "object-removal" && (
                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "text.secondary",
                      mb: 1,
                      display: "block",
                      fontWeight: 500,
                    }}
                  >
                    Object Removal Settings
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      color: "text.secondary",
                      fontSize: "0.7rem",
                      display: "block",
                      mb: 1,
                      fontStyle: "italic",
                    }}
                  >
                    Model: Qwen Image Edit 2509
                  </Typography>
                  {/* NOTE: NO White Balance Settings for Object Removal */}
                  {/* White balance is handled separately in White Balancing task */}
                </Box>
              )}

              {benchmarkTask === "white-balance" && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: "warning.light",
                    borderRadius: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: "warning.dark",
                      fontSize: "0.75rem",
                      display: "block",
                    }}
                  >
                    ⚠️ White Balancing task is not yet implemented.
                    <br />
                    Coming soon with Pix2Pix model integration.
                  </Typography>
                  {/* TODO: Implement White Balancing specific settings */}
                  {/* 
                    TODO: Add the following settings:
                    - Color temperature adjustment range
                    - Tint correction
                    - Auto white balance mode (on/off)
                    - Reference white point selection
                    - Illuminant type (D65, A, etc.)
                    - Strength of correction (0-100%)
                  */}
                </Box>
              )}

              {benchmarkTask === "object-insert" && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: "warning.light",
                    borderRadius: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: "warning.dark",
                      fontSize: "0.75rem",
                      display: "block",
                    }}
                  >
                    ⚠️ Object Insertion task is not yet implemented.
                    <br />
                    Will use Qwen Image Edit 2509 with insertion mode.
                  </Typography>
                  {/* TODO: Implement Object Insertion specific settings */}
                  {/* 
                    TODO: Add the following settings:
                    - Object prompt/description input
                    - Insertion position control
                    - Scale/size adjustment
                    - Blending mode (natural, artistic, etc.)
                    - Shadow generation (on/off)
                    - Lighting consistency (match scene lighting)
                    - Perspective correction
                  */}
                </Box>
              )}

              {/* Requirements Checklist */}
              {!isRunningBenchmark && (
                <Box
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: "background.paper",
                    borderRadius: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: "text.secondary",
                      fontSize: "0.7rem",
                      display: "block",
                      mb: 1,
                      fontWeight: 500,
                    }}
                  >
                    Requirements to run benchmark:
                  </Typography>
                  <ul
                    style={{
                      margin: "4px 0",
                      paddingLeft: "20px",
                      fontSize: "0.7rem",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {benchmarkValidation?.success ? (
                      <li style={{ color: "var(--success)" }}>
                        ✅ Upload benchmark folder or ZIP
                      </li>
                    ) : (
                      <li>❌ Upload benchmark folder or ZIP</li>
                    )}
                    <li>❌ Enter benchmark prompt (in header)</li>
                    {benchmarkTask === "object-removal" ? (
                      <li style={{ color: "var(--success)" }}>
                        ✅ Task implemented
                      </li>
                    ) : (
                      <li>❌ Task not yet implemented</li>
                    )}
                  </ul>
                </Box>
              )}

              {/* Benchmark Results */}
              {benchmarkResults && (
                <Box
                  sx={{ mt: 3, pt: 3, borderTop: 1, borderColor: "divider" }}
                >
                  <Typography
                    variant="subtitle2"
                    sx={{
                      mb: 2,
                      fontWeight: 600,
                      color: "text.primary",
                    }}
                  >
                    Benchmark Results
                  </Typography>
                  {benchmarkResults.summary && (
                    <Box
                      sx={{
                        mb: 2,
                        p: 1.5,
                        bgcolor: "background.paper",
                        borderRadius: 1,
                      }}
                    >
                      <Typography
                        variant="caption"
                        sx={{
                          color: "text.secondary",
                          display: "block",
                          mb: 0.5,
                        }}
                      >
                        Summary
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {benchmarkResults.summary.successful_evaluations || 0} /{" "}
                        {benchmarkResults.summary.total_pairs || 0} successful
                      </Typography>
                      {benchmarkResults.summary.total_generation_time && (
                        <Typography
                          variant="caption"
                          sx={{ color: "text.secondary" }}
                        >
                          Total time:{" "}
                          {benchmarkResults.summary.total_generation_time.toFixed(
                            2
                          )}
                          s
                        </Typography>
                      )}
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          )}

          {/* Masking Tool Section - Only for object insert and removal */}
          {!isEditingDone &&
            (aiTask === "object-insert" || aiTask === "object-removal") && (
              <div className="pb-4 border-b border-[var(--border-color)]">
                <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                  {aiTask === "object-insert"
                    ? t("sidebar.markInsertArea")
                    : t("sidebar.markRemovalArea")}
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
                      {isMaskingMode
                        ? t("sidebar.exitMasking")
                        : t("sidebar.startMasking")}
                    </button>
                    {isMaskingMode && (
                      <button
                        onClick={onClearMask}
                        className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors"
                      >
                        {t("sidebar.clearMask")}
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
                        title={t("sidebar.undoMaskStroke")}
                      >
                        ↶ {t("sidebar.undo")}
                      </button>
                      <button
                        onClick={onMaskRedo}
                        disabled={
                          !onMaskRedo ||
                          maskHistoryIndex >= maskHistoryLength - 1
                        }
                        className="px-3 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                        title={t("sidebar.redoMaskStroke")}
                      >
                        ↷ {t("sidebar.redo")}
                      </button>
                    </div>
                  )}

                  {isMaskingMode && (
                    <div className="space-y-3">
                      {/* Smart Masking Settings */}
                      <div className="space-y-2">
                        <label className="block text-[var(--text-secondary)] text-sm mb-2">
                          {t("sidebar.maskSettings")}
                        </label>
                        {onSmartMaskingChange && (
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={enableSmartMasking}
                                onChange={(e) =>
                                  onSmartMaskingChange(e.target.checked)
                                }
                                disabled={isSmartMaskLoading}
                                sx={{
                                  color: "var(--primary-accent)",
                                  "&.Mui-checked": {
                                    color: "var(--primary-accent)",
                                  },
                                  "&.Mui-disabled": {
                                    color: "var(--text-secondary)",
                                  },
                                }}
                              />
                            }
                            label={
                              <span className="flex items-center gap-2">
                                {t("sidebar.smartMasking")}
                                {isSmartMaskLoading && (
                                  <span className="inline-block w-4 h-4 border-2 border-[var(--primary-accent)] border-t-transparent rounded-full animate-spin" />
                                )}
                              </span>
                            }
                            className="text-[var(--text-primary)] text-sm"
                          />
                        )}
                      </div>

                      <div>
                        <label className="block text-[var(--text-secondary)] text-sm mb-2">
                          {t("sidebar.maskTool")}
                        </label>
                        <div className="flex gap-2 mb-3">
                          <button
                            onClick={() => onMaskToolTypeChange?.("brush")}
                            className={`flex-1 px-3 py-2 text-sm rounded border transition-colors ${
                              maskToolType === "brush"
                                ? "bg-[var(--primary-accent)] text-white border-[var(--primary-accent)]"
                                : "bg-[var(--secondary-bg)] text-[var(--text-primary)] border-[var(--border-color)] hover:bg-[var(--hover-bg)]"
                            }`}
                          >
                            {t("sidebar.brush")}
                          </button>
                          <button
                            onClick={() => onMaskToolTypeChange?.("box")}
                            className={`flex-1 px-3 py-2 text-sm rounded border transition-colors ${
                              maskToolType === "box"
                                ? "bg-[var(--primary-accent)] text-white border-[var(--primary-accent)]"
                                : "bg-[var(--secondary-bg)] text-[var(--text-primary)] border-[var(--border-color)] hover:bg-[var(--hover-bg)]"
                            }`}
                          >
                            {t("sidebar.box")}
                          </button>
                        </div>
                      </div>

                      {maskToolType === "brush" && (
                        <div>
                          <label className="block text-[var(--text-secondary)] text-sm mb-2">
                            {t("sidebar.brushSize")}
                          </label>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              min="1"
                              max="50"
                              value={maskBrushSize}
                              onChange={(e) => {
                                const value = parseInt(e.target.value);
                                if (
                                  !isNaN(value) &&
                                  value >= 1 &&
                                  value <= 50
                                ) {
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
                      )}
                    </div>
                  )}

                  <p className="text-xs text-[var(--text-secondary)]">
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
            )}

          {/* White Balance Settings (only for white balance task) */}
          {!isEditingDone && aiTask === "white-balance" && (
            <div className="pb-4 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t("sidebar.whiteBalanceSettings")}
              </h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t("sidebar.autoCorrectionStrength")}: 80%
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
                  {t("sidebar.aiAutoAdjust")}
                </p>
              </div>
            </div>
          )}

          {/* Advanced Settings */}
          {!isEditingDone && (
            <div className="pt-2">
              <h3 className="text-[var(--text-primary)] font-medium mb-4">
                {t("sidebar.advanced")}
              </h3>
              <div className="space-y-6">
                {/* Negative Prompt */}
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    {t("sidebar.negativePrompt")}
                  </label>
                  <textarea
                    placeholder={t("sidebar.negativePromptPlaceholder")}
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
                      {t("sidebar.guidanceScale")}
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
                    onChange={(e) =>
                      onGuidanceScaleChange?.(parseFloat(e.target.value))
                    }
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                    <span>{t("sidebar.moreCreative")}</span>
                    <span>{t("sidebar.followPromptStrictly")}</span>
                  </div>
                </div>

                {/* Steps (Updated from existing) */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t("sidebar.inferenceSteps")}
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
                    onChange={(e) =>
                      onInferenceStepsChange?.(parseInt(e.target.value))
                    }
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                    <span>{t("sidebar.faster")}</span>
                    <span>{t("sidebar.higherQuality")}</span>
                  </div>
                </div>

                {/* True CFG Scale */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t("sidebar.cfgScale")}
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
                    onChange={(e) =>
                      onCfgScaleChange?.(parseFloat(e.target.value))
                    }
                    className="w-full accent-[var(--primary-accent)]"
                  />
                  <p className="text-xs text-[var(--text-secondary)] mt-2">
                    {t("sidebar.cfgGuidanceDescription")}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Low-End Optimizations */}
          {!isEditingDone && (
            <div className="pt-2 border-t border-[var(--border-color)] mt-4">
              <h3 className="text-[var(--text-primary)] font-medium mb-4">
                Tối ưu cho GPU thấp (Low-End)
              </h3>
              <div className="space-y-4">
                <p className="text-xs text-[var(--text-secondary)] mb-3">
                  Các tính năng này giúp giảm VRAM cho GPU 12GB hoặc thấp hơn.
                  Lưu ý: Cần restart backend để áp dụng thay đổi.
                </p>

                {/* 4-bit Text Encoder */}
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <label className="block text-[var(--text-secondary)] text-sm mb-1">
                      4-bit Text Encoder
                    </label>
                    <p className="text-xs text-[var(--text-secondary)]">
                      Giảm ~4GB VRAM, chậm hơn một chút
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enable4BitTextEncoder}
                      onChange={(e) =>
                        onEnable4BitTextEncoderChange?.(e.target.checked)
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-[var(--border-color)] peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[var(--primary-accent)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--primary-accent)]"></div>
                  </label>
                </div>

                {/* CPU Offload */}
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <label className="block text-[var(--text-secondary)] text-sm mb-1">
                      CPU Offload
                    </label>
                    <p className="text-xs text-[var(--text-secondary)]">
                      Offload transformer & VAE về CPU, tiết kiệm VRAM nhưng
                      chậm hơn
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableCpuOffload}
                      onChange={(e) =>
                        onEnableCpuOffloadChange?.(e.target.checked)
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-[var(--border-color)] peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[var(--primary-accent)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--primary-accent)]"></div>
                  </label>
                </div>

                {/* Memory Optimizations */}
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <label className="block text-[var(--text-secondary)] text-sm mb-1">
                      Memory Optimizations
                    </label>
                    <p className="text-xs text-[var(--text-secondary)]">
                      Safetensors, low_cpu_mem_usage, TF32 matmul
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableMemoryOptimizations}
                      onChange={(e) =>
                        onEnableMemoryOptimizationsChange?.(e.target.checked)
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-[var(--border-color)] peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[var(--primary-accent)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--primary-accent)]"></div>
                  </label>
                </div>

                {/* FlowMatch Scheduler */}
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <label className="block text-[var(--text-secondary)] text-sm mb-1">
                      FlowMatch Scheduler
                    </label>
                    <p className="text-xs text-[var(--text-secondary)]">
                      Sử dụng FlowMatchEulerDiscreteScheduler thay vì scheduler
                      mặc định
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableFlowmatchScheduler}
                      onChange={(e) =>
                        onEnableFlowmatchSchedulerChange?.(e.target.checked)
                      }
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-[var(--border-color)] peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[var(--primary-accent)] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[var(--primary-accent)]"></div>
                  </label>
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
        ></div>
      )}
    </div>
  );
}
