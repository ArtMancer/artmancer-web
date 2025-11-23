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

interface SidebarProps {
  isOpen: boolean;
  width?: number; // Optional width for resizable functionality
  uploadedImage: string | null;
  referenceImage: string | null;
  aiTask: "white-balance" | "object-insert" | "object-removal" | "evaluation";
  appMode?: "inference" | "evaluation";
  isMaskingMode: boolean;
  maskBrushSize: number;
  maskToolType?: "brush" | "box";
  isResizing?: boolean; // For resize handle styling
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
}

export default function Sidebar({
  isOpen,
  width = 320, // Default width
  uploadedImage,
  referenceImage,
  aiTask,
  appMode = "inference",
  isMaskingMode,
  maskBrushSize,
  maskToolType = "brush",
  isResizing = false,
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
}: SidebarProps) {
  // Translation hook
  const { t } = useLanguage();

  // Determine if we're in "editing done" state (comparison mode)
  const isEditingDone =
    originalImage && modifiedImage && originalImage !== modifiedImage;
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
                        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8003';
                        window.open(`${apiUrl}/api/visualization/${lastRequestId}/download?format=zip`, '_blank');
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

          {/* AI Task Selection - Only show in inference mode */}
          {!isEditingDone && appMode === "inference" && (
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

          {/* White Balance Controls - Only in inference mode */}
          {!isEditingDone &&
            appMode === "inference" &&
            aiTask === "white-balance" && (
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
                        const method = e.target.value as
                          | "auto"
                          | "manual"
                          | "ai";
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

          {/* Reference Image Upload (only for object insert) - Only in inference mode */}
          {!isEditingDone &&
            appMode === "inference" &&
            aiTask === "object-insert" && (
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

          {/* Evaluation Section */}
          {!isEditingDone && appMode === "evaluation" && (
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
                {t("sidebar.evaluationMode")}
              </Typography>

              {/* Task Selection */}
              <Box sx={{ mb: 2 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", mb: 1, display: "block" }}
                >
                  {t("sidebar.aiTask")}
                </Typography>
                <ToggleButtonGroup
                  value={evaluationTask}
                  exclusive
                  onChange={(_, value) => {
                    if (value !== null) {
                      onEvaluationTaskChange?.(value);
                    }
                  }}
                  fullWidth
                  size="small"
                >
                  <ToggleButton value="white-balance">
                    {t("sidebar.whiteBalance")}
                  </ToggleButton>
                  <ToggleButton value="object-removal">
                    {t("sidebar.objectRemoval")}
                  </ToggleButton>
                  <ToggleButton value="object-insert">
                    {t("sidebar.objectInsert")}
                  </ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {/* Mode Selection */}
              <Box sx={{ mb: 3 }}>
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", mb: 1, display: "block" }}
                >
                  {t("sidebar.evaluationMode")}
                </Typography>
                <ToggleButtonGroup
                  value={evaluationMode}
                  exclusive
                  onChange={(_, value) => {
                    if (value !== null) {
                      onEvaluationModeChange?.(value);
                    }
                  }}
                  fullWidth
                  size="small"
                >
                  <ToggleButton value="single">
                    {t("sidebar.singleImage")}
                  </ToggleButton>
                  <ToggleButton value="multiple">
                    {t("sidebar.multipleImages")}
                  </ToggleButton>
                </ToggleButtonGroup>
              </Box>

              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                {/* Single Image Pair Upload - Only show when mode is single */}
                {evaluationMode === "single" && (
                  <Box
                    sx={{ display: "flex", flexDirection: "column", gap: 2 }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "text.secondary", fontWeight: 500 }}
                    >
                      {t("sidebar.uploadSinglePair")}
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 1.5,
                      }}
                    >
                      <Box>
                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/jpg,image/webp"
                          id="eval-single-original-upload"
                          style={{ display: "none" }}
                          onChange={onEvaluationSingleOriginalUpload}
                        />
                        <label htmlFor="eval-single-original-upload">
                          <Button
                            component="span"
                            variant="outlined"
                            fullWidth
                            size="small"
                            startIcon={<CloudUploadIcon />}
                            sx={{
                              textTransform: "none",
                              fontSize: "0.75rem",
                            }}
                          >
                            {t("sidebar.uploadOriginal")}
                          </Button>
                        </label>
                        {evaluationSingleOriginal && (
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              mt: 0.5,
                            }}
                          >
                            <Chip
                              icon={<CheckCircleIcon />}
                              label="Loaded"
                              size="small"
                              color="success"
                              sx={{ height: 20, fontSize: "0.65rem" }}
                            />
                          </Box>
                        )}
                      </Box>
                      <Box>
                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/jpg,image/webp"
                          id="eval-single-target-upload"
                          style={{ display: "none" }}
                          onChange={onEvaluationSingleTargetUpload}
                        />
                        <label htmlFor="eval-single-target-upload">
                          <Button
                            component="span"
                            variant="outlined"
                            fullWidth
                            size="small"
                            startIcon={<CloudUploadIcon />}
                            sx={{
                              textTransform: "none",
                              fontSize: "0.75rem",
                            }}
                          >
                            {t("sidebar.uploadTarget")}
                          </Button>
                        </label>
                        {evaluationSingleTarget && (
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              mt: 0.5,
                            }}
                          >
                            <Chip
                              icon={<CheckCircleIcon />}
                              label="Loaded"
                              size="small"
                              color="success"
                              sx={{ height: 20, fontSize: "0.65rem" }}
                            />
                          </Box>
                        )}
                      </Box>
                    </Box>
                  </Box>
                )}

                {/* Multiple Image Pairs Upload - Only show when mode is multiple */}
                {evaluationMode === "multiple" && (
                  <Box
                    sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "text.secondary", fontWeight: 500 }}
                    >
                      {t("sidebar.uploadMultiplePairs")}
                    </Typography>
                    
                    {/* Original Images Folder Upload */}
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                      <Typography
                        variant="caption"
                        sx={{ color: "text.secondary", fontSize: "0.75rem" }}
                      >
                        {t("sidebar.originalImagesFolder")} (1.png, 2.png, ...)
                      </Typography>
                      <input
                        type="file"
                        accept="image/png,image/jpeg,image/jpg,image/webp"
                        multiple
                        webkitdirectory=""
                        directory=""
                        id="eval-original-folder-upload"
                        style={{ display: "none" }}
                        onChange={onEvaluationOriginalFolderUpload}
                      />
                      <label htmlFor="eval-original-folder-upload">
                        <Button
                          component="span"
                          variant="outlined"
                          fullWidth
                          size="small"
                          startIcon={<CloudUploadIcon />}
                          sx={{ textTransform: "none" }}
                        >
                          {t("sidebar.selectOriginalFolder")}
                        </Button>
                      </label>
                    </Box>

                    {/* Target Images Folder Upload */}
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                      <Typography
                        variant="caption"
                        sx={{ color: "text.secondary", fontSize: "0.75rem" }}
                      >
                        {t("sidebar.targetImagesFolder")} (1.png, 2.png, ...)
                      </Typography>
                      <input
                        type="file"
                        accept="image/png,image/jpeg,image/jpg,image/webp"
                        multiple
                        webkitdirectory=""
                        directory=""
                        id="eval-target-folder-upload"
                        style={{ display: "none" }}
                        onChange={onEvaluationTargetFolderUpload}
                      />
                      <label htmlFor="eval-target-folder-upload">
                        <Button
                          component="span"
                          variant="outlined"
                          fullWidth
                          size="small"
                          startIcon={<CloudUploadIcon />}
                          sx={{ textTransform: "none" }}
                        >
                          {t("sidebar.selectTargetFolder")}
                        </Button>
                      </label>
                    </Box>

                    {evaluationImagePairs.length > 0 && (
                      <Chip
                        label={`${evaluationImagePairs.length} ${t(
                          "sidebar.pairsLoaded"
                        )}`}
                        size="small"
                        color="primary"
                        sx={{ alignSelf: "flex-start" }}
                      />
                    )}
                  </Box>
                )}

                {/* Reference Image - Only for object-insert task */}
                {evaluationTask === "object-insert" && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                    <Typography
                      variant="caption"
                      sx={{ color: "text.secondary", fontWeight: 500 }}
                    >
                      {t("sidebar.referenceImage")} ({t("sidebar.required")})
                    </Typography>
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg,image/webp"
                      id="eval-reference-upload"
                      style={{ display: "none" }}
                      onChange={onEvaluationReferenceImageUpload}
                    />
                    <label htmlFor="eval-reference-upload">
                      <Button
                        component="span"
                        variant="outlined"
                        fullWidth
                        size="small"
                        startIcon={<ImageIcon />}
                        sx={{ textTransform: "none" }}
                      >
                        {t("sidebar.chooseReference")}
                      </Button>
                    </label>
                    {evaluationReferenceImage && (
                      <Chip
                        icon={<CheckCircleIcon />}
                        label={t("sidebar.loaded")}
                        size="small"
                        color="success"
                        sx={{ alignSelf: "flex-start" }}
                      />
                    )}
                  </Box>
                )}

                {/* Conditional Images (Optional) - Show for both modes */}
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", fontWeight: 500 }}
                  >
                    {t("sidebar.uploadConditionalImages")} (
                    {t("sidebar.optional")})
                  </Typography>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    multiple
                    webkitdirectory=""
                    directory=""
                    id="eval-conditional-upload"
                    style={{ display: "none" }}
                    onChange={onEvaluationConditionalImagesUpload}
                  />
                  <label htmlFor="eval-conditional-upload">
                    <Button
                      component="span"
                      variant="outlined"
                      fullWidth
                      size="small"
                      startIcon={<ImageIcon />}
                      sx={{ textTransform: "none" }}
                    >
                      Upload Conditional Images
                    </Button>
                  </label>
                  {evaluationConditionalImages.length > 0 && (
                    <Chip
                      label={`${evaluationConditionalImages.length} ${t(
                        "sidebar.imagesLoaded"
                      )}`}
                      size="small"
                      color="primary"
                      sx={{ alignSelf: "flex-start" }}
                    />
                  )}
                </Box>
              </Box>

              {/* Evaluation Results Section */}
              {evaluationResults && evaluationResults.length > 0 && (
                <Box sx={{ mt: 3, pt: 3, borderTop: 1, borderColor: "divider" }}>
                  <Typography
                    variant="subtitle2"
                    sx={{
                      mb: 2,
                      fontWeight: 600,
                      color: "text.primary",
                    }}
                  >
                    {t("sidebar.evaluationResults")}
                  </Typography>

                  {/* Summary */}
                  {evaluationResponse && (
                    <Box sx={{ mb: 2, p: 1.5, bgcolor: "background.paper", borderRadius: 1 }}>
                      <Typography variant="caption" sx={{ color: "text.secondary", display: "block", mb: 0.5 }}>
                        Summary
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {evaluationResponse.successful_evaluations || 0} / {evaluationResponse.total_pairs || 0} successful
                      </Typography>
                      {evaluationResponse.total_evaluation_time && (
                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                          Total time: {evaluationResponse.total_evaluation_time.toFixed(2)}s
                        </Typography>
                      )}
                    </Box>
                  )}

                  {/* Results List */}
                  <Box sx={{ mb: 2, maxHeight: 200, overflowY: "auto" }}>
                    {evaluationResults.slice(0, 5).map((result, index) => (
                      <Box
                        key={index}
                        sx={{
                          mb: 1,
                          p: 1,
                          bgcolor: result.success ? "success.light" : "error.light",
                          borderRadius: 1,
                          opacity: result.success ? 1 : 0.7,
                        }}
                      >
                        <Typography variant="caption" sx={{ fontWeight: 500, display: "block" }}>
                          {result.filename || `Result ${index + 1}`}
                        </Typography>
                        {result.success && result.metrics && (
                          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mt: 0.5 }}>
                            {result.metrics.psnr && (
                              <Chip
                                label={`PSNR: ${result.metrics.psnr.toFixed(2)} dB`}
                                size="small"
                                sx={{ height: 20, fontSize: "0.65rem" }}
                              />
                            )}
                            {result.metrics.ssim && (
                              <Chip
                                label={`SSIM: ${result.metrics.ssim.toFixed(3)}`}
                                size="small"
                                sx={{ height: 20, fontSize: "0.65rem" }}
                              />
                            )}
                          </Box>
                        )}
                        {!result.success && result.error && (
                          <Typography variant="caption" sx={{ color: "error.main", fontSize: "0.7rem" }}>
                            {result.error}
                          </Typography>
                        )}
                      </Box>
                    ))}
                    {evaluationResults.length > 5 && (
                      <Typography variant="caption" sx={{ color: "text.secondary", fontStyle: "italic" }}>
                        +{evaluationResults.length - 5} more results
                      </Typography>
                    )}
                  </Box>

                  {/* Export Buttons */}
                  <Box sx={{ display: "flex", gap: 1 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      fullWidth
                      startIcon={<FaDownload />}
                      onClick={onExportEvaluationJSON}
                      sx={{ textTransform: "none", fontSize: "0.75rem" }}
                    >
                      Export JSON
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      fullWidth
                      startIcon={<FaDownload />}
                      onClick={onExportEvaluationCSV}
                      sx={{ textTransform: "none", fontSize: "0.75rem" }}
                    >
                      Export CSV
                    </Button>
                  </Box>
                </Box>
              )}
            </Box>
          )}

          {/* Masking Tool Section - Only for object insert and removal in inference mode */}
          {!isEditingDone &&
            appMode === "inference" &&
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
                                onChange={(e) => onSmartMaskingChange(e.target.checked)}
                                disabled={isSmartMaskLoading}
                                sx={{
                                  color: 'var(--primary-accent)',
                                  '&.Mui-checked': {
                                    color: 'var(--primary-accent)',
                                  },
                                  '&.Mui-disabled': {
                                    color: 'var(--text-secondary)',
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
