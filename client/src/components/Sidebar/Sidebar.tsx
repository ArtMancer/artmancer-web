import React, { useState, useEffect } from "react";
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
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  IconButton,
  Tooltip,
  Paper,
  Stack,
} from "@mui/material";
import {
  CloudUpload as CloudUploadIcon,
  Image as ImageIcon,
  CheckCircle as CheckCircleIcon,
  CameraAlt as CameraIcon,
  Delete as DeleteIcon,
  HelpOutline as HelpIcon,
  Image as ImageIconMUI,
} from "@mui/icons-material";
import type { InputQualityPreset } from "@/services/api";
import { apiService } from "@/services/api";

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
  customSquareSize?: number; // Custom size for 1:1 ratio (e.g., 512, 768, 1024)
  isApplyingQuality?: boolean;
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onReferenceImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onRemoveReferenceImage: () => void;
  onEditReferenceImage?: () => void;
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
  smartMaskModelType?: 'segmentation' | 'birefnet';
  onSmartMaskModelTypeChange?: (modelType: 'segmentation' | 'birefnet') => void;
  borderAdjustment?: number;
  onBorderAdjustmentChange?: (value: number) => void;
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
  seed?: number;
  onNegativePromptChange?: (value: string) => void;
  onGuidanceScaleChange?: (value: number) => void;
  onInferenceStepsChange?: (value: number) => void;
  onCfgScaleChange?: (value: number) => void;
  onSeedChange?: (value: number) => void;
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
  onCustomSquareSizeChange?: (size: number) => void; // Handler for custom square size
  // Low-end optimization props
  // Debug panel props
  isDebugPanelVisible?: boolean;
  onDebugPanelVisibilityChange?: (visible: boolean) => void;
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
  customSquareSize = 1024,
  isApplyingQuality = false,
  onImageUpload,
  onReferenceImageUpload,
  onRemoveImage,
  onRemoveReferenceImage,
  onEditReferenceImage,
  onAiTaskChange,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onMaskToolTypeChange,
  // Smart masking props
  enableSmartMasking = true,
  isSmartMaskLoading = false,
  onSmartMaskingChange,
  smartMaskModelType = 'segmentation',
  onSmartMaskModelTypeChange,
  borderAdjustment = 0,
  onBorderAdjustmentChange,
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
  inferenceSteps = 10,
  cfgScale = 4.0,
  seed = 42, // Default seed: 42 (famous default)
  onNegativePromptChange,
  onGuidanceScaleChange,
  onInferenceStepsChange,
  onCfgScaleChange,
  onSeedChange,
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
  onCustomSquareSizeChange,
  // Benchmark mode props with defaults
  benchmarkFolder = "",
  benchmarkValidation = null,
  isValidatingBenchmark = false,
  benchmarkSampleCount = 0,
  benchmarkTask = "object-removal",
  isRunningBenchmark = false,
  benchmarkProgress = null,
  benchmarkResults = null,
  benchmarkPrompt = "",
  onBenchmarkFolderChange,
  onBenchmarkFolderValidate,
  onBenchmarkSampleCountChange,
  onBenchmarkTaskChange,
  onBenchmarkPromptChange,
  onRunBenchmark,
  // Debug panel props with defaults
  isDebugPanelVisible = false,
  onDebugPanelVisibilityChange,
}: SidebarProps) {
  // Translation hook
  const { t } = useLanguage();

  // State for negative prompt toggle - enabled if negativePrompt has value
  const [isNegativePromptEnabled, setIsNegativePromptEnabled] = useState(
    () => negativePrompt.trim().length > 0
  );

  // Sync toggle state when negativePrompt changes externally
  useEffect(() => {
    if (negativePrompt.trim().length > 0 && !isNegativePromptEnabled) {
      setIsNegativePromptEnabled(true);
    }
  }, [negativePrompt, isNegativePromptEnabled]);

  // Determine if we're in "editing done" state (comparison mode)
  const isEditingDone =
    originalImage && modifiedImage && originalImage !== modifiedImage;

  const inputQualityOptions: {
    value: InputQualityPreset;
    label: string;
    ratio: string;
  }[] = [
    {
      value: "resized",
      label: "Resize 1:1",
      ratio: "1:1",
    },
    {
      value: "original",
      label: "Ảnh gốc",
      ratio: "Gốc",
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
      className={`bg-[var(--panel-bg)] flex-shrink-0 flex flex-col lg:flex-col fixed right-0 z-30 sidebar-scrollable ${
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
            <Box sx={{ pb: 2 }}>
              <Typography
                variant="subtitle2"
                sx={{
                  mb: 2,
                  color: "var(--text-primary)",
                  fontWeight: 500,
                  fontSize: { xs: "0.875rem", sm: "1rem" },
                }}
              >
                {t("sidebar.imageUpload")}
              </Typography>
              <Stack spacing={1.5}>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  onChange={onImageUpload}
                  style={{ display: "none" }}
                  id="image-upload-panel"
                />
                <label htmlFor="image-upload-panel">
                  <Button
                    component="span"
                    fullWidth
                    variant="contained"
                    startIcon={<CameraIcon />}
                    sx={{
                      bgcolor: "var(--primary-accent)",
                      color: "white",
                      "&:hover": {
                        bgcolor: "var(--highlight-accent)",
                      },
                      py: 1.5,
                      textTransform: "none",
                      fontSize: "0.875rem",
                    }}
                  >
                    {t("sidebar.chooseImage")}
                  </Button>
                </label>
                {uploadedImage && (
                  <Button
                    fullWidth
                    variant="contained"
                    color="error"
                    onClick={onRemoveImage}
                    sx={{
                      py: 1.25,
                      textTransform: "none",
                      fontSize: "0.875rem",
                    }}
                  >
                    {t("sidebar.removeImage")}
                  </Button>
                )}
              </Stack>
            </Box>
          )}

          {/* Image Resolution Section */}
          {!isEditingDone && uploadedImage && (
            <>
              <Divider sx={{ my: 2, borderColor: "var(--border-color)" }} />
              <Box sx={{ pb: 2 }}>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    mb: 2,
                  }}
                >
                  <Typography
                    variant="subtitle2"
                    sx={{
                      color: "var(--text-primary)",
                      fontWeight: 500,
                      fontSize: { xs: "0.875rem", sm: "1rem" },
                    }}
                  >
                    Độ phân giải ảnh
                  </Typography>
                  <Tooltip title="Giảm kích thước ảnh đầu vào để tiết kiệm VRAM. Mức thấp hơn giúp chạy nhanh hơn nhưng ít chi tiết hơn.">
                    <IconButton
                      size="small"
                      sx={{
                        color: "var(--text-secondary)",
                        "&:hover": {
                          color: "var(--primary-accent)",
                        },
                      }}
                    >
                      <HelpIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                <Stack spacing={1.5}>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                    {inputQualityOptions.map((option) => (
                      <Box key={option.value} sx={{ width: "calc(50% - 4px)", minWidth: 0 }}>
                        <Button
                          fullWidth
                          onClick={() => onInputQualityChange(option.value)}
                          disabled={isApplyingQuality}
                          variant={
                            inputQuality === option.value
                              ? "contained"
                              : "outlined"
                          }
                          sx={{
                            bgcolor:
                              inputQuality === option.value
                                ? "var(--primary-accent)"
                                : "transparent",
                            color:
                              inputQuality === option.value
                                ? "white"
                                : "var(--text-primary)",
                            borderColor:
                              inputQuality === option.value
                                ? "var(--primary-accent)"
                                : "var(--border-color)",
                            "&:hover": {
                              bgcolor:
                                inputQuality === option.value
                                  ? "var(--primary-accent)"
                                  : "var(--hover-bg)",
                              borderColor: "var(--primary-accent)",
                            },
                            py: 1.5,
                            px: 1.5,
                            textTransform: "none",
                            textAlign: "left",
                            justifyContent: "space-between",
                            fontSize: "0.875rem",
                            opacity: isApplyingQuality ? 0.6 : 1,
                            minHeight: 48,
                            height: 48,
                          }}
                        >
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              width: "100%",
                            }}
                          >
                            <Typography variant="body2" sx={{ fontSize: "0.875rem" }}>
                              {option.label}
                            </Typography>
                            <Typography
                              variant="caption"
                              sx={{ fontSize: "0.75rem", opacity: 0.8 }}
                            >
                              {option.ratio}
                            </Typography>
                          </Box>
                        </Button>
                      </Box>
                    ))}
                  </Stack>
                  {/* Custom size input for 1:1 mode */}
                  {inputQuality === "resized" && (
                    <Box sx={{ mt: 2 }}>
                      <Typography
                        variant="caption"
                        sx={{
                          color: "var(--text-secondary)",
                          fontSize: "0.75rem",
                          display: "block",
                          mb: 1,
                        }}
                      >
                        Kích thước vuông (px):
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                        <TextField
                          type="number"
                          size="small"
                          value={customSquareSize}
                          onChange={(e) => {
                            const value = parseInt(e.target.value) || 1024;
                            const clamped = Math.max(256, Math.min(2048, value));
                            onCustomSquareSizeChange?.(clamped);
                          }}
                          disabled={isApplyingQuality}
                          inputProps={{ min: 256, max: 2048, step: 64 }}
                          sx={{
                            width: 96,
                            "& .MuiOutlinedInput-root": {
                              bgcolor: "var(--secondary-bg)",
                              borderColor: "var(--border-color)",
                              color: "var(--text-primary)",
                              fontSize: "0.875rem",
                              "& fieldset": {
                                borderColor: "var(--border-color)",
                              },
                              "&:hover fieldset": {
                                borderColor: "var(--primary-accent)",
                              },
                              "&.Mui-focused fieldset": {
                                borderColor: "var(--primary-accent)",
                              },
                            },
                          }}
                        />
                        <Typography
                          variant="caption"
                          sx={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}
                        >
                          × {customSquareSize}
                        </Typography>
                      </Box>
                      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                        {[512, 768, 1024, 1536, 2048].map((size) => (
                          <Chip
                            key={size}
                            label={size}
                            onClick={() => onCustomSquareSizeChange?.(size)}
                            disabled={isApplyingQuality}
                            size="small"
                            sx={{
                              bgcolor:
                                customSquareSize === size
                                  ? "var(--primary-accent)"
                                  : "var(--secondary-bg)",
                              color:
                                customSquareSize === size
                                  ? "white"
                                  : "var(--text-secondary)",
                              borderColor:
                                customSquareSize === size
                                  ? "var(--primary-accent)"
                                  : "var(--border-color)",
                              border: "1px solid",
                              cursor: "pointer",
                              "&:hover": {
                                bgcolor:
                                  customSquareSize === size
                                    ? "var(--primary-accent)"
                                    : "var(--hover-bg)",
                              },
                              fontSize: "0.75rem",
                            }}
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                  {selectedQuality && (
                    <Typography
                      variant="caption"
                      sx={{
                        color: "var(--text-secondary)",
                        fontSize: "0.75rem",
                        display: "block",
                      }}
                    >
                      {inputQuality === "resized"
                        ? `Ảnh sẽ được resize về ${customSquareSize}×${customSquareSize}`
                        : `Đang chọn: ${selectedQuality.label} (${selectedQuality.ratio} kích thước gốc)`}
                    </Typography>
                  )}
                  {isApplyingQuality && (
                    <Typography
                      variant="caption"
                      sx={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}
                    >
                      Đang áp dụng độ phân giải mới, vui lòng chờ...
                    </Typography>
                  )}
                  {showOriginalWarning && (
                    <Typography
                      variant="caption"
                      sx={{ color: "#fbbf24", fontSize: "0.75rem" }}
                    >
                      Ảnh lớn ({imageDimensions?.width}×{imageDimensions?.height}
                      ). Có thể tốn rất nhiều thời gian và VRAM.
                    </Typography>
                  )}
                </Stack>
              </Box>
            </>
          )}

          {/* Editing Done Message */}
          {isEditingDone && (
            <Paper
              sx={{
                bgcolor: "var(--secondary-bg)",
                border: 1,
                borderColor: "var(--success)",
                borderRadius: 2,
                p: 2,
                textAlign: "center",
              }}
            >
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 1,
                  mb: 1.5,
                }}
              >
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    bgcolor: "var(--success)",
                    borderRadius: "50%",
                  }}
                />
                <Typography
                  variant="subtitle2"
                  sx={{
                    color: "var(--success)",
                    fontWeight: 500,
                    fontSize: "0.875rem",
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5,
                  }}
                >
                  <HiSparkles style={{ width: 12, height: 12 }} />
                  {t("sidebar.editingComplete")}
                </Typography>
              </Box>
              <Typography
                variant="caption"
                sx={{
                  color: "var(--text-secondary)",
                  fontSize: "0.75rem",
                  display: "block",
                  mb: 2,
                }}
              >
                {t("sidebar.editingCompleteDesc")}
              </Typography>
              <Stack spacing={1}>
                <Stack direction="row" spacing={1}>
                  <Box sx={{ width: "calc(50% - 4px)", minWidth: 0 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={<FaDownload style={{ width: 12, height: 12 }} />}
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
                      sx={{
                        bgcolor: "var(--primary-accent)",
                        color: "white",
                        "&:hover": {
                          bgcolor: "var(--highlight-accent)",
                        },
                        py: 1,
                        textTransform: "none",
                        fontSize: "0.75rem",
                      }}
                    >
                      {t("sidebar.downloadImage")}
                    </Button>
                  </Box>
                  {lastRequestId && (
                    <Box sx={{ width: "calc(50% - 4px)", minWidth: 0 }}>
                      <Tooltip title="Download visualization (original + generated images)">
                        <Button
                          fullWidth
                          variant="outlined"
                          startIcon={<FaImages style={{ width: 12, height: 12 }} />}
                          onClick={() => {
                            apiService.downloadVisualization(lastRequestId, "zip");
                          }}
                          sx={{
                            bgcolor: "var(--secondary-bg)",
                            color: "var(--text-primary)",
                            borderColor: "var(--border-color)",
                            "&:hover": {
                              bgcolor: "var(--primary-accent)",
                              color: "white",
                              borderColor: "var(--primary-accent)",
                            },
                            py: 1,
                            textTransform: "none",
                            fontSize: "0.75rem",
                          }}
                        >
                          {t("sidebar.downloadVisualization")}
                        </Button>
                      </Tooltip>
                    </Box>
                  )}
                </Stack>
                <Stack direction="row" spacing={1}>
                  <Box sx={{ width: "calc(50% - 4px)", minWidth: 0 }}>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<FaUndo style={{ width: 12, height: 12 }} />}
                      onClick={() => {
                        if (onReturnToOriginal) {
                          onReturnToOriginal();
                        }
                      }}
                      sx={{
                        bgcolor: "var(--secondary-bg)",
                        color: "var(--text-primary)",
                        borderColor: "var(--border-color)",
                        "&:hover": {
                          bgcolor: "var(--primary-accent)",
                          color: "white",
                          borderColor: "var(--primary-accent)",
                        },
                        py: 1,
                        textTransform: "none",
                        fontSize: "0.75rem",
                      }}
                    >
                      {t("sidebar.returnToOriginal")}
                    </Button>
                  </Box>
                  <Box sx={{ width: "calc(50% - 4px)", minWidth: 0 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      startIcon={<CameraIcon />}
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
                      sx={{
                        bgcolor: "var(--success)",
                        color: "white",
                        "&:hover": {
                          bgcolor: "var(--success)",
                          opacity: 1,
                        },
                        opacity: 0.9,
                        py: 1,
                        textTransform: "none",
                        fontSize: "0.75rem",
                      }}
                    >
                      {t("sidebar.newImage")}
                    </Button>
                  </Box>
                </Stack>
              </Stack>
            </Paper>
          )}

          {/* AI Task Selection */}
          {!isEditingDone && (
            <>
              {(uploadedImage || isEditingDone) && (
                <Divider sx={{ my: 2, borderColor: "var(--border-color)" }} />
              )}
            <div className="pb-4">
              <h3 className="text-[var(--text-primary)] font-medium mb-3 text-sm lg:text-base">
                {t("sidebar.aiTask")}
              </h3>
              <FormControl fullWidth size="small">
                <InputLabel id="ai-task-select-label" sx={{ color: "var(--text-primary)" }}>
                  {t("sidebar.aiTask")}
                </InputLabel>
                <Select
                  labelId="ai-task-select-label"
                  id="ai-task-select"
                  value={aiTask}
                  label={t("sidebar.aiTask")}
                  onChange={(e) =>
                    onAiTaskChange(
                      e.target.value as
                        | "white-balance"
                        | "object-insert"
                        | "object-removal"
                        | "evaluation"
                    )
                  }
                  sx={{
                    color: "var(--text-primary)",
                    "& .MuiOutlinedInput-notchedOutline": {
                      borderColor: "var(--primary-accent)",
                    },
                    "&:hover .MuiOutlinedInput-notchedOutline": {
                      borderColor: "var(--primary-accent)",
                    },
                    "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                      borderColor: "var(--primary-accent)",
                    },
                    "& .MuiSelect-icon": {
                      color: "var(--text-primary)",
                    },
                    backgroundColor: "var(--primary-bg)",
                  }}
                >
                  <MenuItem value="white-balance">
                    {t("sidebar.whiteBalance")}
                  </MenuItem>
                  <MenuItem value="object-insert">
                    {t("sidebar.objectInsert")}
                  </MenuItem>
                  <MenuItem value="object-removal">
                    {t("sidebar.objectRemoval")}
                  </MenuItem>
                </Select>
              </FormControl>
            </div>
            </>
          )}

          {/* Reference Image Upload (only for object insert) */}
          {!isEditingDone && aiTask === "object-insert" && (
            <>
              <Divider sx={{ my: 2, borderColor: "var(--border-color)" }} />
              <Box sx={{ pb: 2 }}>
                <Typography
                  variant="subtitle2"
                  sx={{
                    mb: 2,
                    color: "var(--text-primary)",
                    fontWeight: 500,
                    fontSize: { xs: "0.875rem", sm: "1rem" },
                  }}
                >
                  {t("sidebar.referenceImage")}
                </Typography>
                <Stack spacing={1.5}>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg,image/webp"
                    onChange={onReferenceImageUpload}
                    onClick={(e) => {
                      // Reset value to allow re-selecting the same file
                      (e.target as HTMLInputElement).value = "";
                    }}
                    style={{ display: "none" }}
                    id="reference-image-upload"
                  />
                  <label htmlFor="reference-image-upload">
                    <Button
                      component="span"
                      fullWidth
                      variant="outlined"
                      startIcon={<ImageIconMUI />}
                      sx={{
                        bgcolor: "var(--secondary-bg)",
                        color: "var(--text-secondary)",
                        borderColor: "var(--primary-accent)",
                        borderWidth: 2,
                        borderStyle: "dashed",
                        "&:hover": {
                          bgcolor: "var(--primary-accent)",
                          color: "white",
                          borderColor: "var(--primary-accent)",
                          borderStyle: "solid",
                        },
                        py: 1.5,
                        textTransform: "none",
                        fontSize: "0.875rem",
                      }}
                    >
                      {t("sidebar.chooseReference")}
                    </Button>
                  </label>
                  {referenceImage && (
                    <Stack spacing={1}>
                      <Paper
                        sx={{
                          bgcolor: "var(--secondary-bg)",
                          border: 2,
                          borderColor: "var(--primary-accent)",
                          borderRadius: 2,
                          p: 1,
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          sx={{
                            position: "relative",
                            width: "100%",
                            aspectRatio: "1/1",
                            bgcolor: "var(--primary-bg)",
                            borderRadius: 1,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <Box
                            component="img"
                            src={referenceImage}
                            alt="Reference"
                            sx={{
                              maxWidth: "100%",
                              maxHeight: "100%",
                              objectFit: "contain",
                              borderRadius: 1,
                            }}
                          />
                        </Box>
                      </Paper>
                      <Stack direction="row" spacing={1}>
                        {onEditReferenceImage && (
                          <Button
                            fullWidth
                            variant="contained"
                            onClick={onEditReferenceImage}
                            sx={{
                              bgcolor: "var(--primary-accent)",
                              color: "white",
                              "&:hover": {
                                bgcolor: "var(--highlight-accent)",
                              },
                              py: 1.25,
                              textTransform: "none",
                              fontSize: "0.875rem",
                            }}
                          >
                            {t("sidebar.editReference")}
                          </Button>
                        )}
                        <Button
                          fullWidth
                          variant="contained"
                          color="error"
                          onClick={onRemoveReferenceImage}
                          sx={{
                            py: 1.25,
                            textTransform: "none",
                            fontSize: "0.875rem",
                          }}
                        >
                          {t("sidebar.removeReference")}
                        </Button>
                      </Stack>
                    </Stack>
                  )}
                </Stack>
              </Box>
            </>
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
              <>
                {/* Only show divider if there's content before this section */}
                {(uploadedImage || aiTask === "object-insert") && (
                  <Divider sx={{ my: 2, borderColor: "var(--border-color)" }} />
                )}
                <div className="pb-4">
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
                        
                        {/* Model Type Selection */}
                        {enableSmartMasking && onSmartMaskModelTypeChange && (
                          <div className="mt-3 space-y-2">
                            <label className="text-[var(--text-secondary)] text-sm block">
                              {t("sidebar.smartMaskModel")}
                            </label>
                            <div className="flex gap-2">
                              <button
                                onClick={() => {
                                  onSmartMaskModelTypeChange('segmentation');
                                  // Auto switch to brush if was on box and BiRefNet was selected
                                  if (maskToolType === 'box' && smartMaskModelType === 'birefnet') {
                                    onMaskToolTypeChange?.('brush');
                                  }
                                }}
                                className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                                  smartMaskModelType === 'segmentation'
                                    ? 'bg-[var(--primary-accent)] text-white'
                                    : 'bg-[var(--bg-secondary)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                                }`}
                                disabled={isSmartMaskLoading}
                              >
                                FastSAM
                              </button>
                              <button
                                onClick={() => {
                                  onSmartMaskModelTypeChange('birefnet');
                                  // Auto switch to box when selecting BiRefNet
                                  if (maskToolType === 'brush') {
                                    onMaskToolTypeChange?.('box');
                                  }
                                }}
                                className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                                  smartMaskModelType === 'birefnet'
                                    ? 'bg-[var(--primary-accent)] text-white'
                                    : 'bg-[var(--bg-secondary)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                                }`}
                                disabled={isSmartMaskLoading}
                              >
                                BiRefNet
                              </button>
                            </div>
                            {smartMaskModelType === 'birefnet' && (
                              <p className="text-xs text-[var(--text-secondary)] mt-1">
                                {t("sidebar.birefnetNote")}
                              </p>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Border Adjustment Control */}
                      {enableSmartMasking && onBorderAdjustmentChange && (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="text-[var(--text-secondary)] text-sm">
                              {t("sidebar.borderAdjustment")}
                            </label>
                            <input
                              type="number"
                              min="-10"
                              max="10"
                              value={borderAdjustment}
                              onChange={(e) => {
                                const value = parseInt(e.target.value);
                                if (!isNaN(value)) {
                                  onBorderAdjustmentChange(value);
                                }
                              }}
                              onBlur={(e) => {
                                const value = parseInt(e.target.value);
                                if (isNaN(value) || value < -10) {
                                  onBorderAdjustmentChange(-10);
                                } else if (value > 10) {
                                  onBorderAdjustmentChange(10);
                                }
                              }}
                              className="w-16 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                            />
                          </div>
                          <Box sx={{ px: 1 }}>
                            <Slider
                              value={borderAdjustment}
                              onChange={(_, value) =>
                                onBorderAdjustmentChange(value as number)
                              }
                              min={-10}
                              max={10}
                              step={1}
                              valueLabelDisplay="auto"
                              valueLabelFormat={(value) =>
                                value === 0
                                  ? t("sidebar.noAdjustment")
                                  : value < 0
                                  ? `${t("sidebar.shrink")} ${Math.abs(value)}px`
                                  : `${t("sidebar.grow")} ${value}px`
                              }
                              sx={{
                                color: "var(--primary-accent)",
                                "& .MuiSlider-thumb": {
                                  backgroundColor: "var(--primary-accent)",
                                  "&:hover": {
                                    boxShadow: "0 0 0 8px rgba(0, 0, 0, 0.16)",
                                  },
                                },
                                "& .MuiSlider-track": {
                                  backgroundColor: "var(--primary-accent)",
                                },
                                "& .MuiSlider-rail": {
                                  backgroundColor: "var(--border-color)",
                                },
                              }}
                            />
                          </Box>
                          <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                            <span>{t("sidebar.shrink")}</span>
                            <span>{t("sidebar.grow")}</span>
                          </div>
                        </div>
                      )}

                      <div>
                        <label className="block text-[var(--text-secondary)] text-sm mb-2">
                          {t("sidebar.maskTool")}
                        </label>
                        <div className="flex gap-2 mb-3">
                          <button
                            onClick={() => onMaskToolTypeChange?.("brush")}
                            disabled={enableSmartMasking && smartMaskModelType === 'birefnet'}
                            className={`flex-1 px-3 py-2 text-sm rounded border transition-colors ${
                              maskToolType === "brush"
                                ? "bg-[var(--primary-accent)] text-white border-[var(--primary-accent)]"
                                : "bg-[var(--secondary-bg)] text-[var(--text-primary)] border-[var(--border-color)] hover:bg-[var(--hover-bg)]"
                            } ${
                              enableSmartMasking && smartMaskModelType === 'birefnet'
                                ? "opacity-50 cursor-not-allowed"
                                : ""
                            }`}
                            title={enableSmartMasking && smartMaskModelType === 'birefnet' ? t("sidebar.birefnetBrushDisabled") : undefined}
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
                              max="100"
                              value={maskBrushSize}
                              onChange={(e) => {
                                const value = parseInt(e.target.value);
                                if (!isNaN(value)) {
                                  onMaskBrushSizeChange(value);
                                }
                              }}
                              onBlur={(e) => {
                                const value = parseInt(e.target.value);
                                if (isNaN(value) || value < 1) {
                                  onMaskBrushSizeChange(1);
                                } else if (value > 100) {
                                  onMaskBrushSizeChange(100);
                                }
                              }}
                              className="w-16 px-2 py-1 text-sm bg-[var(--secondary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                            />
                            <span className="text-xs text-[var(--text-secondary)] mr-1">
                              px
                            </span>
                            <Box sx={{ flex: 1, px: 1 }}>
                              <Slider
                                value={maskBrushSize}
                                onChange={(_, value) =>
                                  onMaskBrushSizeChange(value as number)
                                }
                                min={1}
                                max={100}
                                step={1}
                                valueLabelDisplay="auto"
                                sx={{
                                  color: "var(--primary-accent)",
                                  "& .MuiSlider-thumb": {
                                    backgroundColor: "var(--primary-accent)",
                                    "&:hover": {
                                      boxShadow: "0 0 0 8px rgba(var(--primary-accent-rgb), 0.16)",
                                    },
                                  },
                                  "& .MuiSlider-track": {
                                    backgroundColor: "var(--primary-accent)",
                                  },
                                  "& .MuiSlider-rail": {
                                    backgroundColor: "var(--border-color)",
                                  },
                                }}
                              />
                            </Box>
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
                      : t("sidebar.enableMaskingRemoval")                    }
                  </p>
                </div>
              </div>
            </>
            )}

          {/* Advanced Settings */}
          {!isEditingDone && (
            <div className="pt-2">
              <h3 className="text-[var(--text-primary)] font-medium mb-4">
                {t("sidebar.advanced")}
              </h3>
              <div className="space-y-6">
                {/* Negative Prompt with Toggle */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t("sidebar.negativePrompt")}
                    </label>
                    <button
                      onClick={() => {
                        const newEnabled = !isNegativePromptEnabled;
                        setIsNegativePromptEnabled(newEnabled);
                        if (!newEnabled) {
                          onNegativePromptChange?.("");
                        }
                      }}
                      className={`relative w-10 h-5 rounded-full transition-colors ${
                        isNegativePromptEnabled
                          ? "bg-[var(--primary-accent)]"
                          : "bg-[var(--border-color)]"
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                          isNegativePromptEnabled
                            ? "translate-x-5"
                            : "translate-x-0"
                        }`}
                      />
                    </button>
                  </div>
                  {isNegativePromptEnabled && (
                    <textarea
                      placeholder={t("sidebar.negativePromptPlaceholder")}
                      rows={2}
                      value={negativePrompt}
                      onChange={(e) => onNegativePromptChange?.(e.target.value)}
                      className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--primary-accent)] focus:border-transparent"
                    />
                  )}
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
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={guidanceScale}
                      onChange={(_, value) =>
                        onGuidanceScaleChange?.(value as number)
                      }
                      min={1}
                      max={10}
                      step={0.5}
                      valueLabelDisplay="auto"
                      sx={{
                        color: "var(--primary-accent)",
                        "& .MuiSlider-thumb": {
                          backgroundColor: "var(--primary-accent)",
                          "&:hover": {
                            boxShadow: "0 0 0 8px rgba(var(--primary-accent-rgb), 0.16)",
                          },
                        },
                        "& .MuiSlider-track": {
                          backgroundColor: "var(--primary-accent)",
                        },
                        "& .MuiSlider-rail": {
                          backgroundColor: "var(--border-color)",
                        },
                      }}
                    />
                  </Box>
                </div>

                {/* Steps (Updated from existing) */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t("sidebar.inferenceSteps")}
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={inferenceSteps}
                      onChange={(e) => {
                        const value = parseInt(e.target.value);
                        if (!isNaN(value)) {
                          onInferenceStepsChange?.(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value);
                        if (isNaN(value) || value < 1) {
                          onInferenceStepsChange?.(1);
                        } else if (value > 100) {
                          onInferenceStepsChange?.(100);
                        }
                      }}
                      className="w-16 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                    />
                  </div>
                  <Box sx={{ px: 1 }}>
                    <Slider
                      value={inferenceSteps}
                      onChange={(_, value) =>
                        onInferenceStepsChange?.(value as number)
                      }
                      min={1}
                      max={100}
                      step={1}
                      valueLabelDisplay="auto"
                      sx={{
                        color: "var(--primary-accent)",
                        "& .MuiSlider-thumb": {
                          backgroundColor: "var(--primary-accent)",
                          "&:hover": {
                            boxShadow: "0 0 0 8px rgba(var(--primary-accent-rgb), 0.16)",
                          },
                        },
                        "& .MuiSlider-track": {
                          backgroundColor: "var(--primary-accent)",
                        },
                        "& .MuiSlider-rail": {
                          backgroundColor: "var(--border-color)",
                        },
                      }}
                    />
                  </Box>
                  <div className="flex justify-between items-center text-xs text-[var(--text-secondary)] mt-2 px-1">
                    <span>{t("sidebar.faster")}</span>
                    <span>{t("sidebar.higherQuality")}</span>
                  </div>
                </div>

                {/* True CFG Scale - only show when negative prompt has value */}
                {negativePrompt.trim().length > 0 && (
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
                          if (!isNaN(value)) {
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
                    <Box sx={{ px: 1 }}>
                      <Slider
                        value={cfgScale}
                        onChange={(_, value) =>
                          onCfgScaleChange?.(value as number)
                        }
                        min={1}
                        max={5}
                        step={0.1}
                        valueLabelDisplay="auto"
                        sx={{
                          color: "var(--primary-accent)",
                          "& .MuiSlider-thumb": {
                            backgroundColor: "var(--primary-accent)",
                            "&:hover": {
                              boxShadow: "0 0 0 8px rgba(var(--primary-accent-rgb), 0.16)",
                            },
                          },
                          "& .MuiSlider-track": {
                            backgroundColor: "var(--primary-accent)",
                          },
                          "& .MuiSlider-rail": {
                            backgroundColor: "var(--border-color)",
                          },
                        }}
                      />
                    </Box>
                    <p className="text-xs text-[var(--text-secondary)] mt-2">
                      {t("sidebar.cfgGuidanceDescription")}
                    </p>
                  </div>
                )}

                {/* Seed */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[var(--text-secondary)] text-sm">
                      {t("sidebar.seed")}
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      step="1"
                      value={seed ?? 42}
                      onChange={(e) => {
                        const value = parseInt(e.target.value);
                        if (!isNaN(value) && value >= 0) {
                          onSeedChange?.(value);
                        }
                      }}
                      onBlur={(e) => {
                        const value = parseInt(e.target.value);
                        if (isNaN(value) || value < 0) {
                          onSeedChange?.(42); // Reset to default 42 if invalid
                        }
                      }}
                      className="w-24 px-2 py-1 text-sm bg-[var(--primary-bg)] border border-[var(--border-color)] rounded text-[var(--text-primary)] focus:outline-none focus:border-[var(--primary-accent)]"
                    />
                  </div>
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    {t("sidebar.seedDescription")}
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
