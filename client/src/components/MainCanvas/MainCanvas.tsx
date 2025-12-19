import { useCallback, useEffect, useState, useRef } from "react";
import { Camera } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import Toolbox from "../Toolbox";
import StatusBar from "../StatusBar";
import ViewportLayer from "./layers/ViewportLayer";
import ImageContainerLayer from "./layers/ImageContainerLayer";
import ContentLayer from "./layers/ContentLayer";
import TransformLayer from "./layers/TransformLayer";
import ImageLayer from "./layers/ImageLayer";
import MaskCanvasLayer from "./layers/MaskCanvasLayer";
import EdgeOverlayLayer from "./layers/EdgeOverlayLayer";
import UIOverlayLayer from "./layers/UIOverlayLayer";
import LoadingOverlayLayer from "./layers/LoadingOverlayLayer";
import BrushPreviewLayer from "./layers/BrushPreviewLayer";

interface CanvasProps {
  // Image state
  uploadedImage: string | null;
  imageDimensions: { width: number; height: number } | null;
  displayScale: number;
  transform: { scale: number };

  // Viewport state
  viewportZoom: number;

  // Masking state
  isMaskingMode: boolean;
  maskBrushSize: number;
  maskToolType?: "brush" | "box" | "eraser";
  maskVisible?: boolean;
  isSmartMaskLoading?: boolean;
  hasMaskContent?: boolean;

  // Toolbox state
  originalImage: string | null;
  modifiedImage: string | null;
  comparisonSlider: number;
  historyIndex: number;
  historyStackLength: number;
  isHelpOpen: boolean;

  // Refs (passed from parent)
  imageContainerRef: React.RefObject<HTMLDivElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  maskCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  edgeOverlayCanvasRef?: React.RefObject<HTMLCanvasElement | null>;
  enableEdgeDetection?: boolean;
  imageRef: React.RefObject<HTMLImageElement | null>;

  // Event handlers
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onImageClick: (e: React.MouseEvent) => void;
  onWheel: (e: React.WheelEvent) => void;

  // Masking handlers
  onMaskMouseDown: (e: React.MouseEvent) => void;
  onMaskMouseMove: (e: React.MouseEvent) => void;
  onMaskMouseUp: () => void;

  // Toolbox handlers
  onComparisonSliderChange: (value: number) => void;
  onUndo: () => void;
  onRedo: () => void;
  onDownload: () => void;
  onZoomViewportIn: () => void;
  onZoomViewportOut: () => void;
  onResetViewportZoom: () => void;
  onToggleHelp: () => void;
}

export default function Canvas({
  uploadedImage,
  imageDimensions,
  displayScale,
  transform,
  viewportZoom,
  isMaskingMode,
  maskBrushSize,
  maskToolType = "brush",
  maskVisible = true,
  isSmartMaskLoading = false,
  hasMaskContent = false,
  originalImage,
  modifiedImage,
  comparisonSlider,
  historyIndex,
  historyStackLength,
  isHelpOpen,
  imageContainerRef,
  containerRef,
  maskCanvasRef,
  edgeOverlayCanvasRef,
  enableEdgeDetection = false,
  imageRef,
  onImageUpload,
  onImageClick,
  onWheel,
  onMaskMouseDown,
  onMaskMouseMove,
  onMaskMouseUp,
  onComparisonSliderChange,
  onUndo,
  onRedo,
  onDownload,
  onZoomViewportIn,
  onZoomViewportOut,
  onResetViewportZoom,
  onToggleHelp,
}: CanvasProps) {
  const { t } = useLanguage();

  // Giữ API props nhưng hiện tại chưa dùng trực tiếp trong component
  void historyIndex;
  void historyStackLength;
  void onUndo;
  void onRedo;
  void onDownload;

  const [isDraggingSeparator, setIsDraggingSeparator] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // --- LOGIC KÉO THẢ THANH DIVIDER ---

  const handleSeparatorMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation(); // Ngăn chặn sự kiện click lan ra ngoài
    setIsDraggingSeparator(true);
    document.body.style.userSelect = "none"; // Tắt bôi đen text khi kéo
  }, []);

  const handleSeparatorMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDraggingSeparator || !imageContainerRef.current) return;

      const container = imageContainerRef.current;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;

      // Tính phần trăm vị trí chuột trong container (0 -> 100)
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

      onComparisonSliderChange(percentage);
    },
    [isDraggingSeparator, imageContainerRef, onComparisonSliderChange]
  );

  const handleSeparatorMouseUp = useCallback(() => {
    setIsDraggingSeparator(false);
    document.body.style.userSelect = "";
  }, []);

  // --- LOGIC CẢM ỨNG (TOUCH) CHO MOBILE ---

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    // Ngăn chặn scroll màn hình khi chạm vào thanh slider
    if (e.cancelable) e.preventDefault();

    setIsDraggingSeparator(true);
  }, []);

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (!isDraggingSeparator || !imageContainerRef.current) return;

      // Logic tính toán tương tự MouseMove nhưng dùng e.touches
      const container = imageContainerRef.current;
      const rect = container.getBoundingClientRect();
      const x = e.touches[0].clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));

      onComparisonSliderChange(percentage);
    },
    [isDraggingSeparator, imageContainerRef, onComparisonSliderChange]
  );

  const handleTouchEnd = useCallback(() => {
    setIsDraggingSeparator(false);
  }, []);

  // --- GLOBAL EVENTS ---
  // Thêm event listener vào document để kéo mượt mà ngay cả khi chuột ra khỏi khung ảnh
  useEffect(() => {
    if (isDraggingSeparator) {
      document.addEventListener("mousemove", handleSeparatorMouseMove);
      document.addEventListener("mouseup", handleSeparatorMouseUp);
      document.body.style.cursor = "col-resize";

      return () => {
        document.removeEventListener("mousemove", handleSeparatorMouseMove);
        document.removeEventListener("mouseup", handleSeparatorMouseUp);
        document.body.style.cursor = "";
      };
    }
  }, [isDraggingSeparator, handleSeparatorMouseMove, handleSeparatorMouseUp]);

  // Xử lý phím tắt (Mũi tên trái/phải để di chuyển slider)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (!originalImage || !modifiedImage) return;
      let increment = 0;
      if (e.key === "ArrowLeft") increment = -5;
      if (e.key === "ArrowRight") increment = 5;

      if (increment !== 0) {
        e.preventDefault();
        onComparisonSliderChange(
          Math.max(0, Math.min(100, comparisonSlider + increment))
        );
      }
    },
    [originalImage, modifiedImage, comparisonSlider, onComparisonSliderChange]
  );

  // Biến kiểm tra có đang ở chế độ so sánh không
  const isComparisonMode =
    originalImage && modifiedImage && originalImage !== modifiedImage;

  // --- DRAG AND DROP HANDLERS ---
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Chỉ cho phép drag over khi không có ảnh hoặc không ở comparison mode
    if (!uploadedImage || !isComparisonMode) {
      setIsDragOver(true);
    }
  }, [uploadedImage, isComparisonMode]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Chỉ set false nếu không còn element nào trong drag area
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragOver(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    // Chỉ xử lý khi không có ảnh hoặc không ở comparison mode
    if (uploadedImage && isComparisonMode) {
      return;
    }

    const files = e.dataTransfer.files;
    if (files.length === 0) return;

    const file = files[0];
    
    // Kiểm tra file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      console.warn('Invalid file type:', file.type);
      return;
    }

    // Tạo synthetic event để tái sử dụng logic upload hiện có
    const syntheticEvent = {
      target: {
        files: [file],
      },
    } as unknown as React.ChangeEvent<HTMLInputElement>;

    onImageUpload(syntheticEvent);
  }, [uploadedImage, isComparisonMode, onImageUpload]);

  return (
    <div
      ref={containerRef}
      className="flex-1 relative min-w-0 overflow-hidden flex items-center justify-center bg-primary-bg dots-pattern"
      onWheel={onWheel}
      style={{ cursor: "default", padding: "2rem" }}
    >
      {/* Status Bar */}
      <StatusBar
        imageDimensions={imageDimensions}
        uploadedImage={uploadedImage}
        displayScale={displayScale}
        viewportZoom={viewportZoom}
        isMaskingMode={isMaskingMode}
        transform={transform}
      />

      {/* Main Image Container - Sử dụng Layer Architecture */}
        <ViewportLayer viewportZoom={viewportZoom}>
            <ImageContainerLayer
            ref={imageContainerRef}
            imageDimensions={imageDimensions}
            displayScale={displayScale}
              cursor={
                isSmartMaskLoading
                  ? "wait"
                  : isComparisonMode
                  ? "col-resize"
                  : isMaskingMode &&
                    (maskToolType === "brush" || maskToolType === "eraser")
                  ? "none"
                  : isMaskingMode
                  ? "crosshair"
                  : "default"
              }
            onMouseDown={
              !isComparisonMode && isMaskingMode && !isSmartMaskLoading
                ? onMaskMouseDown
                : undefined
            }
            onMouseMove={
              !isComparisonMode && isMaskingMode && !isSmartMaskLoading
                ? onMaskMouseMove
                : undefined
            }
            onMouseUp={
              !isComparisonMode && isMaskingMode && !isSmartMaskLoading
                ? onMaskMouseUp
                : undefined
            }
            // Do not stop stroke on mouse leave – we handle it via global mouseup
            onMouseLeave={undefined}
            onClick={
              !isComparisonMode && !isMaskingMode ? onImageClick : undefined
            }
            onKeyDown={isComparisonMode ? handleKeyDown : undefined}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            isDragOver={isDragOver}
            tabIndex={isComparisonMode ? 0 : undefined}
          >
            {uploadedImage ? (
              <ContentLayer
                imageDimensions={imageDimensions}
                displayScale={displayScale}
              >
                <TransformLayer
                  imageDimensions={imageDimensions}
                  displayScale={displayScale}
                >
                  <ImageLayer
                    mode={isComparisonMode ? "comparison" : "single"}
                    uploadedImage={uploadedImage}
                    originalImage={originalImage}
                    modifiedImage={modifiedImage}
                    imageDimensions={imageDimensions}
                    displayScale={displayScale}
                    comparisonSlider={comparisonSlider}
                    isDraggingSeparator={isDraggingSeparator}
                    imageRef={imageRef}
                  />
                  <MaskCanvasLayer
                    isMaskingMode={isMaskingMode}
                    maskCanvasRef={maskCanvasRef}
                    imageDimensions={imageDimensions}
                    displayScale={displayScale}
                    maskVisible={maskVisible}
                    hasMaskContent={hasMaskContent}
                  />
                  <BrushPreviewLayer
                    isMaskingMode={isMaskingMode}
                    maskBrushSize={maskBrushSize}
                    imageDimensions={imageDimensions}
                    displayScale={displayScale}
                    transform={transform}
                    viewportZoom={viewportZoom}
                    imageContainerRef={imageContainerRef}
                    maskCanvasRef={maskCanvasRef}
                    maskToolType={maskToolType}
                  />
                  {edgeOverlayCanvasRef && (
                    <EdgeOverlayLayer
                      isMaskingMode={isMaskingMode}
                      enableEdgeDetection={enableEdgeDetection}
                      edgeOverlayCanvasRef={edgeOverlayCanvasRef}
                      imageDimensions={imageDimensions}
                      displayScale={displayScale}
                    />
                  )}
                </TransformLayer>
                <UIOverlayLayer
                  isComparisonMode={!!isComparisonMode}
                  isMaskingMode={isMaskingMode}
                  imageDimensions={imageDimensions}
                  displayScale={displayScale}
                  comparisonSlider={comparisonSlider}
                  onSeparatorMouseDown={handleSeparatorMouseDown}
                  onSeparatorTouchStart={handleTouchStart}
                  onSeparatorTouchMove={handleTouchMove}
                  onSeparatorTouchEnd={handleTouchEnd}
                />
                <LoadingOverlayLayer
                  isLoading={isSmartMaskLoading}
                  imageDimensions={imageDimensions}
                  displayScale={displayScale}
                />
              </ContentLayer>
            ) : (
              // Upload Placeholder - Chỉ có nút upload
              <div
                className={`w-full h-full flex items-center justify-center transition-all duration-200 ${
                  isDragOver ? "scale-105" : ""
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  onChange={onImageUpload}
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={`btn-interactive btn-primary-hover inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3 text-base font-semibold text-white transition-all duration-200 ${
                    isDragOver
                      ? "bg-highlight-accent scale-[1.015]"
                      : "bg-primary-accent hover:bg-highlight-accent"
                  } shadow-lg`}
                >
                  <Camera size={20} />
                  {isDragOver ? `${t("sidebar.chooseImage")} (Drop here)` : t("sidebar.chooseImage")}
                </button>
              </div>
            )}

          </ImageContainerLayer>
        </ViewportLayer>

      {/* Toolbox */}
      <Toolbox
        uploadedImage={uploadedImage}
        originalImage={originalImage}
        modifiedImage={modifiedImage}
        comparisonSlider={comparisonSlider}
        onComparisonSliderChange={onComparisonSliderChange}
        viewportZoom={viewportZoom}
        onZoomViewportIn={onZoomViewportIn}
        onZoomViewportOut={onZoomViewportOut}
        onResetViewportZoom={onResetViewportZoom}
        isHelpOpen={isHelpOpen}
        onToggleHelp={onToggleHelp}
      />
    </div>
  );
}
