"use client";

import { useState, useEffect } from "react";
import {
  ChevronDown,
  ChevronUp,
  Bug,
  X,
  ZoomIn,
  Image as ImageIcon,
  Info,
  Download,
  Loader2,
} from "lucide-react";
import type { DebugInfo } from "@/services/api";

interface DebugPanelProps {
  debugInfo: DebugInfo | null;
  isVisible: boolean;
  onClose?: () => void; // Optional since debug is controlled by settings
  onOpen?: () => void;
}

type TabType = "images" | "info";

export default function DebugPanel({
  debugInfo,
  isVisible,
  onOpen,
}: DebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  type SelectedImage = { src: string; label: string };
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>("images");
  const [downloadingImages, setDownloadingImages] = useState<Set<string>>(new Set()); // Track which images are downloading
  const [isDownloadingMetadata, setIsDownloadingMetadata] = useState(false);

  // Note: We no longer fetch session details from API endpoint
  // All debug info is already available in debugInfo from generation response
  // This avoids 404 errors since debug sessions are container-local to H200 workers
  // and not accessible from Job Manager Service

  // Download individual image
  const handleDownloadImage = (imageName: string, label: string, base64Data?: string) => {
    if (!base64Data) {
      alert("Image data not available for download");
      return;
    }
    if (downloadingImages.has(imageName)) {
      return;
    }
    setDownloadingImages((prev) => new Set(prev).add(imageName));
    try {
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "image/png" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${label || imageName}.png`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 100);
    } catch (error) {
      console.error(`❌ [DebugPanel] Failed to download image ${imageName}:`, error);
      alert(
        `Failed to download image: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setDownloadingImages((prev) => {
        const next = new Set(prev);
        next.delete(imageName);
        return next;
      });
    }
  };

  // Download metadata as JSON (frontend-only, no backend call)
  // Uses only debugInfo from generation response (no API call needed)
  const handleDownloadMetadata = () => {
    setIsDownloadingMetadata(true);
    try {
      const metadata = {
        session_name: debugInfo?.session_name ?? "debug_session",
        debug_path: debugInfo?.debug_path,
        debug_info: debugInfo
          ? {
              original_prompt: debugInfo.original_prompt,
              refined_prompt: debugInfo.refined_prompt,
              prompt_was_refined: debugInfo.prompt_was_refined,
              input_image_size: debugInfo.input_image_size,
              output_image_size: debugInfo.output_image_size,
              lora_adapter: debugInfo.lora_adapter,
              loaded_adapters: debugInfo.loaded_adapters,
              conditional_labels: debugInfo.conditional_labels,
              conditional_images_count: debugInfo.conditional_images?.length || 0,
              has_original_image: !!debugInfo.original_image,
              has_mask_A: !!debugInfo.mask_A,
              has_reference_image: !!debugInfo.reference_image,
              has_reference_mask_R: !!debugInfo.reference_mask_R,
              has_positioned_mask_R: !!debugInfo.positioned_mask_R,
              has_mask_mae_dilated: !!debugInfo.mask_mae_dilated,
            }
          : undefined,
      };

      const jsonContent = JSON.stringify(metadata, null, 2);
      const blob = new Blob([jsonContent], { type: "application/json" });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${metadata.session_name || "debug_session"}_metadata.json`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 100);
    } catch (error) {
      console.error(`❌ [DebugPanel] Failed to download metadata:`, error);
      alert(
        `Failed to download metadata: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setIsDownloadingMetadata(false);
    }
  };

  // Auto-show panel when debug info is available
  useEffect(() => {
    if (debugInfo && !isVisible && onOpen) {
      onOpen();
    }
  }, [debugInfo, isVisible, onOpen]);

  // Show minimal toggle button when panel is closed but has debug info
  if (!isVisible && debugInfo && onOpen) {
    return (
      <button
        onClick={onOpen}
        className="fixed bottom-4 right-4 z-50 bg-zinc-800/80 hover:bg-zinc-700/80 text-amber-400/80 hover:text-amber-400 p-2 rounded-lg shadow-lg transition-all border border-zinc-700/50 hover:border-amber-500/50"
        title="Debug Info"
      >
        <Bug size={14} />
      </button>
    );
  }

  if (!isVisible || !debugInfo) return null;

  const conditionalImages = debugInfo.conditional_images || [];
  // Default labels - must match backend order defined in GenerationService:
  // - White-balance: [input, canny]
  // - Removal: [masked_bg, mask, mae]
  // - Insertion (standard): [ref_img, mask, masked_bg]
  // - Insertion (reference-guided): [original, mask_A, masked_R, reference_image, positioned_mask_R]
  const labels = debugInfo.conditional_labels || [
    "ref_img",
    "mask",
    "masked_bg",
    "mae",
  ];

  // Additional debug images from backend
  const debugImages: Array<{ label: string; data?: string }> = [
    { label: "original_image", data: debugInfo.original_image },
    { label: "mask_A", data: debugInfo.mask_A },
    { label: "reference_image", data: debugInfo.reference_image },
    { label: "reference_mask_R", data: debugInfo.reference_mask_R },
    { label: "positioned_mask_R", data: debugInfo.positioned_mask_R },
    { label: "mask_mae_dilated", data: debugInfo.mask_mae_dilated },
  ].filter((item) => !!item.data);
  const debugLabels = new Set(debugImages.map((d) => d.label));

  return (
    <>
      {/* Debug Panel - Minimal UI */}
      <div className="fixed bottom-4 right-4 z-50 bg-zinc-900/95 backdrop-blur-sm border border-zinc-700/50 rounded-lg shadow-xl max-w-sm overflow-hidden">
        {/* Minimal Header */}
        <div className="flex items-center justify-between px-3 py-1.5 bg-zinc-800/80">
          <div className="flex items-center gap-1.5 text-amber-400/90">
            <Bug size={12} />
            <span className="text-xs font-medium">Debug</span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-zinc-400 hover:text-zinc-200 transition-colors p-0.5"
            >
              {isExpanded ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
            </button>
          </div>
        </div>

        {/* Content */}
        {isExpanded && (
          <div className="bg-zinc-900/95">
            {/* Tabs */}
            <div className="flex border-b border-zinc-700/50">
              <button
                onClick={() => setActiveTab("images")}
                className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors flex items-center justify-center gap-1 ${
                  activeTab === "images"
                    ? "text-amber-400 border-b-2 border-amber-400 bg-zinc-800/50"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <ImageIcon size={12} />
                Images
              </button>
              <button
                onClick={() => setActiveTab("info")}
                className={`flex-1 px-3 py-1.5 text-xs font-medium transition-colors flex items-center justify-center gap-1 ${
                  activeTab === "info"
                    ? "text-amber-400 border-b-2 border-amber-400 bg-zinc-800/50"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                <Info size={12} />
                Info
              </button>
            </div>

            {/* Tab Content */}
            <div className="p-3 max-h-[50vh] overflow-y-auto">
              {activeTab === "images" && (
                <div className="space-y-3">
                  {/* Debug images row */}
                  {debugImages.length > 0 && (
                    <div>
                      <div className="text-[10px] text-zinc-400 mb-1.5">Debug images</div>
                      <div className="grid grid-cols-4 gap-1.5">
                        {debugImages.map((item, idx) => {
                          const label = item.label;
                          const imageName = `${label}.png`;
                          const isDownloading = downloadingImages.has(imageName);
                          return (
                            <div key={idx} className="relative group">
                              <div
                                className="cursor-pointer"
                                onClick={() =>
                                  setSelectedImage({
                                    src: `data:image/png;base64,${item.data}`,
                                    label,
                                  })
                                }
                              >
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={`data:image/png;base64,${item.data}`}
                                  alt={label}
                                  className="w-full aspect-square object-cover rounded border border-zinc-600 hover:border-amber-500 transition-colors"
                                />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                                  <ZoomIn size={12} className="text-white" />
                                </div>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDownloadImage(imageName, label, item.data);
                                }}
                                disabled={isDownloading}
                                className="absolute top-1 right-1 p-1 bg-zinc-800/90 hover:bg-zinc-700/90 text-zinc-400 hover:text-amber-400 rounded transition-colors disabled:opacity-50"
                                title={`Download ${label}`}
                              >
                                {isDownloading ? (
                                  <Loader2 size={10} className="animate-spin" />
                                ) : (
                                  <Download size={10} />
                                )}
                              </button>
                              <div className="text-[9px] text-center text-zinc-400 mt-0.5 truncate px-0.5">
                                {label}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Conditional images row */}
                  {conditionalImages.length > 0 ? (
                    <>
                      <div className="text-[10px] text-zinc-400 mb-1.5">Conditional images</div>
                      <div className="grid grid-cols-4 gap-1.5">
                        {conditionalImages.map((img, index) => {
                          const label = labels[index] || `Condition ${index + 1}`;
                          // Skip if label overlaps with debug images to avoid duplicates
                          if (debugLabels.has(label)) {
                            return null;
                          }
                          const imageName = `${label}.png`;
                          const isDownloading = downloadingImages.has(imageName);
                          
                          return (
                            <div
                              key={index}
                              className="relative group"
                            >
                              <div
                                className="cursor-pointer"
                                onClick={() =>
                                  setSelectedImage({
                                    src: `data:image/png;base64,${img}`,
                                    label,
                                  })
                                }
                              >
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={`data:image/png;base64,${img}`}
                                  alt={label}
                                  className="w-full aspect-square object-cover rounded border border-zinc-600 hover:border-amber-500 transition-colors"
                                />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                                  <ZoomIn size={12} className="text-white" />
                                </div>
                              </div>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDownloadImage(imageName, label, img);
                                }}
                                disabled={isDownloading}
                                className="absolute top-1 right-1 p-1 bg-zinc-800/90 hover:bg-zinc-700/90 text-zinc-400 hover:text-amber-400 rounded transition-colors disabled:opacity-50"
                                title={`Download ${label}`}
                              >
                                {isDownloading ? (
                                  <Loader2 size={10} className="animate-spin" />
                                ) : (
                                  <Download size={10} />
                                )}
                              </button>
                              <div className="text-[9px] text-center text-zinc-400 mt-0.5 truncate px-0.5">
                                {label}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : (
                    <div className="text-xs text-zinc-500 text-center py-4">
                      No conditional images
                    </div>
                  )}
                </div>
              )}

              {activeTab === "info" && (
                <div className="space-y-3">
                  {/* Prompt Info */}
                  {debugInfo.original_prompt && (
                    <div className="space-y-1.5">
                      <div className="text-[10px] text-zinc-400 font-medium">
                        Prompt:
                      </div>
                      <div className="bg-zinc-800/50 rounded p-2 text-[10px]">
                        {debugInfo.prompt_was_refined ? (
                          <>
                            <div className="text-zinc-500 mb-1">
                              <span className="text-zinc-400">Original:</span>{" "}
                              {debugInfo.original_prompt}
                            </div>
                            <div className="text-emerald-400/90">
                              <span className="text-zinc-400">Refined ✨:</span>{" "}
                              {debugInfo.refined_prompt}
                            </div>
                          </>
                        ) : (
                          <div className="text-zinc-200">
                            {debugInfo.original_prompt}
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Technical Info */}
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px]">
                    <div className="text-zinc-400">Input Size:</div>
                    <div className="text-zinc-200 font-mono text-right">
                      {debugInfo.input_image_size || "N/A"}
                    </div>

                    <div className="text-zinc-400">Output Size:</div>
                    <div className="text-zinc-200 font-mono text-right">
                      {debugInfo.output_image_size || "N/A"}
                    </div>

                    <div className="text-zinc-400">LoRA:</div>
                    <div
                      className="text-zinc-200 font-mono text-right text-[9px] truncate"
                      title={debugInfo.lora_adapter || "None"}
                    >
                      {debugInfo.lora_adapter || "None"}
                    </div>

                    {debugInfo.loaded_adapters &&
                      debugInfo.loaded_adapters.length > 0 && (
                        <>
                          <div className="text-zinc-400 col-span-2">
                            Loaded:
                          </div>
                          <div className="text-zinc-200 font-mono text-[9px] col-span-2 text-right">
                            {debugInfo.loaded_adapters.join(", ")}
                          </div>
                        </>
                      )}
                  </div>

                  {/* Download Metadata Button (frontend-only) */}
                  <div className="pt-2 border-t border-zinc-700/50">
                    <button
                      onClick={handleDownloadMetadata}
                      disabled={isDownloadingMetadata}
                      className="w-full px-3 py-2 bg-zinc-800/50 hover:bg-zinc-700/50 text-zinc-300 hover:text-amber-400 transition-colors rounded text-xs font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed border border-zinc-700/50"
                      title={
                        isDownloadingMetadata
                          ? "Downloading metadata..."
                          : "Download metadata as JSON"
                      }
                    >
                      {isDownloadingMetadata ? (
                        <>
                          <Loader2 size={14} className="animate-spin" />
                          <span>Downloading...</span>
                        </>
                      ) : (
                        <>
                          <Download size={14} />
                          <span>Download Metadata (JSON)</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Fullscreen Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-60 bg-black/90 flex items-center justify-center p-8"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute -top-10 right-0 text-white hover:text-amber-400 transition-colors"
            >
              <X size={24} />
            </button>
            <div className="text-center mb-2 text-white font-medium">
              {selectedImage.label}
            </div>
            
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={selectedImage.src}
              alt={selectedImage.label}
              className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl border-2 border-amber-500/70"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </>
  );
}
