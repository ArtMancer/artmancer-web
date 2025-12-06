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
} from "lucide-react";
import type { DebugInfo } from "@/services/api";
import { apiService } from "@/services/api";

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
  const [selectedImage, setSelectedImage] = useState<{
    src: string;
    label: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>("images");
  const [isDownloading, setIsDownloading] = useState(false);

  // Extract session name from debug_path or use session_name
  const getSessionName = (): string | null => {
    if (debugInfo?.session_name) {
      return debugInfo.session_name;
    }
    if (debugInfo?.debug_path) {
      // Extract session name from path (e.g., "debug_output/20241206_123456_abc12345" -> "20241206_123456_abc12345")
      const pathParts = debugInfo.debug_path.split("/");
      return pathParts[pathParts.length - 1] || null;
    }
    return null;
  };

  const sessionName = getSessionName();

  const handleDownload = async () => {
    if (!sessionName) {
      alert("Debug session name not available");
      return;
    }

    setIsDownloading(true);
    try {
      const blob = await apiService.downloadDebugSession(sessionName);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${sessionName}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download debug session:", error);
      alert(
        `Failed to download debug session: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    } finally {
      setIsDownloading(false);
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
  // Default labels - should match backend order:
  // - Insertion: [ref_img, mask, masked_bg]
  // - Removal: [original, mask, mae]
  const labels = debugInfo.conditional_labels || [
    "ref_img",
    "mask",
    "masked_bg",
    "mae",
  ];

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
            {sessionName && (
              <button
                onClick={handleDownload}
                disabled={isDownloading}
                className="text-zinc-400 hover:text-amber-400 transition-colors p-0.5 disabled:opacity-50"
                title="Download debug session"
              >
                <Download size={12} />
              </button>
            )}
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
                <div className="space-y-2">
                  {conditionalImages.length > 0 ? (
                    <div className="grid grid-cols-4 gap-1.5">
                      {conditionalImages.map((img, index) => (
                        <div
                          key={index}
                          className="relative group cursor-pointer"
                          onClick={() =>
                            setSelectedImage({
                              src: `data:image/png;base64,${img}`,
                              label: labels[index] || `Condition ${index + 1}`,
                            })
                          }
                        >
                          <img
                            src={`data:image/png;base64,${img}`}
                            alt={labels[index] || `Condition ${index + 1}`}
                            className="w-full aspect-square object-cover rounded border border-zinc-600 hover:border-amber-500 transition-colors"
                          />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded">
                            <ZoomIn size={12} className="text-white" />
                          </div>
                          <div className="text-[9px] text-center text-zinc-400 mt-0.5 truncate px-0.5">
                            {labels[index] || `#${index + 1}`}
                          </div>
                        </div>
                      ))}
                    </div>
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

                  {/* Download Button */}
                  {sessionName && (
                    <div className="pt-2 border-t border-zinc-700/50">
                      <button
                        onClick={handleDownload}
                        disabled={isDownloading}
                        className="w-full px-3 py-2 bg-zinc-800/50 hover:bg-zinc-700/50 text-zinc-300 hover:text-amber-400 transition-colors rounded text-xs font-medium flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed border border-zinc-700/50"
                        title="Download debug session"
                      >
                        <Download size={14} />
                        {isDownloading
                          ? "Downloading..."
                          : "Download Debug Session"}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Fullscreen Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-8"
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
