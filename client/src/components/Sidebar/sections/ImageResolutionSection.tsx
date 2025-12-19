"use client";

import React, { useMemo } from "react";
import { HelpCircle, Square, Maximize2, Ruler } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import type { InputQualityPreset } from "@/services/api";

interface ImageResolutionSectionProps {
  /** Current input quality preset */
  inputQuality: InputQualityPreset;
  /** Custom square size for 1:1 ratio mode */
  customSquareSize: number;
  /** Whether quality is currently being applied */
  isApplyingQuality: boolean;
  /** Image dimensions for warning calculation */
  imageDimensions: { width: number; height: number } | null;
  /** Callback when quality preset changes */
  onInputQualityChange: (value: InputQualityPreset) => void;
  /** Callback when custom square size changes */
  onCustomSquareSizeChange?: (size: number) => void;
}

/**
 * Image Resolution Section Component
 * 
 * Allows users to select input quality preset (resized 1:1 or original) and
 * configure custom square size for resized mode.
 * 
 * User Interaction Flow:
 * 1. User selects quality preset (resized/original) → onInputQualityChange triggers
 * 2. If resized mode → user can set custom square size (256-2048px) → onCustomSquareSizeChange
 * 3. If original mode and image >= 2048px → warning message appears
 * 4. When quality is applying → buttons disabled, loading message shown
 * 
 * State Changes:
 * - inputQuality change → triggers image resize in page.tsx (applyInputQualityPreset)
 * - customSquareSize change → triggers re-apply if in resized mode
 */
export default function ImageResolutionSection({
  inputQuality,
  customSquareSize,
  isApplyingQuality,
  imageDimensions,
  onInputQualityChange,
  onCustomSquareSizeChange,
}: ImageResolutionSectionProps) {
  const { t } = useLanguage();
  
  // Local computed values (moved from Sidebar.tsx)
  // These are computed in component to avoid prop drilling
  const inputQualityOptions: {
    value: InputQualityPreset;
    shortLabel: string;
    description: string;
    icon: "square" | "original";
  }[] = useMemo(
    () => [
      {
        value: "resized",
        shortLabel: "1:1",
        description: t("imageResolution.resize11"),
        icon: "square",
      },
      {
        value: "original",
        shortLabel: t("imageResolution.original"),
        description: t("imageResolution.originalImage"),
        icon: "original",
      },
    ],
    [t]
  );

  const selectedQuality = useMemo(
    () => inputQualityOptions.find((option) => option.value === inputQuality),
    [inputQualityOptions, inputQuality]
  );

  const showOriginalWarning = useMemo(
    () =>
      inputQuality === "original" &&
      imageDimensions &&
      Math.max(imageDimensions.width, imageDimensions.height) >= 2048,
    [inputQuality, imageDimensions]
  );

  // Early return if section should not be visible
  // This matches original condition but we'll handle it in parent
  // Actually, looking at original code, this section is always shown when uploadedImage exists
  // So we don't need early return here, parent handles visibility

  return (
    <>
      <div className="my-2 border-t border-border-color" />
      <div className="pb-2">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-text-primary font-semibold text-sm sm:text-base">
            {t("imageResolution.title")}
          </h4>
          <button
            title={t("imageResolution.helpTooltip")}
            className="p-1 text-text-secondary hover:text-primary-accent"
          >
            <HelpCircle size={18} />
          </button>
        </div>
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {inputQualityOptions.map((option) => {
              const active = inputQuality === option.value;
              return (
                <button
                  key={option.value}
                  onClick={() => onInputQualityChange(option.value)}
                  disabled={isApplyingQuality}
                  className={`w-[calc(50%-4px)] min-w-0 flex items-center justify-center rounded border px-3 py-3 text-sm ${
                    active
                      ? "bg-primary-accent text-white border-primary-accent"
                      : "bg-transparent text-text-primary border-border-color hover:bg-(--hover-bg)"
                  } ${isApplyingQuality ? "opacity-60 cursor-not-allowed" : ""}`}
                >
                  <span className="flex flex-col items-center justify-center gap-1">
                    <span className="text-[11px] font-medium">
                      {option.shortLabel}
                    </span>
                    <span className="flex items-center gap-1 text-[10px] text-text-secondary">
                      {option.icon === "square" ? (
                        <Square size={14} />
                      ) : (
                        <Maximize2 size={14} />
                      )}
                      <span>{option.description}</span>
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          {inputQuality === "resized" && (
            <div className="mt-4 space-y-3">
              {/* Label with icon - centered */}
              <div className="flex items-center justify-center gap-2">
                <Ruler size={14} className="text-text-secondary" />
                <label className="text-text-secondary text-xs font-medium">
                  {t("imageResolution.squareSize")}
                </label>
              </div>
              
              {/* Input and display - centered */}
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={256}
                    max={2048}
                    step={64}
                    value={customSquareSize}
                    onChange={(e) => {
                      const value = parseInt(e.target.value, 10) || 1024;
                      const clamped = Math.max(256, Math.min(2048, value));
                      onCustomSquareSizeChange?.(clamped);
                    }}
                    disabled={isApplyingQuality}
                    className="w-20 px-2 py-1.5 text-sm bg-secondary-bg border border-border-color rounded text-text-primary text-center focus:outline-none focus:border-primary-accent focus:ring-1 focus:ring-primary-accent"
                  />
                  <span className="text-text-secondary text-xs font-mono">×</span>
                  <div className="w-20 px-2 py-1.5 text-sm bg-secondary-bg border border-border-color rounded text-text-primary text-center font-mono">
                    {customSquareSize}
                  </div>
                </div>
                
                {/* Preset buttons - centered */}
                <div className="flex flex-wrap items-center justify-center gap-1.5">
                  {[512, 768, 1024, 1536, 2048].map((size) => {
                    const active = customSquareSize === size;
                    return (
                      <button
                        key={size}
                        onClick={() => onCustomSquareSizeChange?.(size)}
                        disabled={isApplyingQuality}
                        className={`btn-interactive px-3 py-1.5 rounded border text-xs font-medium transition-all duration-150 ${
                          active
                            ? "bg-primary-accent text-white border-primary-accent shadow-sm"
                            : "bg-secondary-bg text-text-secondary border-border-color hover:bg-primary-accent/10 hover:border-primary-accent/50"
                        } ${isApplyingQuality ? "opacity-60 cursor-not-allowed" : ""}`}
                      >
                        {size}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {selectedQuality && (
            <p className="text-text-secondary text-xs">
              {inputQuality === "resized"
                ? t("imageResolution.willResizeTo").replace("{size}", customSquareSize.toString())
                : t("imageResolution.currentlySelected")
                    .replace("{label}", selectedQuality.description)
                    .replace("{ratio}", selectedQuality.shortLabel)}
            </p>
          )}
          {isApplyingQuality && (
            <p className="text-text-secondary text-xs">
              {t("imageResolution.applyingResolution")}
            </p>
          )}
          {showOriginalWarning && (
            <p className="text-xs text-amber-300">
              {t("imageResolution.largeImageWarning")
                .replace("{width}", imageDimensions?.width?.toString() ?? "")
                .replace("{height}", imageDimensions?.height?.toString() ?? "")}
            </p>
          )}
        </div>
      </div>
    </>
  );
}

