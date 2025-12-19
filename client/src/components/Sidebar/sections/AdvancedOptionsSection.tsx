"use client";

import React from "react";
import { Slider } from "radix-ui";
import {
  CircleSlash,
  SlidersHorizontal,
  ListOrdered,
  Dice5,
  Sparkles,
} from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";
import { useAdvancedOptions } from "@/hooks/Sidebar/useAdvancedOptions";

interface AdvancedOptionsSectionProps {
  /** Current negative prompt value */
  negativePrompt: string;
  /** Current AI task (UI level) */
  taskType: "white-balance" | "object-insert" | "object-removal";
  /** Current guidance scale (1-10) - used for both guidance_scale and true_cfg_scale */
  guidanceScale: number;
  /** Current inference steps (1-100) */
  inferenceSteps: number;
  /** Current seed value */
  seed?: number;
  /** Whether editing is done (comparison mode) */
  isEditingDone: boolean;
  /** Enable MAE refinement (for removal task only) */
  enableMaeRefinement?: boolean;

  // Callbacks from parent
  onNegativePromptChange?: (value: string) => void;
  onGuidanceScaleChange?: (value: number) => void;
  onInferenceStepsChange?: (value: number) => void;
  onSeedChange?: (value: number) => void;
  onEnableMaeRefinementChange?: (value: boolean) => void;
}

/**
 * Advanced Options Section Component
 * 
 * Provides advanced generation controls:
 * - Negative Prompt (with toggle)
 * - Guidance Scale (1-10) - used for both guidance_scale (SD) and true_cfg_scale (Qwen)
 * - Inference Steps (1-100)
 * - Seed (for reproducibility)
 * 
 * User Interaction Flow:
 * 1. User toggles negative prompt → textarea appears/disappears
 * 2. User adjusts guidance scale/steps/seed → values update in parent
 * 3. User disables negative prompt → prompt cleared
 * 
 * State Changes:
 * - guidanceScale/inferenceSteps/seed change → triggers generation parameter updates
 * Note: guidanceScale is used for both guidance_scale (SD) and true_cfg_scale (Qwen)
 */
export default function AdvancedOptionsSection({
  negativePrompt,
  taskType,
  guidanceScale,
  inferenceSteps,
  seed,
  isEditingDone,
  enableMaeRefinement = true,
  onNegativePromptChange,
  onGuidanceScaleChange,
  onInferenceStepsChange,
  onSeedChange,
  onEnableMaeRefinementChange,
}: AdvancedOptionsSectionProps) {
  const { t } = useLanguage();

  // Use hook for negative prompt toggle logic
  const { isNegativePromptEnabled, handleToggleNegativePrompt } =
    useAdvancedOptions({
      negativePrompt,
      onNegativePromptChange,
      taskType,
      onGuidanceScaleChange,
    });

  // Early return if section should not be visible
  if (isEditingDone) {
    return null;
  }

  return (
    <div className="pt-2">
      <h3 className="text-text-primary font-medium mb-4">
        {t("sidebar.advanced")}
      </h3>
      <div className="space-y-6">
        {/* Negative Prompt with Toggle */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-text-secondary text-sm">
              <span className="inline-flex items-center gap-2">
                <CircleSlash size={16} />
                {t("sidebar.negativePrompt")}
              </span>
            </label>
            <button
              onClick={handleToggleNegativePrompt}
              className={`relative w-10 h-5 rounded-full transition-colors ${
                isNegativePromptEnabled
                  ? "bg-primary-accent"
                  : "bg-border-color"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                  isNegativePromptEnabled ? "translate-x-5" : "translate-x-0"
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
              className="w-full px-3 py-2 bg-primary-bg border border-primary-accent rounded text-text-primary text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary-accent focus:border-transparent"
            />
          )}
        </div>

        {/* Guidance Scale */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-text-secondary text-sm">
              <span className="inline-flex items-center gap-2">
                <SlidersHorizontal size={16} />
                {t("sidebar.guidanceScale")}
              </span>
            </label>
            <input
              type="number"
              min={1}
              max={10}
              step={0.5}
              value={guidanceScale}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                if (!Number.isNaN(value) && value >= 1 && value <= 10) {
                  onGuidanceScaleChange?.(value);
                }
              }}
              onBlur={(e) => {
                const value = parseFloat(e.target.value);
                if (Number.isNaN(value) || value < 1) {
                  onGuidanceScaleChange?.(1);
                } else if (value > 10) {
                  onGuidanceScaleChange?.(10);
                }
              }}
              className="w-16 px-2 py-1 text-sm bg-primary-bg border border-border-color rounded text-text-primary text-right focus:outline-none focus:border-primary-accent"
            />
          </div>
          <div className="px-2">
            <Slider.Root
              min={1}
              max={10}
              step={0.5}
              value={[guidanceScale]}
              onValueChange={([value]: number[]) => onGuidanceScaleChange?.(value)}
              className="relative flex h-5 w-full touch-none select-none items-center"
            >
              <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
              </Slider.Track>
              <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
            </Slider.Root>
          </div>
        </div>

        {/* Steps (Updated from existing) */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-text-secondary text-sm">
              <span className="inline-flex items-center gap-2">
                <ListOrdered size={16} />
                {t("sidebar.inferenceSteps")}
              </span>
            </label>
            <input
              type="number"
              min={1}
              max={100}
              value={inferenceSteps}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10);
                if (!Number.isNaN(value)) {
                  onInferenceStepsChange?.(value);
                }
              }}
              onBlur={(e) => {
                const value = parseInt(e.target.value, 10);
                if (Number.isNaN(value) || value < 1) {
                  onInferenceStepsChange?.(1);
                } else if (value > 100) {
                  onInferenceStepsChange?.(100);
                }
              }}
              className="w-16 px-2 py-1 text-sm bg-primary-bg border border-border-color rounded text-text-primary text-right focus:outline-none focus:border-primary-accent"
            />
          </div>
          <div className="px-2">
            <Slider.Root
              min={1}
              max={100}
              step={1}
              value={[inferenceSteps]}
              onValueChange={([value]: number[]) => onInferenceStepsChange?.(value)}
              className="relative flex h-5 w-full touch-none select-none items-center"
            >
              <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
              </Slider.Track>
              <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform focus:outline-none focus:ring-2 focus:ring-primary-accent" />
            </Slider.Root>
          </div>
        </div>

        {/* MAE Refinement Toggle - Only for removal task */}
        {taskType === "object-removal" && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-text-secondary text-sm">
                <span className="inline-flex items-center gap-2">
                  <Sparkles size={16} />
                  {t("sidebar.maeRefinement") || "MAE Refinement"}
                </span>
              </label>
              <button
                onClick={() => onEnableMaeRefinementChange?.(!enableMaeRefinement)}
                className={`relative w-10 h-5 rounded-full transition-colors ${
                  enableMaeRefinement
                    ? "bg-primary-accent"
                    : "bg-border-color"
                }`}
              >
                <span
                  className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                    enableMaeRefinement ? "translate-x-5" : "translate-x-0"
                  }`}
                />
              </button>
            </div>
            <p className="text-xs text-text-secondary mt-1">
              {t("sidebar.maeRefinementDescription") || "Improve texture quality using Stable Diffusion Inpainting refinement"}
            </p>
          </div>
        )}

        {/* Seed */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-text-secondary text-sm">
              <span className="inline-flex items-center gap-2">
                <Dice5 size={16} />
                {t("sidebar.seed")}
              </span>
            </label>
            <input
              type="number"
              min={0}
              max={2147483647}
              step={1}
              value={seed ?? 42}
              onChange={(e) => {
                const value = parseInt(e.target.value, 10);
                if (!Number.isNaN(value) && value >= 0) {
                  onSeedChange?.(value);
                }
              }}
              onBlur={(e) => {
                const value = parseInt(e.target.value, 10);
                if (Number.isNaN(value) || value < 0) {
                  onSeedChange?.(42);
                }
              }}
              className="w-24 px-2 py-1 text-sm bg-primary-bg border border-border-color rounded text-text-primary text-right focus:outline-none focus:border-primary-accent"
            />
          </div>
          <p className="text-xs text-text-secondary mt-1">
            {t("sidebar.seedDescription")}
          </p>
        </div>
      </div>
    </div>
  );
}

