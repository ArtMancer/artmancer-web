import { Slider } from "radix-ui";
import { CircleHelp } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

interface ToolboxProps {
  // Visibility
  uploadedImage: string | null;
  
  // Comparison slider
  originalImage: string | null;
  modifiedImage: string | null;
  comparisonSlider: number;
  onComparisonSliderChange: (value: number) => void;
  
  // Viewport controls
  viewportZoom: number;
  onZoomViewportIn: () => void;
  onZoomViewportOut: () => void;
  onResetViewportZoom: () => void;
  
  // Help
  isHelpOpen: boolean;
  onToggleHelp: () => void;
}

export default function Toolbox({
  uploadedImage,
  originalImage,
  modifiedImage,
  comparisonSlider,
  onComparisonSliderChange,
  viewportZoom,
  onZoomViewportIn,
  onZoomViewportOut,
  onResetViewportZoom,
  isHelpOpen,
  onToggleHelp
}: ToolboxProps) {
  const { t } = useLanguage();

  return (
    <>
      {/* Main Toolbox - Bottom Center - Only show when image has been generated/modified */}
      {uploadedImage && modifiedImage && originalImage !== modifiedImage && (
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 bg-secondary-bg border border-primary-accent rounded-lg px-6 py-3 z-50 shadow-xl min-w-fit">
          <div className="flex items-center justify-center space-x-6">
            {/* Comparison Slider */}
            <div className="flex items-center space-x-3">
              <span className="text-xs text-text-secondary whitespace-nowrap">
                {t("toolbox.original")}
              </span>
              <div className="w-24">
                <Slider.Root
                  min={0}
                  max={100}
                  step={1}
                  value={[comparisonSlider]}
                  onValueChange={([value]: [number]) =>
                    onComparisonSliderChange(value)
                  }
                  className="relative flex h-5 w-full touch-none select-none items-center"
                >
                  <Slider.Track className="relative h-1 w-full rounded-full bg-border-color">
                  <Slider.Range className="absolute h-1 rounded-full bg-primary-accent" />
                  </Slider.Track>
                  <Slider.Thumb className="block h-4 w-4 rounded-full bg-primary-accent shadow transition-transform duration-100 ease-out focus:outline-none focus:ring-2 focus:ring-primary-accent" />
                </Slider.Root>
              </div>
              <span className="text-xs font-mono text-text-secondary w-10 text-center">
                {Math.round(comparisonSlider * 10) / 10}%
              </span>
              <span className="text-xs text-text-secondary whitespace-nowrap">
                {t("toolbox.modified")}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Viewport Zoom Controls - Top Right Corner */}
      <div className="absolute top-8 right-8 z-30">
        <div className="flex flex-col gap-1 bg-secondary-bg border border-primary-accent rounded-xl p-1 shadow-lg backdrop-blur-sm">
          <button
            onClick={onZoomViewportIn}
            className="btn-interactive rounded-lg w-10 h-10 flex items-center justify-center text-text-primary hover:bg-primary-accent hover:text-white transition-colors duration-150"
            title={t("toolbox.zoomIn")}
          >
            <span className="text-lg font-bold">+</span>
          </button>
          <button
            onClick={onZoomViewportOut}
            className="btn-interactive rounded-lg w-10 h-10 flex items-center justify-center text-text-primary hover:bg-primary-accent hover:text-white transition-colors duration-150"
            title={t("toolbox.zoomOut")}
          >
            <span className="text-lg font-bold">−</span>
          </button>
          {viewportZoom !== 1 && (
            <>
              <div className="px-2 py-1 text-xs text-text-secondary text-center border-t border-border-color mt-1 pt-1">
                {Math.round(viewportZoom * 1000) / 10}%
              </div>
              <button
                onClick={onResetViewportZoom}
                className="btn-interactive rounded-lg w-10 h-10 flex items-center justify-center text-text-primary hover:bg-primary-accent hover:text-white transition-colors duration-150 text-xs"
                title={t("toolbox.resetZoom")}
              >
                1:1
              </button>
            </>
          )}
        </div>
      </div>

      {/* Help Button - Fixed position in lower left */}
      <button
        onClick={onToggleHelp}
        className={`btn-interactive btn-primary-hover fixed bottom-4 left-4 rounded-full w-12 h-12 flex items-center justify-center shadow-lg transition-all duration-200 z-30 ${
          isHelpOpen
            ? "bg-highlight-accent text-white"
            : "bg-primary-accent hover:bg-highlight-accent text-white"
        }`}
        title={t("toolbox.help")}
      >
        <CircleHelp size={18} />
      </button>
    </>
  );
}
