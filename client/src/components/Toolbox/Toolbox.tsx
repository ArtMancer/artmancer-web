import { MdUndo, MdRedo, MdDownload, MdRefresh } from "react-icons/md";

interface ToolboxProps {
  // Visibility
  uploadedImage: string | null;
  
  // Comparison slider
  originalImage: string | null;
  modifiedImage: string | null;
  comparisonSlider: number;
  onComparisonSliderChange: (value: number) => void;
  
  // History controls
  historyIndex: number;
  historyStackLength: number;
  onUndo: () => void;
  onRedo: () => void;
  onDownload: () => void;
  
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
  historyIndex,
  historyStackLength,
  onUndo,
  onRedo,
  onDownload,
  viewportZoom,
  onZoomViewportIn,
  onZoomViewportOut,
  onResetViewportZoom,
  isHelpOpen,
  onToggleHelp
}: ToolboxProps) {
  return (
    <>
      {/* Main Toolbox - Bottom Center */}
      {uploadedImage && (
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg px-6 py-3 z-50 shadow-xl min-w-fit">
          <div className="flex items-center justify-center space-x-6">
            {/* Comparison Slider */}
            {originalImage && modifiedImage && (
              <div className="flex items-center space-x-3">
                <span className="text-xs text-[var(--text-secondary)] whitespace-nowrap">Original</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={comparisonSlider}
                  onChange={(e) => onComparisonSliderChange(Number(e.target.value))}
                  className="w-24 h-2 bg-[var(--primary-bg)] rounded-lg appearance-none cursor-pointer accent-[var(--primary-accent)]"
                />
                <span className="text-xs text-[var(--text-secondary)] whitespace-nowrap">Modified</span>
              </div>
            )}

            {/* Undo */}
            <button
              onClick={onUndo}
              disabled={historyIndex <= 0}
              className="p-3 bg-[var(--primary-accent)] text-white rounded-lg hover:bg-[var(--highlight-accent)] transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed min-w-[40px] h-[40px] flex items-center justify-center"
              title="Undo"
            >
              <MdUndo size={18} />
            </button>

            {/* Redo */}
            <button
              onClick={onRedo}
              disabled={historyIndex >= historyStackLength - 1}
              className="p-3 bg-[var(--primary-accent)] text-white rounded-lg hover:bg-[var(--highlight-accent)] transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed min-w-[40px] h-[40px] flex items-center justify-center"
              title="Redo"
            >
              <MdRedo size={18} />
            </button>

            {/* Download */}
            <button
              onClick={onDownload}
              className="p-3 bg-[var(--primary-accent)] text-white rounded-lg hover:bg-[var(--highlight-accent)] transition-colors text-sm min-w-[40px] h-[40px] flex items-center justify-center"
              title="Download Image"
            >
              <MdDownload size={18} />
            </button>
          </div>
        </div>
      )}
      
      {/* Viewport Zoom Controls - Top Right Corner */}
      <div className="absolute top-8 right-8 flex flex-col gap-2 z-30">
        <button
          onClick={onZoomViewportIn}
          className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded-lg w-10 h-10 flex items-center justify-center transition-all duration-200 shadow-lg"
          title="Zoom In Viewport"
        >
          <span className="text-lg font-bold">+</span>
        </button>
        <button
          onClick={onZoomViewportOut}
          className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded-lg w-10 h-10 flex items-center justify-center transition-all duration-200 shadow-lg"
          title="Zoom Out Viewport"
        >
          <span className="text-lg font-bold">−</span>
        </button>
        {viewportZoom !== 1 && (
          <button
            onClick={onResetViewportZoom}
            className="bg-[var(--secondary-bg)] border border-[var(--primary-accent)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded-lg w-10 h-10 flex items-center justify-center transition-all duration-200 shadow-lg text-xs"
            title="Reset Viewport Zoom"
          >
            1:1
          </button>
        )}
      </div>
      
      {/* Help Button - Fixed position in lower left */}
      <button
        onClick={onToggleHelp}
        className={`fixed bottom-4 left-4 rounded-full w-12 h-12 flex items-center justify-center shadow-lg transition-all duration-200 z-30 ${
          isHelpOpen 
            ? 'bg-[var(--highlight-accent)] text-white' 
            : 'bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white'
        }`}
        title="Help & Controls"
      >
        <span className="text-lg font-bold">?</span>
      </button>
    </>
  );
}
