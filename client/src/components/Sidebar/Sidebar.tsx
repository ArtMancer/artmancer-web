import React from 'react';

interface SidebarProps {
  isOpen: boolean;
  width?: number; // Optional width for resizable functionality
  uploadedImage: string | null;
  isMaskingMode: boolean;
  maskBrushSize: number;
  isResizing?: boolean; // For resize handle styling
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
  onResizeStart?: (e: React.MouseEvent) => void; // Resize handle handler
  onWidthChange?: (width: number) => void; // For keyboard resize
}

export default function Sidebar({
  isOpen,
  width = 320, // Default width
  uploadedImage,
  isMaskingMode,
  maskBrushSize,
  isResizing = false,
  onImageUpload,
  onRemoveImage,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange,
  onResizeStart,
  onWidthChange
}: SidebarProps) {
  return (
    <div
      className={`bg-[var(--secondary-bg)] flex-shrink-0 flex flex-col lg:flex-col overflow-hidden fixed right-0 z-30 ${
        isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
      style={{
        top: '0', // Start from the very top
        width: `${width}px`,
        height: '100vh', // Full viewport height
        paddingTop: '4rem', // Account for header height with padding
        transform: isOpen ? 'translateX(0)' : `translateX(${width}px)`,
        transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease',
        transformOrigin: 'right center',
        willChange: 'transform, opacity'
      }}
    >
      {/* Customize Header - no extra top padding */}
      {isOpen && (
        <div className="px-4 pb-2 border-b border-[var(--primary-bg)]">
          <div className="flex items-center justify-between w-full">
            <span className="text-[var(--text-primary)] font-medium">
              Customize
            </span>
          </div>
        </div>
      )}

      {/* Customize Content - minimal spacing */}
      {isOpen && (
        <div className="flex-1 px-4 pt-2 pb-4 space-y-4 lg:space-y-6 overflow-y-auto">
          {/* Image Upload Section */}
          <div>
            <h3 className="text-[var(--text-primary)] font-medium mb-2 lg:mb-3 text-sm lg:text-base">
              Image Upload
            </h3>
            <div className="space-y-3">
              <label
                htmlFor="image-upload-panel"
                className="w-full px-4 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white rounded-lg cursor-pointer transition-colors text-sm font-medium flex items-center justify-center gap-2"
              >
                📷 Choose Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={onImageUpload}
                className="hidden"
                id="image-upload-panel"
              />
              {uploadedImage && (
                <button
                  onClick={onRemoveImage}
                  className="w-full px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors text-sm font-medium"
                >
                  Remove Current Image
                </button>
              )}
            </div>
          </div>

          {/* Style Section */}
          <div>
            <h3 className="text-[var(--text-primary)] font-medium mb-2 lg:mb-3 text-sm lg:text-base">
              Style
            </h3>
            <div className="space-y-1 lg:space-y-2 text-sm lg:text-base">
              <label className="flex items-center">
                <input type="radio" name="style" className="mr-2" />
                <span className="text-[var(--text-secondary)]">
                  Realistic
                </span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="style"
                  className="mr-2"
                  defaultChecked
                />
                <span className="text-[var(--text-secondary)]">
                  Fantasy
                </span>
              </label>
              <label className="flex items-center">
                <input type="radio" name="style" className="mr-2" />
                <span className="text-[var(--text-secondary)]">
                  Abstract
                </span>
              </label>
            </div>
          </div>

          {/* Quality Section */}
          <div>
            <h3 className="text-[var(--text-primary)] font-medium mb-2 lg:mb-3 text-sm lg:text-base">
              Quality
            </h3>
            <select className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base">
              <option>Standard</option>
              <option>High</option>
              <option>Ultra</option>
            </select>
          </div>

          {/* Masking Tool Section */}
          <div>
            <h3 className="text-[var(--text-primary)] font-medium mb-2 lg:mb-3 text-sm lg:text-base">
              Masking Tool
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <button
                  onClick={onToggleMaskingMode}
                  disabled={!uploadedImage}
                  className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                    isMaskingMode
                      ? 'bg-[var(--highlight-accent)] text-white'
                      : 'bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white disabled:bg-gray-600 disabled:cursor-not-allowed'
                  }`}
                >
                  {isMaskingMode ? 'Exit Masking' : 'Start Masking'}
                </button>
                {isMaskingMode && (
                  <button
                    onClick={onClearMask}
                    className="px-3 py-2 bg-red-500 hover:bg-red-600 text-white rounded text-sm font-medium transition-colors"
                  >
                    Clear Mask
                  </button>
                )}
              </div>
              
              {isMaskingMode && (
                <div>
                  <label className="block text-[var(--text-secondary)] text-sm mb-2">
                    Brush Size: {maskBrushSize}px
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    value={maskBrushSize}
                    onChange={(e) => onMaskBrushSizeChange(parseInt(e.target.value))}
                    className="w-full accent-[var(--primary-accent)]"
                  />
                </div>
              )}
              
              <p className="text-xs text-[var(--text-secondary)]">
                {!uploadedImage 
                  ? "Upload an image first to use masking"
                  : isMaskingMode 
                    ? "Draw on the image to create mask areas. Red areas will be regenerated."
                    : "Enable masking mode to draw mask areas directly on the image."
                }
              </p>
            </div>
          </div>

          {/* Advanced Settings */}
          <div className="hidden lg:block">
            <h3 className="text-[var(--text-primary)] font-medium mb-3">
              Advanced
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-[var(--text-secondary)] text-sm mb-1">
                  Creativity:{" "}
                  <span className="text-[var(--primary-accent)]">70%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  defaultValue="70"
                  className="w-full accent-[var(--primary-accent)]"
                />
              </div>
              <div>
                <label className="block text-[var(--text-secondary)] text-sm mb-1">
                  Steps
                </label>
                <input
                  type="number"
                  min="10"
                  max="100"
                  defaultValue="50"
                  className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)]"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Resize Handle - Attached to sidebar for smooth movement */}
      {isOpen && onResizeStart && (
        <div
          className={`absolute top-0 left-0 w-3 h-full cursor-col-resize z-10 transition-all duration-150 ${
            isResizing 
              ? 'bg-[var(--primary-accent)] opacity-100' 
              : 'bg-transparent hover:bg-[var(--primary-accent)] hover:opacity-60'
          }`}
          style={{ 
            transform: 'translateX(-50%)', // Center on the left edge
            willChange: isResizing ? 'background-color' : 'auto'
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
            if (e.key === 'ArrowLeft' && onWidthChange) {
              e.preventDefault();
              onWidthChange(Math.max(280, width - 10));
              // Save to localStorage
              localStorage.setItem('sidebarWidth', String(Math.max(280, width - 10)));
            } else if (e.key === 'ArrowRight' && onWidthChange) {
              e.preventDefault();
              onWidthChange(Math.min(600, width + 10));
              // Save to localStorage
              localStorage.setItem('sidebarWidth', String(Math.min(600, width + 10)));
            }
          }}
        />
      )}
    </div>
  );
}
