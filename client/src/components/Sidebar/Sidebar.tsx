interface SidebarProps {
  isOpen: boolean;
  uploadedImage: string | null;
  isMaskingMode: boolean;
  maskBrushSize: number;
  onImageUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveImage: () => void;
  onToggleMaskingMode: () => void;
  onClearMask: () => void;
  onMaskBrushSizeChange: (size: number) => void;
}

export default function Sidebar({
  isOpen,
  uploadedImage,
  isMaskingMode,
  maskBrushSize,
  onImageUpload,
  onRemoveImage,
  onToggleMaskingMode,
  onClearMask,
  onMaskBrushSizeChange
}: SidebarProps) {
  return (
    <div
      className={`bg-[var(--secondary-bg)] transition-all duration-300 flex-shrink-0 ${
        isOpen
          ? "lg:w-80 w-full h-64 lg:h-auto"
          : "w-0 lg:w-0 h-0 lg:h-auto overflow-hidden"
      } flex flex-col lg:flex-col overflow-hidden`}
    >
      {/* Customize Header */}
      {isOpen && (
        <div className="p-4 border-b border-[var(--primary-bg)]">
          <div className="flex items-center justify-between w-full">
            <span className="text-[var(--text-primary)] font-medium">
              Customize
            </span>
          </div>
        </div>
      )}

      {/* Customize Content */}
      {isOpen && (
        <div className="flex-1 p-4 space-y-4 lg:space-y-6 overflow-y-auto max-h-52 lg:max-h-none">
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
    </div>
  );
}
