interface HelpBoxProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function HelpBox({ isOpen, onClose }: HelpBoxProps) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed bottom-20 left-4 bg-[var(--secondary-bg)] border border-[var(--primary-accent)] rounded-lg p-4 shadow-xl z-30 max-w-xs">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[var(--text-primary)] font-medium text-sm">
          Controls & Tips
        </h3>
        <button
          onClick={onClose}
          className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          ×
        </button>
      </div>
      <div className="space-y-2 text-xs text-[var(--text-secondary)]">
        <div>
          <span className="text-[var(--primary-accent)] font-medium">Viewport Control:</span>
          <p>• Mouse wheel to zoom in/out</p>
          <p>• Zoom buttons for precise control</p>
        </div>
        <div>
          <span className="text-[var(--primary-accent)] font-medium">Viewport Zoom:</span>
          <p>• Use +/− buttons in top-right corner</p>
          <p>• Mouse wheel to zoom viewport in/out</p>
          <p>• Zooms the entire image display area</p>
          <p>• 1:1 button resets viewport zoom</p>
        </div>
        <div>
          <span className="text-[var(--primary-accent)] font-medium">Image Upload:</span>
          <p>• Click on upload area to select image</p>
          <p>• Drag & drop supported</p>
        </div>
        <div>
          <span className="text-[var(--primary-accent)] font-medium">Image Controls:</span>
          <p>• Left click + drag to pan image</p>
          <p>• Remove (×) to delete image</p>
        </div>
        <div>
          <span className="text-[var(--primary-accent)] font-medium">Masking Tool:</span>
          <p>• Enable masking mode in sidebar</p>
          <p>• Draw directly on image to mask</p>
          <p>• Adjust brush size as needed</p>
          <p>• Red areas = regenerate content</p>
          <p>• Clear mask to start over</p>
        </div>
      </div>
    </div>
  );
}
