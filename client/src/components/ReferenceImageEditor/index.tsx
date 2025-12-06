"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { X, Paintbrush, Wand2, Check, RotateCcw, Trash2, Plus } from "lucide-react";

interface ReferenceImageEditorProps {
  isOpen: boolean;
  imageData: string; // Base64 image data
  onClose: () => void;
  onSubmit: (processedImage: string, maskData?: string | null) => void; // Return processed image + mask for main canvas
}

export default function ReferenceImageEditor({
  isOpen,
  imageData,
  onClose,
  onSubmit,
}: ReferenceImageEditorProps) {
  // Canvas refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null); // Display canvas (shows combined mask)
  const pendingBrushCanvasRef = useRef<HTMLCanvasElement>(null); // Hidden canvas for new brush strokes
  const containerRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null); // Context for pending brush
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const confirmedMaskRef = useRef<ImageData | null>(null); // Stores confirmed SAM mask
  
  // State
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [loadedImage, setLoadedImage] = useState<HTMLImageElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(30); // UI value 0-100
  const [hasMask, setHasMask] = useState(false);
  const [hasPendingBrush, setHasPendingBrush] = useState(false); // Track if user drew new strokes
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [extractedObject, setExtractedObject] = useState<string | null>(null);
  const [mousePosition, setMousePosition] = useState<{ x: number; y: number } | null>(null);
  
  // Helper: get current combined mask (confirmed + pending) as binary PNG (data URL)
  const getCurrentMaskDataUrl = useCallback((): string | null => {
    if (!hasMask || !maskCanvasRef.current) return null;
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) return null;

    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = maskCanvas.width;
    tempCanvas.height = maskCanvas.height;
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return null;

    const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    // Convert red overlay to binary white mask
    for (let i = 0; i < maskImageData.data.length; i += 4) {
      if (maskImageData.data[i] > 100 && maskImageData.data[i + 3] > 50) {
        tempImageData.data[i] = 255;
        tempImageData.data[i + 1] = 255;
        tempImageData.data[i + 2] = 255;
        tempImageData.data[i + 3] = 255;
      } else {
        tempImageData.data[i] = 0;
        tempImageData.data[i + 1] = 0;
        tempImageData.data[i + 2] = 0;
        tempImageData.data[i + 3] = 255;
      }
    }
    tempCtx.putImageData(tempImageData, 0, 0);
    return tempCanvas.toDataURL("image/png");
  }, [hasMask]);

  // Calculate actual brush size based on image dimensions (like useMasking)
  const getActualBrushSize = useCallback(() => {
    if (!imageDimensions) return brushSize;
    const baseImageSize = Math.min(imageDimensions.width, imageDimensions.height);
    // Scale from 0.5% to 10% of base image size
    return (brushSize / 100) * (baseImageSize / 5);
  }, [brushSize, imageDimensions]);
  
  // Load image first (only dimensions)
  useEffect(() => {
    if (!isOpen || !imageData) return;
    
    setImageDimensions(null);
    setLoadedImage(null);
    
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });
      setLoadedImage(img);
      setHasMask(false);
      setHasPendingBrush(false);
      setExtractedObject(null);
      confirmedMaskRef.current = null;
    };
    img.onerror = () => {
      console.error("Failed to load image");
    };
    img.src = imageData;
  }, [isOpen, imageData]);
  
  // Draw image to canvas after canvas is mounted
  useEffect(() => {
    if (!loadedImage || !imageDimensions) return;
    
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    if (!canvas || !maskCanvas || !pendingCanvas) return;
    
    // Setup all canvas sizes
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;
    maskCanvas.width = imageDimensions.width;
    maskCanvas.height = imageDimensions.height;
    pendingCanvas.width = imageDimensions.width;
    pendingCanvas.height = imageDimensions.height;
    
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (ctx) {
      ctx.drawImage(loadedImage, 0, 0);
    }
    
    // Clear mask canvas
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (maskCtx) {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    }
    
    // Setup pending brush canvas context (this is where user draws)
    const pendingCtx = pendingCanvas.getContext("2d", { willReadFrequently: true });
    if (pendingCtx) {
      pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
      ctxRef.current = pendingCtx;
    }
    
    // Reset confirmed mask
    confirmedMaskRef.current = null;
  }, [loadedImage, imageDimensions]);
  
  // Setup brush properties when brush size changes
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !imageDimensions) return;
    
    const actualBrushSize = getActualBrushSize();
    ctx.lineWidth = actualBrushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
    ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
  }, [brushSize, imageDimensions, getActualBrushSize]);
  
  // Get canvas coordinates from mouse event
  const getCanvasCoordinates = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !imageDimensions) return null;
    
    const rect = canvas.getBoundingClientRect();
    
    // Handle both mouse and touch events
    let clientX: number, clientY: number;
    if ("touches" in e) {
      if (e.touches.length === 0) return null;
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    
    // Calculate relative position and scale to canvas internal coordinates
    const relativeX = (clientX - rect.left) / rect.width;
    const relativeY = (clientY - rect.top) / rect.height;
    
    return {
      x: relativeX * imageDimensions.width,
      y: relativeY * imageDimensions.height,
    };
  }, [imageDimensions]);
  
  // Helper: Update display canvas with combined mask (confirmed + pending)
  const updateDisplayCanvas = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    if (!maskCanvas || !pendingCanvas || !imageDimensions) return;
    
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!maskCtx) return;
    
    // Clear display canvas
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    // Draw confirmed mask first (if exists)
    if (confirmedMaskRef.current) {
      maskCtx.putImageData(confirmedMaskRef.current, 0, 0);
    }
    
    // Draw pending brush on top (with source-over to add)
    maskCtx.drawImage(pendingCanvas, 0, 0);
  }, [imageDimensions]);

  // Drawing handlers - similar to useMasking
  const startDrawing = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    const coords = getCanvasCoordinates(e);
    if (!coords) return;
    
    const ctx = ctxRef.current; // This is pending brush canvas context
    if (!ctx) return;
    
    setIsDrawing(true);
    lastPointRef.current = coords;
    
    // Setup brush properties
    const actualBrushSize = getActualBrushSize();
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;
    ctx.lineWidth = actualBrushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
    ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    
    // Draw initial dot
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, actualBrushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    
    // Start path for continuous stroke
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    
    setHasMask(true);
    setHasPendingBrush(true);
    
    // Update display
    updateDisplayCanvas();
  }, [getCanvasCoordinates, getActualBrushSize, updateDisplayCanvas]);
  
  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    e.preventDefault();
    
    const coords = getCanvasCoordinates(e);
    if (!coords) return;
    
    const ctx = ctxRef.current;
    if (!ctx) return;
    
    // Draw line from last point to current point
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    // Move to current point for next segment
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    
    lastPointRef.current = coords;
    
    // Update display
    updateDisplayCanvas();
  }, [isDrawing, getCanvasCoordinates, updateDisplayCanvas]);
  
  const stopDrawing = useCallback(() => {
    if (isDrawing) {
      setIsDrawing(false);
      lastPointRef.current = null;
    }
  }, [isDrawing]);
  
  // Handle mouse move for brush preview
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  }, []);
  
  // Clear all masks (confirmed + pending)
  const clearMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current;
    const pendingCanvas = pendingBrushCanvasRef.current;
    
    if (maskCanvas) {
      const maskCtx = maskCanvas.getContext("2d");
      if (maskCtx) {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      }
    }
    
    if (pendingCanvas) {
      const pendingCtx = pendingCanvas.getContext("2d");
      if (pendingCtx) {
        pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
      }
    }
    
    confirmedMaskRef.current = null;
    setHasMask(false);
    setHasPendingBrush(false);
    setExtractedObject(null);
  }, []);
  
  // FastSAM detect
  const detectWithSAM = useCallback(async () => {
    if (!imageDimensions) return;
    
    setIsLoading(true);
    setLoadingMessage(hasPendingBrush ? "Detecting object from brush..." : "Auto-detecting object...");
    
    try {
      // Use API Gateway instead of direct service call
      const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 
        process.env.NEXT_PUBLIC_API_URL ||
        "https://nxan2911--api-gateway.modal.run";
      
      // Get ONLY pending brush data as guidance (not the confirmed mask)
      // This allows detecting new objects without interference from previously detected ones
      let maskData: string | null = null;
      if (hasPendingBrush && pendingBrushCanvasRef.current) {
        // Convert pending brush to binary (white = brush area)
        const pendingCanvas = pendingBrushCanvasRef.current;
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = pendingCanvas.width;
        tempCanvas.height = pendingCanvas.height;
        const tempCtx = tempCanvas.getContext("2d");
        
        if (tempCtx) {
          tempCtx.fillStyle = "black";
          tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
          
          const pendingCtx = pendingCanvas.getContext("2d");
          if (pendingCtx) {
            const pendingImageData = pendingCtx.getImageData(0, 0, pendingCanvas.width, pendingCanvas.height);
            const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            
            for (let i = 0; i < pendingImageData.data.length; i += 4) {
              // If red channel has value (brush area), set to white
              if (pendingImageData.data[i] > 100 && pendingImageData.data[i + 3] > 50) {
                tempImageData.data[i] = 255;
                tempImageData.data[i + 1] = 255;
                tempImageData.data[i + 2] = 255;
                tempImageData.data[i + 3] = 255;
              }
            }
            
            tempCtx.putImageData(tempImageData, 0, 0);
          }
          
          maskData = tempCanvas.toDataURL("image/png").split(",")[1];
        }
      }
      
      // Call FastSAM API through API Gateway
      const response = await fetch(`${API_GATEWAY_URL}/api/smart-mask/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageData.startsWith("data:") ? imageData.split(",")[1] : imageData,
          mask: maskData, // Only pending brush as guidance
          auto_detect: !hasPendingBrush, // Auto detect if no brush strokes
        }),
      });
      
      if (!response.ok) {
        throw new Error("Failed to detect object");
      }
      
      const data = await response.json();
      
      if (data.mask) {
        // Apply detected mask and merge with existing confirmed mask
        const maskImg = new Image();
        maskImg.onload = () => {
          const maskCanvas = maskCanvasRef.current;
          const pendingCanvas = pendingBrushCanvasRef.current;
          if (!maskCanvas || !pendingCanvas) return;
          
          // Convert SAM result to red transparent
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = maskCanvas.width;
          tempCanvas.height = maskCanvas.height;
          const tempCtx = tempCanvas.getContext("2d");
          
          if (tempCtx) {
            tempCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
            const newMaskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            
            // Convert white areas to red transparent
            for (let i = 0; i < newMaskData.data.length; i += 4) {
              if (newMaskData.data[i] > 100) { // White area
                newMaskData.data[i] = 255;     // R
                newMaskData.data[i + 1] = 0;   // G
                newMaskData.data[i + 2] = 0;   // B
                newMaskData.data[i + 3] = 128; // A (semi-transparent)
              } else {
                newMaskData.data[i + 3] = 0;   // Transparent
              }
            }
            
            // Merge with existing confirmed mask
            if (confirmedMaskRef.current) {
              const existingData = confirmedMaskRef.current.data;
              for (let i = 0; i < newMaskData.data.length; i += 4) {
                // If existing mask has content, keep it (OR operation)
                if (existingData[i + 3] > 0) {
                  newMaskData.data[i] = 255;
                  newMaskData.data[i + 1] = 0;
                  newMaskData.data[i + 2] = 0;
                  newMaskData.data[i + 3] = 128;
                }
              }
            }
            
            // Save as new confirmed mask
            confirmedMaskRef.current = newMaskData;
            
            // Clear pending brush (it's now part of confirmed mask)
            const pendingCtx = pendingCanvas.getContext("2d");
            if (pendingCtx) {
              pendingCtx.clearRect(0, 0, pendingCanvas.width, pendingCanvas.height);
            }
            setHasPendingBrush(false);
            
            // Update display
            const maskCtx = maskCanvas.getContext("2d");
            if (maskCtx) {
              maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
              maskCtx.putImageData(newMaskData, 0, 0);
            }
          }
          setHasMask(true);
        };
        maskImg.src = `data:image/png;base64,${data.mask}`;
      }
    } catch (error) {
      console.error("FastSAM detection failed:", error);
      alert("Failed to detect object. Please try again.");
    } finally {
      setIsLoading(false);
      setLoadingMessage("");
    }
  }, [imageDimensions, hasPendingBrush, imageData]);
  
  // Extract object (remove background)
  const extractObject = useCallback(async () => {
    if (!hasMask || !imageDimensions) {
      // No mask, use original image
      onSubmit(imageData, null);
      onClose();
      return;
    }
    
    setIsLoading(true);
    setLoadingMessage("Extracting object...");
    
    try {
      // Use API Gateway instead of direct service call
      const API_GATEWAY_URL = process.env.NEXT_PUBLIC_API_GATEWAY_URL || 
        process.env.NEXT_PUBLIC_API_URL ||
        "https://nxan2911--api-gateway.modal.run";
      const currentMaskDataUrl = getCurrentMaskDataUrl();
      
      // Get mask data
      const maskCanvas = maskCanvasRef.current;
      if (!maskCanvas) {
        onSubmit(imageData, currentMaskDataUrl);
        onClose();
        return;
      }
      
      // Convert mask to binary
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = maskCanvas.width;
      tempCanvas.height = maskCanvas.height;
      const tempCtx = tempCanvas.getContext("2d");
      
      if (!tempCtx) {
        onSubmit(imageData);
        onClose();
        return;
      }
      
      tempCtx.fillStyle = "black";
      tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
      
      const maskCtx = maskCanvas.getContext("2d");
      if (maskCtx) {
        const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const tempImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        
        for (let i = 0; i < maskImageData.data.length; i += 4) {
          if (maskImageData.data[i] > 100 && maskImageData.data[i + 3] > 50) {
            tempImageData.data[i] = 255;
            tempImageData.data[i + 1] = 255;
            tempImageData.data[i + 2] = 255;
            tempImageData.data[i + 3] = 255;
          }
        }
        
        tempCtx.putImageData(tempImageData, 0, 0);
      }
      
      const maskData = tempCanvas.toDataURL("image/png").split(",")[1];
      
      // Call extract API through API Gateway
      const response = await fetch(`${API_GATEWAY_URL}/api/image-utils/extract-object`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageData.startsWith("data:") ? imageData.split(",")[1] : imageData,
          mask: maskData,
        }),
      });
      
      if (!response.ok) {
        throw new Error("Failed to extract object");
      }
      
      const data = await response.json();
      
      if (data.extracted_image) {
        const extractedDataUrl = `data:image/png;base64,${data.extracted_image}`;
        setExtractedObject(extractedDataUrl);
        onSubmit(extractedDataUrl, currentMaskDataUrl);
        onClose();
      } else {
        onSubmit(imageData, currentMaskDataUrl);
        onClose();
      }
    } catch (error) {
      console.error("Object extraction failed:", error);
      alert("Failed to extract object. Using original image.");
      onSubmit(imageData, getCurrentMaskDataUrl());
      onClose();
    } finally {
      setIsLoading(false);
      setLoadingMessage("");
    }
  }, [hasMask, imageDimensions, imageData, onSubmit, onClose, getCurrentMaskDataUrl]);
  
  // Handle submit
  const handleSubmit = useCallback(() => {
    if (!hasMask) {
      // No mask, use original image
      onSubmit(imageData, null);
      onClose();
    } else {
      // Has mask, extract object
      extractObject();
    }
  }, [hasMask, imageData, onSubmit, onClose, extractObject]);
  
  if (!isOpen) return null;
  
  const displayScale = imageDimensions 
    ? Math.min(600 / imageDimensions.width, 500 / imageDimensions.height, 1)
    : 1;
  
  return (
    <div className="fixed inset-0 z-[100] bg-black/80 flex items-center justify-center p-4">
      <div className="bg-zinc-900 rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-700">
          <h2 className="text-lg font-semibold text-white">Edit Reference Image</h2>
          <button
            onClick={onClose}
            className="p-1 text-zinc-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        
        {/* Toolbar */}
        <div className="flex items-center gap-3 px-4 py-2 border-b border-zinc-700 bg-zinc-800">
          {/* Brush size */}
          <div className="flex items-center gap-2">
            <Paintbrush size={16} className="text-zinc-400" />
            <input
              type="range"
              onMouseDown={(e) => e.stopPropagation()}
              onMouseMove={(e) => e.stopPropagation()}
              min="5"
              max="100"
              value={brushSize}
              onChange={(e) => setBrushSize(parseInt(e.target.value))}
              className="w-24 accent-amber-500"
            />
            <span className="text-xs text-zinc-400 w-8">{brushSize}px</span>
          </div>
          
          <div className="h-4 w-px bg-zinc-600" />
          
          {/* FastSAM detect */}
          <button
            onClick={detectWithSAM}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            {hasMask && !hasPendingBrush ? (
              <>
                <Plus size={14} />
                Add object
              </>
            ) : hasPendingBrush ? (
              <>
                <Wand2 size={14} />
                Detect from brush
              </>
            ) : (
              <>
                <Wand2 size={14} />
                Auto detect
              </>
            )}
          </button>
          
          {/* Clear mask */}
          <button
            onClick={clearMask}
            disabled={isLoading || !hasMask}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded-lg text-sm transition-colors disabled:opacity-50"
          >
            <Trash2 size={14} />
            Clear
          </button>
        </div>
        
        {/* Canvas area */}
        <div 
          ref={containerRef}
          className="flex-1 overflow-auto p-4 flex items-center justify-center bg-zinc-950 relative"
        >
          {isLoading && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
              <div className="text-center">
                <div className="animate-spin w-8 h-8 border-2 border-amber-500 border-t-transparent rounded-full mx-auto mb-2" />
                <p className="text-sm text-zinc-300">{loadingMessage}</p>
              </div>
            </div>
          )}
          
          {!imageDimensions && (
            <div className="text-zinc-500 text-sm">Loading image...</div>
          )}
          
          {/* Hidden canvas for pending brush strokes */}
          <canvas
            ref={pendingBrushCanvasRef}
            className="hidden"
          />
          
          {imageDimensions && (
            <div 
              className="relative border border-zinc-700 rounded-lg overflow-hidden"
              style={{
                width: Math.max(imageDimensions.width * displayScale, 200),
                height: Math.max(imageDimensions.height * displayScale, 200),
              }}
            >
              {/* Main image canvas */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 rounded-lg"
                style={{ 
                  width: "100%", 
                  height: "100%",
                  imageRendering: "auto" 
                }}
              />
              
              {/* Mask overlay canvas */}
              <canvas
                ref={maskCanvasRef}
                className="absolute inset-0 rounded-lg cursor-none"
                style={{ 
                  width: "100%", 
                  height: "100%",
                  imageRendering: "auto",
                  touchAction: "none",
                }}
                onMouseDown={startDrawing}
                onMouseMove={(e) => {
                  handleMouseMove(e);
                  draw(e);
                }}
                onMouseUp={stopDrawing}
                onMouseLeave={() => {
                  stopDrawing();
                  setMousePosition(null);
                }}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
              
              {/* Brush preview cursor */}
              {mousePosition && imageDimensions && (
                <div
                  className="pointer-events-none absolute"
                  style={{
                    left: mousePosition.x,
                    top: mousePosition.y,
                    transform: "translate(-50%, -50%)",
                  }}
                >
                  {/* Outer ring - black border */}
                  <div
                    className="absolute rounded-full border-2 border-black/70"
                    style={{
                      width: getActualBrushSize() * displayScale,
                      height: getActualBrushSize() * displayScale,
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                  {/* Inner ring - white */}
                  <div
                    className="absolute rounded-full border border-white/90"
                    style={{
                      width: getActualBrushSize() * displayScale - 2,
                      height: getActualBrushSize() * displayScale - 2,
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                  {/* Center dot */}
                  <div
                    className="absolute w-1 h-1 bg-white rounded-full border border-black/50"
                    style={{
                      transform: "translate(-50%, -50%)",
                      left: "50%",
                      top: "50%",
                    }}
                  />
                </div>
              )}
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-700 bg-zinc-800">
          <p className="text-xs text-zinc-400">
            {hasMask 
              ? "Object selected. Click Submit to extract." 
              : "Draw on image to select object, or click Auto detect. Submit without selection to use original image."}
          </p>
          
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              disabled={isLoading}
              className="px-4 py-2 text-zinc-300 hover:text-white transition-colors text-sm disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="flex items-center gap-1.5 px-4 py-2 bg-amber-500 hover:bg-amber-600 text-black font-medium rounded-lg text-sm transition-colors disabled:opacity-50"
            >
              <Check size={16} />
              Submit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

