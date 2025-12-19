"use client";

import type { ReactNode } from "react";

type Language = "en";

const translations = {
  en: {
    // Header
    "header.placeholder": "Describe the edits you want to make...",
    "header.edit": "Edit!",
    "header.generating": "Generating...",
    "header.evaluate": "Evaluate!",
    "header.evaluating": "Evaluating...",
    "header.settings": "Settings",
    "header.profile": "Profile",
    "header.toggleSidebar": "Toggle Sidebar",

    // Settings
    "settings.title": "Settings",
    "settings.theme": "Theme",
    "settings.language": "Language",
    "settings.mode": "App Mode",
    "settings.inference": "Inference",
    "settings.evaluation": "Evaluation",
    "settings.light": "Light",
    "settings.dark": "Dark",
    "settings.english": "English",
    "settings.vietnamese": "Vietnamese",
    "settings.debug": "Debug mode",
    "settings.debugEnabled": "Debug mode is enabled",
    "settings.debugDisabled": "Debug mode is disabled",
    "settings.close": "Close",

    // Sidebar
    "sidebar.imageUpload": "Image Upload",
    "sidebar.chooseImage": "Choose Image",
    "sidebar.removeImage": "Remove Current Image",
    "sidebar.aiTask": "AI Task",
    "sidebar.whiteBalance": "White Balancing",
    "sidebar.objectInsert": "Object Insert",
    "sidebar.objectRemoval": "Object Removal",
    "sidebar.evaluation": "Evaluation",
    "sidebar.referenceImage": "Reference Image",
    "sidebar.editReference": "Edit",
    "sidebar.chooseReference": "Choose Reference Image",
    "sidebar.removeReference": "Remove",
    "sidebar.evaluationMode": "Evaluation Mode",
    "sidebar.uploadOriginal": "Upload Original Image",
    "sidebar.uploadTarget": "Upload Target Image",
    "sidebar.uploadSinglePair": "Upload Single Image Pair",
    "sidebar.uploadMultiplePairs": "Upload Multiple Image Pairs",
    "sidebar.uploadConditionalImages": "Upload Conditional Images",
    "sidebar.uploadInputImage": "Upload Input Image",
    "sidebar.evaluateImages": "Evaluate Images",
    "sidebar.evaluationResults": "Evaluation Results",
    "sidebar.fileNameMismatch": "File names must match for pairs",
    "sidebar.selectMatchingFiles": "Select files with matching names",
    "sidebar.selectOriginalFolder": "Select Original Images Folder",
    "sidebar.selectTargetFolder": "Select Target Images Folder",
    "sidebar.originalImagesFolder": "Original Images Folder",
    "sidebar.targetImagesFolder": "Target Images Folder",
    "sidebar.singleImage": "Single",
    "sidebar.multipleImages": "Multiple",
    "sidebar.optional": "Optional",
    "sidebar.required": "Required",
    "sidebar.pairsLoaded": "pair(s) loaded",
    "sidebar.imagesLoaded": "image(s) loaded",
    "sidebar.loaded": "Loaded",
    "sidebar.allowMultipleFolders": "Allow multiple folder uploads",
    "sidebar.quality": "Quality",
    "sidebar.standard": "Standard",
    "sidebar.high": "High",
    "sidebar.ultra": "Ultra",
    "sidebar.markInsertArea": "Mark Insert Area",
    "sidebar.markRemovalArea": "Mark Removal Area",
    "sidebar.style": "Style",
    "sidebar.realistic": "Realistic",
    "sidebar.artistic": "Artistic",
    "sidebar.cartoon": "Cartoon",
    "sidebar.sketch": "Sketch",
    "sidebar.masking": "Masking",
    "sidebar.enableMasking": "Start Masking",
    "sidebar.disableMasking": "Exit Masking",
    "sidebar.clearMask": "Clear Mask",
    "sidebar.brushSize": "Brush Size",
    "sidebar.maskSettings": "Mask Settings",
    "sidebar.autoDetectEdges": "Auto-detect edges",
    "sidebar.autoFillToEdges": "Auto-fill to edges",
    "sidebar.smartMasking": "Smart Masking",
    "sidebar.smartMaskModel": "Model",
    "sidebar.birefnetBrushDisabled": "BiRefNet does not support brush tool",
    "sidebar.borderAdjustment": "Border Adjustment",
    "sidebar.shrink": "Shrink",
    "sidebar.grow": "Grow",
    "sidebar.noAdjustment": "No adjustment",
    "sidebar.advanced": "Advanced",
    "sidebar.creativity": "Creativity",
    "sidebar.negativePrompt": "Negative Prompt",
    "sidebar.negativePromptPlaceholder": "What to avoid in the generation...",
    "sidebar.guidanceScale": "Guidance Scale",
    "sidebar.moreCreative": "More Creative",
    "sidebar.followPromptStrictly": "Follow Prompt Strictly",
    "sidebar.imageSize": "Image Size",
    "sidebar.width": "Width",
    "sidebar.height": "Height",
    "sidebar.inferenceSteps": "Inference Steps",
    "sidebar.faster": "Faster",
    "sidebar.higherQuality": "Higher Quality",
    "sidebar.numImages": "Number of Images",
    "sidebar.cfgScale": "CFG Scale",
    "sidebar.cfgDescription":
      "Enable classifier-free guidance (requires negative prompt)",
    "sidebar.seed": "Seed",
    "sidebar.seedDescription": "Random seed for reproducible results (default: 42)",
    "sidebar.whiteBalanceSettings": "White Balance Settings",
    "sidebar.autoCorrectionStrength": "Auto-correction strength",
    "sidebar.whiteBalanceDescription":
      "AI will automatically adjust the white balance of your image for natural color correction.",
    "sidebar.method": "Method",
    "sidebar.autoWhiteBalance": "Auto White Balance",
    "sidebar.manualWhiteBalance": "Manual White Balance",
    "sidebar.aiWhiteBalance": "AI White Balance",
    "sidebar.temperature": "Temperature",
    "sidebar.tint": "Tint",
    "sidebar.applyWhiteBalance": "Apply White Balance",
    "sidebar.editingComplete": "Editing Complete!",
    "sidebar.editingCompleteDesc":
      "Your image has been successfully processed. Use the comparison slider to see the changes, or download/upload a new image.",
    "sidebar.downloadImage": "Download Image",
    "sidebar.downloadVisualization": "Download Visualization",
    "sidebar.returnToOriginal": "Return to Original",
    "sidebar.newImage": "New Image",
    "sidebar.uploadFirst": "Upload an image first to use masking",
    "sidebar.maskingHelp":
      "Draw on the image to create mask areas. Red areas will be regenerated.",
    "sidebar.enableMaskingHelp":
      "Enable masking mode to draw mask areas directly on the image.",
    "sidebar.objectInsertion": "object insertion",
    "sidebar.objectRemovalText": "object removal",
    "sidebar.insertion": "insertion",
    "sidebar.removal": "removal",
    "sidebar.undo": "Undo",
    "sidebar.redo": "Redo",

    // New translation keys for sidebar sections
    "sidebar.maskingTool": "Masking Tool",
    "sidebar.maskTool": "Mask Tool",
    "sidebar.brush": "Brush",
    "sidebar.box": "Box",
    "sidebar.eraser": "Eraser",
    "sidebar.startMasking": "Start Masking",
    "sidebar.exitMasking": "Exit Masking",
    "sidebar.undoMaskStroke": "Undo mask stroke",
    "sidebar.redoMaskStroke": "Redo mask stroke",
    "sidebar.uploadFirstObjectInsert":
      "Upload an image first to use object insertion",
    "sidebar.uploadFirstObjectRemove":
      "Upload an image first to use object removal",
    "sidebar.drawForInsertion":
      "Draw on the image to mark areas for object insertion. Red areas will be processed.",
    "sidebar.drawForRemoval":
      "Draw on the image to mark areas for removal. Red areas will be processed.",
    "sidebar.enableMaskingInsertion":
      "Enable masking mode to mark insertion areas directly on the image.",
    "sidebar.enableMaskingRemoval":
      "Enable masking mode to mark removal areas directly on the image.",
    "sidebar.aiAutoAdjust":
      "AI will automatically adjust the white balance of your image for natural color correction.",
    "sidebar.numberOfImages": "Number of Images",
    "sidebar.oneImage": "1 Image",
    "sidebar.twoImages": "2 Images",
    "sidebar.threeImages": "3 Images",
    "sidebar.fourImages": "4 Images",
    "sidebar.cfgGuidanceDescription":
      "Enable classifier-free guidance (requires negative prompt)",

    "sidebar.1Image": "1 Image",
    "sidebar.2Images": "2 Images",
    "sidebar.3Images": "3 Images",
    "sidebar.4Images": "4 Images",
    "sidebar.maeRefinement": "MAE Refinement",
    "sidebar.maeRefinementDescription": "Refine image using MAE (Mean Absolute Error) method",

    // Image Resolution Section
    "imageResolution.title": "Image Resolution",
    "imageResolution.helpTooltip": "Reduce input image size to save VRAM. Lower levels run faster but with less detail.",
    "imageResolution.resize11": "Resize 1:1",
    "imageResolution.originalImage": "Original Image",
    "imageResolution.original": "Original",
    "imageResolution.squareSize": "Square size (px):",
    "imageResolution.willResizeTo": "Image will be resized to {size}×{size}",
    "imageResolution.currentlySelected": "Currently selected: {label} ({ratio} original size)",
    "imageResolution.applyingResolution": "Applying new resolution, please wait...",
    "imageResolution.largeImageWarning": "Large image ({width}×{height}). May take a lot of time and VRAM.",

    // Canvas
    "canvas.clickToChange": "Click to change image",
    "canvas.uploadImage": "Click to upload an image",
    "canvas.dragDrop": "or drag and drop here",
    "canvas.removeImage": "Remove image",

    // Status Bar
    "status.dimensions": "Dimensions",
    "status.scale": "Scale",
    "status.zoom": "Zoom",
    "status.masking": "Masking",
    "status.enabled": "Enabled",
    "status.disabled": "Disabled",

    // Toolbox
    "toolbox.comparison": "Comparison",
    "toolbox.download": "Download",
    "toolbox.undo": "Undo",
    "toolbox.redo": "Redo",
    "toolbox.zoomIn": "Zoom In",
    "toolbox.zoomOut": "Zoom Out",
    "toolbox.resetZoom": "Reset Zoom",
    "toolbox.help": "Help",
    "toolbox.original": "Original",
    "toolbox.modified": "Modified",
  },
};

const tFn = (key: string): string => {
  const dict = translations.en as Record<string, string>;
  return dict[key] || key;
};

export function LanguageProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

export function useLanguage() {
  return {
    language: "en" as Language,
    setLanguage: () => undefined,
    t: tFn,
  };
}
