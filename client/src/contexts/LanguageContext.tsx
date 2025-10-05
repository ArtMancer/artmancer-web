"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';

type Language = 'en' | 'vi';

interface Translations {
  [key: string]: string;
}

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: string) => string;
}

const translations: Record<Language, Translations> = {
  en: {
    // Header
    'header.placeholder': 'Describe the edits you want to make...',
    'header.edit': 'Edit!',
    'header.generating': 'Generating...',
    'header.settings': 'Settings',
    'header.profile': 'Profile',
    'header.toggleSidebar': 'Toggle Sidebar',
    
    // Settings
    'settings.title': 'Settings',
    'settings.theme': 'Theme',
    'settings.language': 'Language',
    'settings.light': 'Light',
    'settings.dark': 'Dark',
    'settings.english': 'English',
    'settings.vietnamese': 'Vietnamese',
    
    // Sidebar
    'sidebar.imageUpload': 'Image Upload',
    'sidebar.chooseImage': 'Choose Image',
    'sidebar.removeImage': 'Remove Current Image',
    'sidebar.aiTask': 'AI Task',
    'sidebar.whiteBalance': 'White Balancing',
    'sidebar.objectInsert': 'Object Insert',
    'sidebar.objectRemoval': 'Object Removal',
    'sidebar.referenceImage': 'Reference Image',
    'sidebar.chooseReference': 'Choose Reference Image',
    'sidebar.removeReference': 'Remove Reference Image',
    'sidebar.quality': 'Quality',
    'sidebar.standard': 'Standard',
    'sidebar.high': 'High',
    'sidebar.ultra': 'Ultra',
    'sidebar.markInsertArea': 'Mark Insert Area',
    'sidebar.markRemovalArea': 'Mark Removal Area',
    'sidebar.style': 'Style',
    'sidebar.realistic': 'Realistic',
    'sidebar.artistic': 'Artistic',
    'sidebar.cartoon': 'Cartoon',
    'sidebar.sketch': 'Sketch',
    'sidebar.masking': 'Masking',
    'sidebar.enableMasking': 'Start Masking',
    'sidebar.disableMasking': 'Exit Masking',
    'sidebar.clearMask': 'Clear Mask',
    'sidebar.brushSize': 'Brush Size',
    'sidebar.advanced': 'Advanced',
    'sidebar.creativity': 'Creativity',
    'sidebar.negativePrompt': 'Negative Prompt',
    'sidebar.negativePromptPlaceholder': 'What to avoid in the generation...',
    'sidebar.guidanceScale': 'Guidance Scale',
    'sidebar.moreCreative': 'More Creative',
    'sidebar.followPromptStrictly': 'Follow Prompt Strictly',
    'sidebar.imageSize': 'Image Size',
    'sidebar.width': 'Width',
    'sidebar.height': 'Height',
    'sidebar.inferenceSteps': 'Inference Steps',
    'sidebar.faster': 'Faster',
    'sidebar.higherQuality': 'Higher Quality',
    'sidebar.numImages': 'Number of Images',
    'sidebar.cfgScale': 'CFG Scale',
    'sidebar.cfgDescription': 'Enable classifier-free guidance (requires negative prompt)',
    'sidebar.whiteBalanceSettings': 'White Balance Settings',
    'sidebar.autoCorrectionStrength': 'Auto-correction strength',
    'sidebar.whiteBalanceDescription': 'AI will automatically adjust the white balance of your image for natural color correction.',
    'sidebar.editingComplete': 'Editing Complete!',
    'sidebar.editingCompleteDesc': 'Your image has been successfully processed. Use the comparison slider to see the changes, or download/upload a new image.',
    'sidebar.downloadImage': 'Download Image',
    'sidebar.returnToOriginal': 'Return to Original',
    'sidebar.newImage': 'New Image',
    'sidebar.uploadFirst': 'Upload an image first to use masking',
    'sidebar.maskingHelp': 'Draw on the image to create mask areas. Red areas will be regenerated.',
    'sidebar.enableMaskingHelp': 'Enable masking mode to draw mask areas directly on the image.',
    'sidebar.objectInsertion': 'object insertion',
    'sidebar.objectRemovalText': 'object removal',
    'sidebar.insertion': 'insertion',
    'sidebar.removal': 'removal',
    'sidebar.undo': 'Undo',
    'sidebar.redo': 'Redo',
    
    // New translation keys for sidebar sections
    'sidebar.maskingTool': 'Masking Tool',
    'sidebar.brush': 'Brush',
    'sidebar.eraser': 'Eraser',
    'sidebar.startMasking': 'Start Masking',
    'sidebar.exitMasking': 'Exit Masking',
    'sidebar.undoMaskStroke': 'Undo mask stroke',
    'sidebar.redoMaskStroke': 'Redo mask stroke',
    'sidebar.uploadFirstObjectInsert': 'Upload an image first to use object insertion',
    'sidebar.uploadFirstObjectRemove': 'Upload an image first to use object removal',
    'sidebar.drawForInsertion': 'Draw on the image to mark areas for object insertion. Red areas will be processed.',
    'sidebar.drawForRemoval': 'Draw on the image to mark areas for removal. Red areas will be processed.',
    'sidebar.enableMaskingInsertion': 'Enable masking mode to mark insertion areas directly on the image.',
    'sidebar.enableMaskingRemoval': 'Enable masking mode to mark removal areas directly on the image.',
    'sidebar.aiAutoAdjust': 'AI will automatically adjust the white balance of your image for natural color correction.',
    'sidebar.numberOfImages': 'Number of Images',
    'sidebar.oneImage': '1 Image',
    'sidebar.twoImages': '2 Images',
    'sidebar.threeImages': '3 Images',
    'sidebar.fourImages': '4 Images',
    'sidebar.cfgGuidanceDescription': 'Enable classifier-free guidance (requires negative prompt)',
    
    'sidebar.1Image': '1 Image',
    'sidebar.2Images': '2 Images',
    'sidebar.3Images': '3 Images',
    'sidebar.4Images': '4 Images',
    
    // Canvas
    'canvas.clickToChange': 'Click to change image',
    'canvas.uploadImage': 'Click to upload an image',
    'canvas.dragDrop': 'or drag and drop here',
    'canvas.removeImage': 'Remove image',
    
    // Status Bar
    'status.dimensions': 'Dimensions',
    'status.scale': 'Scale',
    'status.zoom': 'Zoom',
    'status.masking': 'Masking',
    'status.enabled': 'Enabled',
    'status.disabled': 'Disabled',
    
    // Toolbox
    'toolbox.comparison': 'Comparison',
    'toolbox.download': 'Download',
    'toolbox.undo': 'Undo',
    'toolbox.redo': 'Redo',
    'toolbox.zoomIn': 'Zoom In',
    'toolbox.zoomOut': 'Zoom Out',
    'toolbox.resetZoom': 'Reset Zoom',
    'toolbox.help': 'Help',
  },
  vi: {
    // Header
    'header.placeholder': 'Mô tả những chỉnh sửa bạn muốn thực hiện...',
    'header.edit': 'Chỉnh sửa!',
    'header.generating': 'Đang tạo...',
    'header.settings': 'Cài đặt',
    'header.profile': 'Hồ sơ',
    'header.toggleSidebar': 'Bật/Tắt thanh bên',
    
    // Settings
    'settings.title': 'Cài đặt',
    'settings.theme': 'Chủ đề',
    'settings.language': 'Ngôn ngữ',
    'settings.light': 'Sáng',
    'settings.dark': 'Tối',
    'settings.english': 'Tiếng Anh',
    'settings.vietnamese': 'Tiếng Việt',
    
    // Sidebar
    'sidebar.imageUpload': 'Tải ảnh lên',
    'sidebar.chooseImage': 'Chọn ảnh',
    'sidebar.removeImage': 'Xóa ảnh hiện tại',
    'sidebar.aiTask': 'Nhiệm vụ AI',
    'sidebar.whiteBalance': 'Cân bằng trắng',
    'sidebar.objectInsert': 'Chèn đối tượng',
    'sidebar.objectRemoval': 'Xóa đối tượng',
    'sidebar.referenceImage': 'Ảnh tham khảo',
    'sidebar.chooseReference': 'Chọn ảnh tham khảo',
    'sidebar.removeReference': 'Xóa ảnh tham khảo',
    'sidebar.quality': 'Chất lượng',
    'sidebar.standard': 'Tiêu chuẩn',
    'sidebar.high': 'Cao',
    'sidebar.ultra': 'Siêu cao',
    'sidebar.markInsertArea': 'Đánh dấu vùng chèn',
    'sidebar.markRemovalArea': 'Đánh dấu vùng xóa',
    'sidebar.style': 'Phong cách',
    'sidebar.realistic': 'Thực tế',
    'sidebar.artistic': 'Nghệ thuật',
    'sidebar.cartoon': 'Hoạt hình',
    'sidebar.sketch': 'Phác thảo',
    'sidebar.masking': 'Tạo mặt nạ',
    'sidebar.enableMasking': 'Bắt đầu tạo mặt nạ',
    'sidebar.disableMasking': 'Thoát khỏi tạo mặt nạ',
    'sidebar.clearMask': 'Xóa mặt nạ',
    'sidebar.brushSize': 'Kích thước cọ',
    'sidebar.advanced': 'Nâng cao',
    'sidebar.creativity': 'Sáng tạo',
    'sidebar.negativePrompt': 'Lời nhắc tiêu cực',
    'sidebar.negativePromptPlaceholder': 'Những gì cần tránh trong quá trình tạo...',
    'sidebar.guidanceScale': 'Thang điểm hướng dẫn',
    'sidebar.moreCreative': 'Sáng tạo hơn',
    'sidebar.followPromptStrictly': 'Tuân thủ lời nhắc nghiêm ngặt',
    'sidebar.imageSize': 'Kích thước ảnh',
    'sidebar.width': 'Chiều rộng',
    'sidebar.height': 'Chiều cao',
    'sidebar.inferenceSteps': 'Số bước suy luận',
    'sidebar.faster': 'Nhanh hơn',
    'sidebar.higherQuality': 'Chất lượng cao hơn',
    'sidebar.numImages': 'Số lượng ảnh',
    'sidebar.cfgScale': 'Thang điểm CFG',
    'sidebar.cfgDescription': 'Bật hướng dẫn không phân loại (cần lời nhắc tiêu cực)',
    'sidebar.whiteBalanceSettings': 'Cài đặt cân bằng trắng',
    'sidebar.autoCorrectionStrength': 'Cường độ tự động hiệu chỉnh',
    'sidebar.whiteBalanceDescription': 'AI sẽ tự động điều chỉnh cân bằng trắng của ảnh để hiệu chỉnh màu sắc tự nhiên.',
    'sidebar.editingComplete': 'Hoàn tất chỉnh sửa!',
    'sidebar.editingCompleteDesc': 'Ảnh của bạn đã được xử lý thành công. Sử dụng thanh trượt so sánh để xem các thay đổi, hoặc tải xuống/tải lên ảnh mới.',
    'sidebar.downloadImage': 'Tải xuống ảnh',
    'sidebar.returnToOriginal': 'Quay về ảnh gốc',
    'sidebar.newImage': 'Ảnh mới',
    'sidebar.uploadFirst': 'Tải ảnh lên trước để sử dụng tính năng tạo mặt nạ',
    'sidebar.maskingHelp': 'Vẽ lên ảnh để tạo vùng mặt nạ. Vùng màu đỏ sẽ được tạo lại.',
    'sidebar.enableMaskingHelp': 'Bật chế độ tạo mặt nạ để vẽ vùng mặt nạ trực tiếp lên ảnh.',
    'sidebar.objectInsertion': 'chèn đối tượng',
    'sidebar.objectRemovalText': 'xóa đối tượng',
    'sidebar.insertion': 'chèn',
    'sidebar.removal': 'xóa',
    'sidebar.undo': 'Hoàn tác',
    'sidebar.redo': 'Làm lại',
    
    // New translation keys for sidebar sections
    'sidebar.maskingTool': 'Công cụ tạo mặt nạ',
    'sidebar.brush': 'Cọ',
    'sidebar.eraser': 'Tẩy',
    'sidebar.startMasking': 'Bắt đầu tạo mặt nạ',
    'sidebar.exitMasking': 'Thoát khỏi tạo mặt nạ',
    'sidebar.undoMaskStroke': 'Hoàn tác nét vẽ mặt nạ',
    'sidebar.redoMaskStroke': 'Làm lại nét vẽ mặt nạ',
    'sidebar.uploadFirstObjectInsert': 'Tải ảnh lên trước để sử dụng chèn đối tượng',
    'sidebar.uploadFirstObjectRemove': 'Tải ảnh lên trước để sử dụng xóa đối tượng',
    'sidebar.drawForInsertion': 'Vẽ lên ảnh để đánh dấu vùng chèn đối tượng. Vùng màu đỏ sẽ được xử lý.',
    'sidebar.drawForRemoval': 'Vẽ lên ảnh để đánh dấu vùng xóa. Vùng màu đỏ sẽ được xử lý.',
    'sidebar.enableMaskingInsertion': 'Bật chế độ tạo mặt nạ để đánh dấu vùng chèn trực tiếp lên ảnh.',
    'sidebar.enableMaskingRemoval': 'Bật chế độ tạo mặt nạ để đánh dấu vùng xóa trực tiếp lên ảnh.',
    'sidebar.aiAutoAdjust': 'AI sẽ tự động điều chỉnh cân bằng trắng của ảnh để hiệu chỉnh màu sắc tự nhiên.',
    'sidebar.numberOfImages': 'Số lượng ảnh',
    'sidebar.oneImage': '1 Ảnh',
    'sidebar.twoImages': '2 Ảnh',
    'sidebar.threeImages': '3 Ảnh',
    'sidebar.fourImages': '4 Ảnh',
    'sidebar.cfgGuidanceDescription': 'Bật hướng dẫn không phân loại (cần lời nhắc tiêu cực)',
    
    'sidebar.1Image': '1 Ảnh',
    'sidebar.2Images': '2 Ảnh',
    'sidebar.3Images': '3 Ảnh',
    'sidebar.4Images': '4 Ảnh',
    
    // Canvas
    'canvas.clickToChange': 'Nhấp để thay đổi ảnh',
    'canvas.uploadImage': 'Nhấp để tải ảnh lên',
    'canvas.dragDrop': 'hoặc kéo thả vào đây',
    'canvas.removeImage': 'Xóa ảnh',
    
    // Status Bar
    'status.dimensions': 'Kích thước',
    'status.scale': 'Tỉ lệ',
    'status.zoom': 'Thu phóng',
    'status.masking': 'Tạo mặt nạ',
    'status.enabled': 'Đã bật',
    'status.disabled': 'Đã tắt',
    
    // Toolbox
    'toolbox.comparison': 'So sánh',
    'toolbox.download': 'Tải xuống',
    'toolbox.undo': 'Hoàn tác',
    'toolbox.redo': 'Làm lại',
    'toolbox.zoomIn': 'Phóng to',
    'toolbox.zoomOut': 'Thu nhỏ',
    'toolbox.resetZoom': 'Đặt lại thu phóng',
    'toolbox.help': 'Trợ giúp',
  }
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>('en');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('artmancer-language') as Language;
    if (saved && (saved === 'en' || saved === 'vi')) {
      setLanguageState(saved);
    }
  }, []);

  useEffect(() => {
    if (mounted) {
      localStorage.setItem('artmancer-language', language);
    }
  }, [language, mounted]);

  const setLanguage = (newLanguage: Language) => {
    setLanguageState(newLanguage);
  };

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  if (!mounted) {
    return null;
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}