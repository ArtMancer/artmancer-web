"use client";

import { useState } from "react";
import Image from "next/image";

export default function Home() {
  const [isCustomizeOpen, setIsCustomizeOpen] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [selectedSize, setSelectedSize] = useState("768x768");

  const sizeOptions = {
    "512x512": {
      display: "w-64 h-64 lg:w-72 lg:h-72",
      label: "Standard (512x512)",
    },
    "768x768": {
      display: "w-72 h-72 lg:w-96 lg:h-96",
      label: "High (768x768)",
    },
    "1024x1024": {
      display: "w-80 h-80 lg:w-[28rem] lg:h-[28rem]",
      label: "Ultra (1024x1024)",
    },
  };

  const handleSummon = () => {
    // TODO: Implement art generation logic
    console.log("Summoning art...", { size: selectedSize });
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setUploadedImage(null);
  };

  return (
    <div className="min-h-screen max-h-screen bg-[var(--primary-bg)] text-[var(--text-primary)] flex flex-col dots-pattern-small overflow-hidden">
      {/* Header */}
      <header className="p-4 border-b border-[var(--secondary-bg)] flex-shrink-0">
        <div className="w-full flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Image
              src="/logo.svg"
              alt="Artmancer"
              width={180}
              height={48}
              className="h-12 w-auto"
              priority
            />
          </div>

          {/* Input Field and Summon Button */}
          <div className="flex-1 max-w-2xl mx-4 flex gap-3 items-center">
            <input
              type="text"
              placeholder="Describe the art you want to summon..."
              className="flex-1 px-4 py-3 bg-transparent border-2 border-[var(--primary-accent)] rounded-lg text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--highlight-accent)] focus:outline-none transition-colors text-sm"
            />
            <button
              onClick={handleSummon}
              className="px-6 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white font-semibold rounded-lg transition-colors text-sm flex-shrink-0"
            >
              Summon!
            </button>
          </div>

          {/* Header Actions */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <button
              className="p-2 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--highlight-accent)] transition-colors"
              aria-label="Settings"
            >
              <Image
                src="/settings.svg"
                alt="Settings"
                width={24}
                height={24}
                className="opacity-80 hover:opacity-100 transition-opacity"
              />
            </button>
            <button
              className="p-2 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--highlight-accent)] transition-colors"
              aria-label="Profile"
            >
              <Image
                src="/profile.svg"
                alt="Profile"
                width={24}
                height={24}
                className="opacity-80 hover:opacity-100 transition-opacity"
              />
            </button>
            <button
              onClick={() => setIsCustomizeOpen(!isCustomizeOpen)}
              className="p-2 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--highlight-accent)] transition-colors"
              aria-label="Toggle Sidebar"
            >
              <Image
                src="/sidebar.svg"
                alt="Toggle Sidebar"
                width={24}
                height={24}
                className="opacity-80 hover:opacity-100 transition-opacity"
              />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left Side - Art Generation */}
        <div className="flex-1 flex flex-col items-center justify-center p-4 lg:p-8 min-w-0">
          {/* Art Display Area with Image Input */}
          <label
            className={`${
              sizeOptions[selectedSize as keyof typeof sizeOptions].display
            } bg-[var(--primary-accent)] rounded-lg flex items-center justify-center shadow-lg relative overflow-hidden cursor-pointer hover:bg-[var(--highlight-accent)] transition-colors`}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            {uploadedImage ? (
              <div className="relative w-full h-full">
                <img
                  src={uploadedImage}
                  alt="Uploaded reference"
                  className="w-full h-full object-cover rounded-lg"
                />
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    removeImage();
                  }}
                  className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs transition-colors z-10"
                >
                  ×
                </button>
                <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  Reference Image
                </div>
                <div className="absolute inset-0 bg-black/20 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="text-white text-sm font-medium">
                    Click to change image
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center text-white/80">
                <div className="text-3xl lg:text-4xl mb-2">📷</div>
                <p className="text-xs lg:text-sm px-4 mb-2">
                  Click to upload an image
                </p>
                <p className="text-xs px-4 text-white/60">
                  or your summoned art will appear here
                </p>
                <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
                  {selectedSize}
                </div>
              </div>
            )}
          </label>
        </div>

        {/* Right Side - Customize Panel */}
        <div
          className={`bg-[var(--secondary-bg)] transition-all duration-300 flex-shrink-0 ${
            isCustomizeOpen
              ? "lg:w-80 w-full h-64 lg:h-auto"
              : "w-0 lg:w-0 h-0 lg:h-auto overflow-hidden"
          } flex flex-col lg:flex-col overflow-hidden`}
        >
          {/* Customize Header */}
          {isCustomizeOpen && (
            <div className="p-4 border-b border-[var(--primary-bg)]">
              <div className="flex items-center justify-between w-full">
                <span className="text-[var(--text-primary)] font-medium">
                  Customize
                </span>
              </div>
            </div>
          )}

          {/* Customize Content */}
          {isCustomizeOpen && (
            <div className="flex-1 p-4 space-y-4 lg:space-y-6 overflow-y-auto max-h-52 lg:max-h-none">
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

              {/* Size Section */}
              <div>
                <h3 className="text-[var(--text-primary)] font-medium mb-2 lg:mb-3 text-sm lg:text-base">
                  Size
                </h3>
                <select
                  value={selectedSize}
                  onChange={(e) => setSelectedSize(e.target.value)}
                  className="w-full px-3 py-2 bg-[var(--primary-bg)] border border-[var(--primary-accent)] rounded text-[var(--text-primary)] text-sm lg:text-base"
                >
                  {Object.entries(sizeOptions).map(([size, config]) => (
                    <option key={size} value={size}>
                      {config.label}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-[var(--text-secondary)] mt-1">
                  Changes both preview and generation size
                </p>
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
      </main>
    </div>
  );
}
