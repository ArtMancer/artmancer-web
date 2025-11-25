import Image from "next/image";
import {
  MdSettings,
  MdPerson,
  MdMenu,
  MdLanguage,
  MdLightMode,
  MdDarkMode,
  MdCheck,
} from "react-icons/md";
import { useState, useRef, useEffect, useCallback } from "react";
import { useTheme, useLanguage } from "@/contexts";

interface HeaderProps {
  onSummon: (prompt: string) => void;
  onEvaluate?: (prompt: string) => void;
  isCustomizeOpen: boolean;
  onToggleCustomize: () => void;
  isGenerating?: boolean;
  isEvaluating?: boolean;
  appMode?: "inference" | "benchmark";
  onAppModeChange?: (mode: "inference" | "benchmark") => void;
  aiTask?: "white-balance" | "object-insert" | "object-removal" | "evaluation";
  onCancel?: () => void;
}

export default function Header({
  onSummon,
  onEvaluate,
  isCustomizeOpen,
  onToggleCustomize,
  isGenerating = false,
  isEvaluating = false,
  appMode = "inference",
  onAppModeChange,
  aiTask,
  onCancel,
}: HeaderProps) {
  const [prompt, setPrompt] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [isOpening, setIsOpening] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();

  // Function to handle modal opening with animation
  const handleOpenModal = useCallback(() => {
    setIsSettingsOpen(true);
    setIsOpening(true);
    // Remove opening state after animation completes
    setTimeout(() => {
      setIsOpening(false);
    }, 50); // Small delay to ensure the initial state is rendered
  }, []);

  // Function to handle modal closing with animation
  const handleCloseModal = useCallback(() => {
    setIsClosing(true);
    setTimeout(() => {
      setIsSettingsOpen(false);
      setIsClosing(false);
    }, 300); // Match animation duration
  }, []);

  // Close settings dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        settingsRef.current &&
        !settingsRef.current.contains(event.target as Node)
      ) {
        handleCloseModal();
      }
    }

    if (isSettingsOpen && !isClosing) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [isSettingsOpen, isClosing, handleCloseModal]);

  const handleSubmit = () => {
    const promptValue = prompt.trim();
    if (appMode === "benchmark" && onEvaluate) {
      // For benchmark mode, prompt is required
      if (!promptValue) {
        // Show error or disable button - validation handled by button disabled state
        return;
      }
      onEvaluate(promptValue);
    } else {
      // For white-balance task, prompt is optional (can be empty)
      // For white-balance, allow empty prompt
      if (aiTask === "white-balance" || promptValue) {
        onSummon(promptValue);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  return (
    <header className="p-4 border-b border-[var(--secondary-bg)] flex-shrink-0 bg-[var(--primary-bg)]">
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

        {/* Input Field and Edit Button */}
        <div className="flex-1 max-w-2xl mx-4 flex gap-3 items-center">
          <input
            type="text"
            placeholder={
              appMode === "benchmark" 
                ? "Enter prompt for ALL benchmark images..." 
                : t("header.placeholder")
            }
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isGenerating || isEvaluating}
            className="flex-1 px-4 py-3 bg-transparent border-2 border-[var(--primary-accent)] rounded-lg text-[var(--text-primary)] placeholder:text-[var(--text-secondary)] focus:border-[var(--highlight-accent)] focus:outline-none transition-colors text-sm h-12 disabled:opacity-50"
          />
          {isGenerating && onCancel ? (
            <button
              onClick={onCancel}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors text-sm flex-shrink-0 h-12"
            >
              {t("header.cancel")}
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={
                (isGenerating || isEvaluating) || 
                (appMode === "benchmark" ? !prompt.trim() : (!prompt.trim() && aiTask !== "white-balance"))
              }
              className="px-6 py-3 bg-[var(--primary-accent)] hover:bg-[var(--highlight-accent)] text-white font-semibold rounded-lg transition-colors text-sm flex-shrink-0 h-12 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {appMode === "benchmark" 
                ? (isEvaluating ? "Benchmarking..." : "Benchmark!")
                : (isGenerating ? t("header.generating") : t("header.edit"))}
            </button>
          )}
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-4 flex-shrink-0">
          {/* Settings Button */}
          <button
            onClick={() => {
              if (isSettingsOpen) {
                handleCloseModal();
              } else {
                handleOpenModal();
              }
            }}
            className="p-3 rounded-lg bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white transition-all duration-200 h-12 w-12 flex items-center justify-center"
            aria-label={t("header.settings")}
          >
            <MdSettings size={20} />
          </button>
          <button
            onClick={onToggleCustomize}
            className={`p-3 rounded-lg transition-all duration-200 h-12 w-12 flex items-center justify-center ${
              isCustomizeOpen
                ? "bg-[var(--primary-accent)] text-white"
                : "bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-secondary)] hover:text-white"
            }`}
            aria-label={t("header.toggleSidebar")}
          >
            <MdMenu size={20} />
          </button>
        </div>
      </div>

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop with blur */}
          <div
            className={`absolute inset-0 bg-black/50 backdrop-blur-md transition-opacity duration-300 ${
              isClosing ? "opacity-0" : isOpening ? "opacity-0" : "opacity-100"
            }`}
            onClick={handleCloseModal}
          />

          {/* Modal Content */}
          <div
            ref={settingsRef}
            className={`relative bg-[var(--primary-bg)] border border-[var(--border-color)] rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden transition-all duration-300 ease-out ${
              isClosing
                ? "opacity-0 scale-95 translate-y-2"
                : isOpening
                ? "opacity-0 scale-95 translate-y-2"
                : "opacity-100 scale-100 translate-y-0"
            }`}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-[var(--border-color)]">
              <h3 className="text-[var(--text-primary)] font-semibold text-lg">
                {t("settings.title")}
              </h3>
              <button
                onClick={handleCloseModal}
                className="p-2 hover:bg-[var(--secondary-bg)] rounded-lg transition-colors"
              >
                <svg
                  className="w-5 h-5 text-[var(--text-secondary)]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-6">
              {/* Theme Setting */}
              <div>
                <h4 className="text-[var(--text-primary)] font-medium mb-4 flex items-center gap-2">
                  <MdLightMode
                    className="text-[var(--primary-accent)]"
                    size={18}
                  />
                  {t("settings.theme")}
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => theme !== "light" && toggleTheme()}
                    className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                      theme === "light"
                        ? "bg-[var(--primary-accent)] text-white shadow-lg"
                        : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                    }`}
                  >
                    <MdLightMode size={18} />
                    <span>{t("settings.light")}</span>
                    {theme === "light" && <MdCheck size={16} />}
                  </button>
                  <button
                    onClick={() => theme !== "dark" && toggleTheme()}
                    className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                      theme === "dark"
                        ? "bg-[var(--primary-accent)] text-white shadow-lg"
                        : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                    }`}
                  >
                    <MdDarkMode size={18} />
                    <span>{t("settings.dark")}</span>
                    {theme === "dark" && <MdCheck size={16} />}
                  </button>
                </div>
              </div>

              {/* Language Setting */}
              <div>
                <h4 className="text-[var(--text-primary)] font-medium mb-4 flex items-center gap-2">
                  <MdLanguage
                    className="text-[var(--primary-accent)]"
                    size={18}
                  />
                  {t("settings.language")}
                </h4>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setLanguage("en")}
                    className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                      language === "en"
                        ? "bg-[var(--primary-accent)] text-white shadow-lg"
                        : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                    }`}
                  >
                    <span className="text-lg">🇺🇸</span>
                    <span>{t("settings.english")}</span>
                    {language === "en" && <MdCheck size={16} />}
                  </button>
                  <button
                    onClick={() => setLanguage("vi")}
                    className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                      language === "vi"
                        ? "bg-[var(--primary-accent)] text-white shadow-lg"
                        : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                    }`}
                  >
                    <span className="text-lg">🇻🇳</span>
                    <span>{t("settings.vietnamese")}</span>
                    {language === "vi" && <MdCheck size={16} />}
                  </button>
                </div>
              </div>

              {/* App Mode Setting */}
              {onAppModeChange && (
                <div>
                  <h4 className="text-[var(--text-primary)] font-medium mb-4 flex items-center gap-2">
                    <MdSettings
                      className="text-[var(--primary-accent)]"
                      size={18}
                    />
                    {t("settings.mode")}
                  </h4>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => onAppModeChange("inference")}
                      className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                        appMode === "inference"
                          ? "bg-[var(--primary-accent)] text-white shadow-lg"
                          : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                      }`}
                    >
                      <span>{t("settings.inference")}</span>
                      {appMode === "inference" && <MdCheck size={16} />}
                    </button>
                    <button
                      onClick={() => onAppModeChange("benchmark")}
                      className={`flex items-center justify-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                        appMode === "benchmark"
                          ? "bg-[var(--primary-accent)] text-white shadow-lg"
                          : "bg-[var(--secondary-bg)] text-[var(--text-secondary)] hover:bg-[var(--primary-accent)] hover:text-white"
                      }`}
                    >
                      <span>Benchmark</span>
                      {appMode === "benchmark" && <MdCheck size={16} />}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-6 pt-0">
              <button
                onClick={handleCloseModal}
                className="w-full px-4 py-2 bg-[var(--secondary-bg)] hover:bg-[var(--primary-accent)] text-[var(--text-primary)] hover:text-white rounded-lg transition-all duration-200 text-sm font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
