import Image from "next/image";
import {
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  startTransition,
} from "react";
import { Dialog, ToggleGroup, Switch as RadixSwitch } from "radix-ui";
import { useTheme, useLanguage } from "@/contexts";
import { Settings, Menu, X, Sun, Moon, Bug, Check, Wand2 } from "lucide-react";

interface HeaderProps {
  onSummon: (prompt: string) => void;
  isCustomizeOpen: boolean;
  onToggleCustomize: () => void;
  isGenerating?: boolean;
  aiTask?: "white-balance" | "object-insert" | "object-removal";
  onCancel?: () => void;
  // Debug panel props
  isDebugPanelVisible?: boolean;
  onDebugPanelVisibilityChange?: (visible: boolean) => void;
}

export default function Header({
  onSummon,
  isCustomizeOpen,
  onToggleCustomize,
  isGenerating = false,
  aiTask,
  onCancel,
  // Debug panel props
  isDebugPanelVisible = false,
  onDebugPanelVisibilityChange,
}: HeaderProps) {
  const [prompt, setPrompt] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  const { theme, setTheme } = useTheme();
  const { t } = useLanguage();

  // Default prompt per task; only replace when empty or when previous default was used
  const taskDefaultPrompt = useMemo(
    () => ({
      "object-removal": "remove object",
      "object-insert": "insert object",
      "white-balance": "white balance this image",
    }),
    []
  );
  const previousTaskRef = useRef<typeof aiTask>(aiTask);

  useEffect(() => {
    const defaultPrompt = taskDefaultPrompt[aiTask ?? "object-removal"];
    const previousDefault =
      previousTaskRef.current && taskDefaultPrompt[previousTaskRef.current];

    const shouldReplace =
      ((!prompt.trim() || prompt === previousDefault) &&
        prompt !== defaultPrompt) ||
      !prompt.trim();

    if (shouldReplace) {
      startTransition(() => setPrompt(defaultPrompt));
    }

    previousTaskRef.current = aiTask;
  }, [aiTask, prompt, taskDefaultPrompt]);

  // Ensure focusable elements outside dialog don't retain focus when dialog opens
  useEffect(() => {
    if (!isSettingsOpen) return;

    // Use MutationObserver to watch for aria-hidden changes and blur focused elements
    const observer = new MutationObserver(() => {
      const activeElement = document.activeElement as HTMLElement;
      if (!activeElement) return;

      // Check if the active element is inside an aria-hidden container
      let parent = activeElement.parentElement;
      while (parent && parent !== document.body) {
        if (parent.getAttribute("aria-hidden") === "true") {
          // If focused element is inside aria-hidden container, blur it
          // The dialog will handle focus management
          activeElement.blur();
          break;
        }
        parent = parent.parentElement;
      }
    });

    // Observe the document body for aria-hidden attribute changes
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["aria-hidden"],
      subtree: true,
    });

    // Also check immediately after a short delay to catch initial state
    const timeoutId = setTimeout(() => {
      const activeElement = document.activeElement as HTMLElement;
      if (!activeElement) return;

      let parent = activeElement.parentElement;
      while (parent && parent !== document.body) {
        if (parent.getAttribute("aria-hidden") === "true") {
          activeElement.blur();
          break;
        }
        parent = parent.parentElement;
      }
    }, 100);

    return () => {
      observer.disconnect();
      clearTimeout(timeoutId);
    };
  }, [isSettingsOpen]);

  const handleSubmit = () => {
    // 🔍 DEBUG: Log when handleSubmit is called
    console.log("🎯 [Header handleSubmit] Called", {
      timestamp: new Date().toISOString(),
      prompt,
      aiTask,
      stackTrace: new Error().stack?.split('\n').slice(2, 5).join('\n'),
    });

    const promptValue = prompt.trim();
    if (aiTask === "white-balance" || promptValue) {
      onSummon(promptValue);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  return (
    <header
      className="bg-primary-bg border-b border-secondary-bg px-2 py-1"
      style={{ boxShadow: "none" }}
    >
      <div className="flex items-center justify-between gap-2 min-h-auto">
        {/* Logo */}
        <div style={{ flexShrink: 0 }}>
          <Image
            src="/logo.svg"
            alt="Artmancer"
            width={180}
            height={48}
            style={{ height: 48, width: "auto" }}
            priority
          />
        </div>

        {/* Input Field and Edit Button */}
        <div
          className="flex items-center gap-4 relative"
          style={{ flex: 1, maxWidth: "800px", marginLeft: 0, marginRight: 0 }}
        >
          <input
            type="text"
            placeholder={t("header.placeholder")}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isGenerating}
            className={`w-full rounded-md border-2 px-3 py-2 text-sm outline-none transition-colors ${
              isGenerating ? "opacity-50 cursor-not-allowed" : ""
            }`}
            style={{
              backgroundColor: "transparent",
              borderColor: "var(--primary-accent)",
              color: "var(--text-primary)",
            }}
          />
          {isGenerating && onCancel ? (
            <button
              onClick={onCancel}
              className="shrink-0 h-10 px-3 bg-red-600 text-white font-semibold rounded flex items-center justify-center"
              style={{ fontSize: "0.875rem" }}
            >
              {t("header.cancel")}
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={
                isGenerating || (!prompt.trim() && aiTask !== "white-balance")
              }
              className={`shrink-0 h-10 px-4 font-semibold rounded text-white flex items-center justify-center gap-2 ${
                isGenerating || (!prompt.trim() && aiTask !== "white-balance")
                  ? "opacity-50 cursor-not-allowed"
                  : "bg-primary-accent hover:bg-highlight-accent"
              }`}
              style={{ fontSize: "0.875rem" }}
            >
              <Wand2 size={16} />
              {isGenerating ? t("header.generating") : t("header.edit")}
            </button>
          )}
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Settings Button */}
          <button
            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
            aria-label={t("header.settings")}
            className="btn-interactive w-10 h-10 inline-flex items-center justify-center rounded bg-secondary-bg text-text-secondary hover:bg-primary-accent hover:text-white transition-colors"
          >
            <Settings size={18} />
          </button>
          <button
            onClick={onToggleCustomize}
            aria-label={t("header.toggleSidebar")}
            className={`btn-interactive w-10 h-10 inline-flex items-center justify-center rounded transition-colors ${
              isCustomizeOpen
                ? "bg-primary-accent text-white"
                : "bg-secondary-bg text-text-secondary hover:bg-primary-accent hover:text-white"
            }`}
          >
            <Menu size={18} />
          </button>
        </div>
      </div>

      {/* Settings Modal (Radix Dialog) */}
      <Dialog.Root
        open={isSettingsOpen}
        onOpenChange={(open: boolean) => setIsSettingsOpen(open)}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay fixed inset-0 z-1300 bg-black/50 backdrop-blur-md" />
          <Dialog.Content
            ref={settingsRef}
            className="dialog-content fixed z-1301 left-1/2 top-1/2 w-full max-w-[700px] -translate-x-1/2 -translate-y-1/2 bg-primary-bg border border-border-color rounded-lg shadow-xl outline-none"
          >
            <div className="flex items-center justify-between pb-2 px-3 pt-2 border-b border-border-color text-text-primary font-semibold">
                <Dialog.Title className="text-text-primary text-base font-semibold">
                  {t("settings.title")}
                </Dialog.Title>
                <Dialog.Close asChild>
                <button className="p-1 rounded text-text-secondary hover:bg-secondary-bg">
                    <X size={18} />
                </button>
                </Dialog.Close>
            </div>

            <div className="px-4 pt-4 pb-3">
              <div className="flex flex-col gap-6">
                  {/* Theme Setting */}
                <div>
                  <div className="flex items-center gap-1 mb-2 text-text-primary font-semibold text-sm">
                    <Sun color="var(--primary-accent)" size={16} />
                    {t("settings.theme")}
                  </div>
                  <ToggleGroup.Root
                    type="single"
                    value={theme}
                    onValueChange={(value: string) => {
                      if (value === "light" || value === "dark") {
                        setTheme(value);
                      }
                    }}
                    className="grid grid-cols-2 gap-1"
                  >
                    <ToggleGroup.Item
                      value="light"
                      className={`flex items-center justify-center gap-1 rounded border px-2 py-1 text-sm transition-colors ${
                        theme === "light"
                          ? "bg-primary-accent text-white border-primary-accent"
                          : "bg-primary-bg text-text-secondary border-border-color hover:bg-secondary-bg"
                      }`}
                    >
                      <Sun size={16} />
                      <span>{t("settings.light")}</span>
                      {theme === "light" && <Check size={16} />}
                    </ToggleGroup.Item>
                    <ToggleGroup.Item
                      value="dark"
                      className={`flex items-center justify-center gap-1 rounded border px-2 py-1 text-sm transition-colors ${
                        theme === "dark"
                          ? "bg-primary-accent text-white border-primary-accent"
                          : "bg-primary-bg text-text-secondary border-border-color hover:bg-secondary-bg"
                      }`}
                    >
                      <Moon size={16} />
                      <span>{t("settings.dark")}</span>
                      {theme === "dark" && <Check size={16} />}
                    </ToggleGroup.Item>
                  </ToggleGroup.Root>
                </div>

                {/* Debug Panel Setting */}
                <div>
                  <div className="flex items-center gap-1 mb-2 text-text-primary font-semibold text-sm">
                    <Bug color="var(--primary-accent)" size={16} />
                    {t("settings.debug")}
                  </div>
                  <label className="flex items-center gap-2">
                    <RadixSwitch.Root
                      checked={isDebugPanelVisible}
                      onCheckedChange={(checked: boolean | "indeterminate") =>
                        onDebugPanelVisibilityChange?.(!!checked)
                      }
                      className="relative h-5 w-9 cursor-pointer rounded-full bg-border-color data-[state=checked]:bg-primary-accent transition-colors"
                    >
                      <RadixSwitch.Thumb className="block h-4 w-4 rounded-full bg-white shadow transition-transform translate-x-0.5 data-[state=checked]:translate-x-4" />
                    </RadixSwitch.Root>
                    <span className="text-text-primary text-sm">
                      {isDebugPanelVisible
                        ? t("settings.debugEnabled")
                        : t("settings.debugDisabled")}
                    </span>
                  </label>
                </div>
              </div>
            </div>

            <div className="px-3 pb-2">
              <Dialog.Close asChild>
                <button
                  className="w-full rounded border border-border-color bg-secondary-bg text-text-primary hover:bg-primary-accent hover:text-white hover:border-primary-accent py-2 transition-colors"
                >
                  {t("settings.close")}
                </button>
              </Dialog.Close>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </header>
  );
}
