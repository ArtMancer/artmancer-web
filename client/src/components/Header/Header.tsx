import Image from "next/image";
import {
  MdSettings,
  MdMenu,
  MdLanguage,
  MdLightMode,
  MdDarkMode,
  MdCheck,
  MdBugReport,
} from "react-icons/md";
import { useState, useRef, useEffect, useCallback } from "react";
import { useTheme, useLanguage, useServer } from "@/contexts";
import ServerControl from "@/components/ServerControl";
import {
  AppBar,
  Toolbar,
  TextField,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Switch,
  FormControlLabel,
  Stack,
  Grid,
  Backdrop,
  Fade,
} from "@mui/material";
import {
  Settings as SettingsIcon,
  Menu as MenuIcon,
  Close as CloseIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Language as LanguageIcon,
  BugReport as BugReportIcon,
  Check as CheckIcon,
} from "@mui/icons-material";

interface HeaderProps {
  onSummon: (prompt: string) => void;
  isCustomizeOpen: boolean;
  onToggleCustomize: () => void;
  isGenerating?: boolean;
  aiTask?: "white-balance" | "object-insert" | "object-removal" | "evaluation";
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
  const [isClosing, setIsClosing] = useState(false);
  const [isOpening, setIsOpening] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);

  const { theme, toggleTheme } = useTheme();
  const { language, setLanguage, t } = useLanguage();
  const { isReady } = useServer();

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
    if (!isReady) return; // Không cho submit nếu server chưa sẵn sàng
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
    <AppBar
      position="static"
      sx={{
        bgcolor: "var(--panel-bg)",
        borderBottom: 1,
        borderColor: "var(--secondary-bg)",
        boxShadow: "none",
        px: 2,
        py: 1,
      }}
    >
      <Toolbar
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 2,
          minHeight: "auto !important",
        }}
      >
        {/* Logo */}
        <Box sx={{ flexShrink: 0 }}>
          <Image
            src="/logo.svg"
            alt="Artmancer"
            width={180}
            height={48}
            style={{ height: 48, width: "auto" }}
            priority
          />
        </Box>

        {/* Input Field and Edit Button */}
        <Box
          sx={{
            flex: 1,
            maxWidth: { xs: "100%", md: 800 },
            mx: { xs: 0, md: 2 },
            display: "flex",
            gap: 1.5,
            alignItems: "center",
            position: "relative",
          }}
        >
          {!isReady && (
            <Backdrop
              open={!isReady}
              sx={{
                position: "absolute",
                zIndex: 10,
                bgcolor: "rgba(0, 0, 0, 0.1)",
                backdropFilter: "blur(4px)",
                borderRadius: 1,
                border: 2,
                borderStyle: "dashed",
                borderColor: "divider",
              }}
            >
              <Typography
                variant="body2"
                sx={{
                  color: "text.secondary",
                  fontWeight: 500,
                  textAlign: "center",
                }}
              >
                Hệ thống đang Offline
              </Typography>
            </Backdrop>
          )}
          <TextField
            fullWidth
            placeholder={t("header.placeholder")}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isGenerating || !isReady}
            size="small"
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "transparent",
                borderColor: "var(--primary-accent)",
                borderWidth: 2,
                borderStyle: "solid",
                color: "var(--text-primary)",
                fontSize: isReady ? "0.875rem" : "0.75rem",
                "& fieldset": {
                  borderColor: "var(--primary-accent)",
                },
                "&:hover fieldset": {
                  borderColor: "var(--highlight-accent)",
                },
                "&.Mui-focused fieldset": {
                  borderColor: "var(--highlight-accent)",
                },
                "&.Mui-disabled": {
                  opacity: 0.5,
                },
              },
              "& .MuiInputBase-input::placeholder": {
                color: "var(--text-secondary)",
                opacity: 1,
              },
            }}
          />
          {isGenerating && onCancel ? (
            <Button
              variant="contained"
              color="error"
              onClick={onCancel}
              sx={{
                flexShrink: 0,
                height: 48,
                px: 3,
                fontSize: isReady ? "0.875rem" : "0.75rem",
                textTransform: "none",
                fontWeight: 600,
              }}
            >
              {t("header.cancel")}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={
                !isReady ||
                isGenerating ||
                (!prompt.trim() && aiTask !== "white-balance")
              }
              sx={{
                flexShrink: 0,
                height: 48,
                px: 3,
                bgcolor: "var(--primary-accent)",
                color: "white",
                fontSize: isReady ? "0.875rem" : "0.75rem",
                textTransform: "none",
                fontWeight: 600,
                "&:hover": {
                  bgcolor: "var(--highlight-accent)",
                },
                "&.Mui-disabled": {
                  opacity: 0.5,
                },
              }}
            >
              {isGenerating ? t("header.generating") : t("header.edit")}
            </Button>
          )}
        </Box>

        {/* Header Actions */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            flexShrink: 0,
          }}
        >
          {/* Server Control */}
          <ServerControl />

          {/* Settings Button */}
          <IconButton
            onClick={() => {
              if (isSettingsOpen) {
                handleCloseModal();
              } else {
                handleOpenModal();
              }
            }}
            sx={{
              bgcolor: "var(--secondary-bg)",
              color: "var(--text-secondary)",
              width: 48,
              height: 48,
              "&:hover": {
                bgcolor: "var(--primary-accent)",
                color: "white",
              },
            }}
            aria-label={t("header.settings")}
          >
            <SettingsIcon />
          </IconButton>
          <IconButton
            onClick={onToggleCustomize}
            sx={{
              bgcolor: isCustomizeOpen
                ? "var(--primary-accent)"
                : "var(--secondary-bg)",
              color: isCustomizeOpen ? "white" : "var(--text-secondary)",
              width: 48,
              height: 48,
              "&:hover": {
                bgcolor: "var(--primary-accent)",
                color: "white",
              },
            }}
            aria-label={t("header.toggleSidebar")}
          >
            <MenuIcon />
          </IconButton>
        </Box>
      </Toolbar>

      {/* Settings Modal */}
      <Dialog
        open={isSettingsOpen}
        onClose={handleCloseModal}
        maxWidth="sm"
        fullWidth
        disableEnforceFocus={false}
        disableAutoFocus={false}
        disableScrollLock={false}
        TransitionComponent={Fade}
        TransitionProps={{ timeout: 300 }}
        PaperProps={{
          sx: {
            bgcolor: "var(--primary-bg)",
            border: 1,
            borderColor: "var(--border-color)",
            borderRadius: 3,
          },
        }}
        BackdropProps={{
          sx: {
            backdropFilter: "blur(8px)",
            bgcolor: "rgba(0, 0, 0, 0.5)",
          },
        }}
      >
        <DialogTitle
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            pb: 2,
            borderBottom: 1,
            borderColor: "var(--border-color)",
            color: "var(--text-primary)",
            fontWeight: 600,
          }}
        >
          {t("settings.title")}
          <IconButton
            onClick={handleCloseModal}
            size="small"
            sx={{
              color: "var(--text-secondary)",
              "&:hover": {
                bgcolor: "var(--secondary-bg)",
              },
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ pt: 3 }}>
          <Stack spacing={3}>
            {/* Theme Setting */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  color: "var(--text-primary)",
                  fontWeight: 500,
                  mb: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <LightModeIcon
                  sx={{ color: "var(--primary-accent)", fontSize: 18 }}
                />
                {t("settings.theme")}
              </Typography>
              <ToggleButtonGroup
                value={theme}
                exclusive
                onChange={(_, value) => {
                  if (value !== null && value !== theme) {
                    toggleTheme();
                  }
                }}
                fullWidth
                sx={{
                  "& .MuiToggleButton-root": {
                    borderColor: "var(--border-color)",
                    color: "var(--text-secondary)",
                    "&.Mui-selected": {
                      bgcolor: "var(--primary-accent)",
                      color: "white",
                      "&:hover": {
                        bgcolor: "var(--primary-accent)",
                      },
                    },
                  },
                }}
              >
                <ToggleButton value="light">
                  <Stack direction="row" spacing={1} alignItems="center">
                    <LightModeIcon fontSize="small" />
                    <Typography variant="body2">
                      {t("settings.light")}
                    </Typography>
                    {theme === "light" && <CheckIcon fontSize="small" />}
                  </Stack>
                </ToggleButton>
                <ToggleButton value="dark">
                  <Stack direction="row" spacing={1} alignItems="center">
                    <DarkModeIcon fontSize="small" />
                    <Typography variant="body2">
                      {t("settings.dark")}
                    </Typography>
                    {theme === "dark" && <CheckIcon fontSize="small" />}
                  </Stack>
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            {/* Language Setting */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  color: "var(--text-primary)",
                  fontWeight: 500,
                  mb: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <LanguageIcon
                  sx={{ color: "var(--primary-accent)", fontSize: 18 }}
                />
                {t("settings.language")}
              </Typography>
              <ToggleButtonGroup
                value={language}
                exclusive
                onChange={(_, value) => {
                  if (value !== null) {
                    setLanguage(value);
                  }
                }}
                fullWidth
                sx={{
                  "& .MuiToggleButton-root": {
                    borderColor: "var(--border-color)",
                    color: "var(--text-secondary)",
                    "&.Mui-selected": {
                      bgcolor: "var(--primary-accent)",
                      color: "white",
                      "&:hover": {
                        bgcolor: "var(--primary-accent)",
                      },
                    },
                  },
                }}
              >
                <ToggleButton value="en">
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="h6">🇺🇸</Typography>
                    <Typography variant="body2">
                      {t("settings.english")}
                    </Typography>
                    {language === "en" && <CheckIcon fontSize="small" />}
                  </Stack>
                </ToggleButton>
                <ToggleButton value="vi">
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="h6">🇻🇳</Typography>
                    <Typography variant="body2">
                      {t("settings.vietnamese")}
                    </Typography>
                    {language === "vi" && <CheckIcon fontSize="small" />}
                  </Stack>
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            {/* Debug Panel Setting */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  color: "var(--text-primary)",
                  fontWeight: 500,
                  mb: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <BugReportIcon
                  sx={{ color: "var(--primary-accent)", fontSize: 18 }}
                />
                Debug
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={isDebugPanelVisible}
                    onChange={(e) =>
                      onDebugPanelVisibilityChange?.(e.target.checked)
                    }
                    sx={{
                      "& .MuiSwitch-switchBase.Mui-checked": {
                        color: "var(--primary-accent)",
                      },
                      "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track":
                        {
                          bgcolor: "var(--primary-accent)",
                        },
                    }}
                  />
                }
                label={
                  <Typography
                    variant="body2"
                    sx={{ color: "var(--text-primary)" }}
                  >
                    {isDebugPanelVisible ? "Enabled" : "Disabled"}
                  </Typography>
                }
              />
            </Box>
          </Stack>
        </DialogContent>

        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={handleCloseModal}
            fullWidth
            variant="outlined"
            sx={{
              bgcolor: "var(--secondary-bg)",
              color: "var(--text-primary)",
              borderColor: "var(--border-color)",
              "&:hover": {
                bgcolor: "var(--primary-accent)",
                color: "white",
                borderColor: "var(--primary-accent)",
              },
              textTransform: "none",
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </AppBar>
  );
}
