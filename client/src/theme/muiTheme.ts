"use client";

import { createTheme } from "@mui/material/styles";

// Create theme based on CSS variables
export const createMuiTheme = () => {
  return createTheme({
    palette: {
      mode: "dark",
      primary: {
        main: "#6a0dad", // --primary-accent
        light: "#9d4edd", // --highlight-accent
        dark: "#4a0a7d",
        contrastText: "#ffffff",
      },
      secondary: {
        main: "#9d4edd", // --highlight-accent
        light: "#c77dff",
        dark: "#6a0dad",
        contrastText: "#ffffff",
      },
      background: {
        default: "#0b0b0d", // --primary-bg
        paper: "#1a1a1f", // --secondary-bg
      },
      text: {
        primary: "#e0e0e0", // --text-primary
        secondary: "#a0a0a8", // --text-secondary
      },
      success: {
        main: "#3ddc97", // --success
      },
      error: {
        main: "#e63946", // --warning
      },
      divider: "#2a2a2f", // --border-color
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: "none",
            borderRadius: "8px",
            fontWeight: 500,
            "&:hover": {
              backgroundColor: "#9d4edd",
            },
          },
          contained: {
            backgroundColor: "#6a0dad",
            color: "#ffffff",
            "&:hover": {
              backgroundColor: "#9d4edd",
            },
            "&:disabled": {
              backgroundColor: "#1a1a1f",
              color: "#a0a0a8",
            },
          },
          outlined: {
            borderColor: "#6a0dad",
            color: "#6a0dad",
            "&:hover": {
              borderColor: "#9d4edd",
              backgroundColor: "rgba(106, 13, 173, 0.1)",
            },
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            "& .MuiOutlinedInput-root": {
              backgroundColor: "#0b0b0d",
              color: "#e0e0e0",
              "& fieldset": {
                borderColor: "#2a2a2f",
              },
              "&:hover fieldset": {
                borderColor: "#6a0dad",
              },
              "&.Mui-focused fieldset": {
                borderColor: "#6a0dad",
              },
            },
            "& .MuiInputLabel-root": {
              color: "#a0a0a8",
              "&.Mui-focused": {
                color: "#6a0dad",
              },
            },
          },
        },
      },
      MuiCheckbox: {
        styleOverrides: {
          root: {
            color: "#2a2a2f",
            "&.Mui-checked": {
              color: "#6a0dad",
            },
            "&:hover": {
              backgroundColor: "rgba(106, 13, 173, 0.1)",
            },
          },
        },
      },
      MuiToggleButton: {
        styleOverrides: {
          root: {
            color: "#a0a0a8",
            borderColor: "#2a2a2f",
            backgroundColor: "#1a1a1f",
            "&.Mui-selected": {
              backgroundColor: "#6a0dad",
              color: "#ffffff",
              "&:hover": {
                backgroundColor: "#9d4edd",
              },
            },
            "&:hover": {
              backgroundColor: "rgba(106, 13, 173, 0.2)",
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            backgroundColor: "#1a1a1f",
            color: "#e0e0e0",
            borderColor: "#2a2a2f",
          },
        },
      },
    },
    typography: {
      fontFamily: '"Fira Code", "JetBrains Mono", "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
      button: {
        textTransform: "none",
        fontWeight: 500,
      },
    },
  });
};

