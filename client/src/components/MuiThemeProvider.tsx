"use client";

import { ThemeProvider as MuiThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { createMuiTheme } from "@/theme/muiTheme";
import { ReactNode } from "react";

interface MuiThemeProviderWrapperProps {
  children: ReactNode;
}

export default function MuiThemeProviderWrapper({
  children,
}: MuiThemeProviderWrapperProps) {
  const theme = createMuiTheme();

  return (
    <MuiThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </MuiThemeProvider>
  );
}
