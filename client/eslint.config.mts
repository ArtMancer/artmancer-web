import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import pluginReact from "eslint-plugin-react";
import pluginReactHooks from "eslint-plugin-react-hooks";
import { defineConfig } from "eslint/config";

export default defineConfig([
  {
    ignores: [
      ".next/**",
      ".turbo/**",
      ".vscode/**",
      ".cursor/**",
      "node_modules/**",
      "dist/**",
      "build/**",
      "**/.env*",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  pluginReact.configs.flat.recommended,
  { 
    files: ["**/*.{js,mjs,cjs,ts,mts,cts,jsx,tsx}"],
    plugins: {
      "react-hooks": pluginReactHooks,
    },
    languageOptions: { 
      globals: globals.browser,
    },
    settings: {
      react: {
        version: "detect",
      },
    },
    rules: {
      "react/react-in-jsx-scope": "off", // Not needed in Next.js (React 17+)
      "@typescript-eslint/no-unused-vars": "warn", // Downgrade to warning
      "@typescript-eslint/no-explicit-any": "warn", // Downgrade to warning
      "@typescript-eslint/ban-ts-comment": "warn", // Downgrade @ts-ignore warning
      "@typescript-eslint/no-unsafe-function-type": "warn", // Downgrade Function type warning
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
]);
