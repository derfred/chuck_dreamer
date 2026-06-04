// Flat ESLint config (ESLint 9) for the webui frontend. Scoped to this package
// — `npm run lint` is `eslint .` run from here, and this is the only JS/TS in
// src/fessel (the Pi and backend projects lint with ruff via the Makefile).
//
// Rule sets: typescript-eslint recommended + the React Hooks rules the source
// already relies on (useLiveSession.ts carries `react-hooks/exhaustive-deps`
// disable comments, so those rules must be active for the disables to be real).

import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";

export default tseslint.config(
  // Build output and generated artifacts are not ours to lint. src/generated
  // is emitted by tools/generate-types.sh and must stay byte-identical to the
  // generator output (the make check-types sync guard compares against it).
  {
    ignores: ["dist/**", "node_modules/**", "**/*.tsbuildinfo", "src/generated/**"],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  // App + test source: browser environment, React Hooks rules on.
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    languageOptions: {
      globals: { ...globals.browser },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Allow intentional unused args/vars when prefixed with `_` (e.g. the
      // no-op stub params in test fakes and bus-message handlers).
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },

  // Test files: vitest provides describe/it/expect/vi as globals
  // (vite.config.ts sets test.globals: true), and they touch Node-ish globals.
  {
    files: ["src/**/*.test.{ts,tsx}", "src/test/**/*.ts"],
    languageOptions: {
      globals: { ...globals.node },
    },
  },

  // Vite/Vitest config and other tooling files run in Node.
  {
    files: ["*.config.{ts,js}"],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
);
