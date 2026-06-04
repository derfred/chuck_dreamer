/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Slice-1 dev: proxy backend API to the local webui-backend.
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    // jsdom gives us window/document/fetch-target for the state-machine and
    // API-client tests. WebRTC (RTCPeerConnection, getStats) is not in jsdom,
    // so the tests stub it — see src/test/setup.ts.
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
