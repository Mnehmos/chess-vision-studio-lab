import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The GUI talks only to the lab API; in dev, proxy to `cvslab serve`.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
