import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendTarget = process.env.BACKEND_URL || 'http://localhost:8000';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/chat': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/analyze-image': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/voice': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/ingest-document': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
});
