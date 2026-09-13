import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/sample-patients': 'http://127.0.0.1:8000',
    },
  },
});
