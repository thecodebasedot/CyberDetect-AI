import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api and /dashboard to the FastAPI backend so the
// frontend can call the Python API without CORS during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/dashboard': 'http://127.0.0.1:8000',
    },
  },
})
