import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Core React libraries
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // State management & data fetching
          'vendor-state': ['zustand', '@tanstack/react-query', 'axios'],
          // Monaco Editor (largest dependency)
          'vendor-monaco': ['@monaco-editor/react'],
          // Charts library
          'vendor-charts': ['recharts'],
          // Syntax highlighter
          'vendor-syntax': ['react-syntax-highlighter'],
          // Markdown
          'vendor-markdown': ['react-markdown', 'remark-gfm'],
          // UI utilities
          'vendor-ui': ['lucide-react', 'clsx'],
        },
      },
    },
    // Increase chunk size warning limit (Monaco is inherently large)
    chunkSizeWarningLimit: 600,
  },
})
