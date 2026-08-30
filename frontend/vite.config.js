// Vite 配置:开发服务器代理 /api 到后端 FastAPI,构建产物输出到 dist
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:9000',
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1024,
  },
})
