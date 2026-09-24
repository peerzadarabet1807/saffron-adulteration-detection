/// <reference types="vitest/config" />
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const MODEL_DIR = join(__dirname, 'public', 'model');

/** Content hash of the shipped TF.js model: the IndexedDB cache key, so a new model invalidates old caches. */
function modelVersion(): string {
  const hash = createHash('sha256');
  for (const file of readdirSync(MODEL_DIR).sort())
    hash.update(file).update(readFileSync(join(MODEL_DIR, file)));
  return hash.digest('hex').slice(0, 12);
}

function modelBytes(): number {
  return readdirSync(MODEL_DIR).reduce((n, f) => n + statSync(join(MODEL_DIR, f)).size, 0);
}

export default defineConfig({
  base: '/saffron-adulteration-detection/',
  plugins: [react(), tailwindcss()],
  define: {
    __MODEL_VERSION__: JSON.stringify(modelVersion()),
    __MODEL_BYTES__: JSON.stringify(modelBytes()),
  },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    // tfjs lives in the worker chunk (~1.3 MB); the main bundle stays small.
    chunkSizeWarningLimit: 1600,
  },
  preview: { port: 4173, strictPort: true },
  test: {
    environment: 'jsdom',
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/setup.ts'],
    css: false,
  },
});
