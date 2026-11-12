/// <reference types="vitest/config" />
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const MODEL_DIR = join(__dirname, 'public', 'model');
const GATE_DIR = join(__dirname, 'public', 'gate');
const modelFiles = (dir: string) =>
  existsSync(dir)
    ? readdirSync(dir)
        .filter((f) => f === 'model.json' || f.endsWith('.bin'))
        .sort()
    : [];

/** Content hash of a shipped TF.js model: the IndexedDB cache key, so a new model invalidates old caches. */
function modelVersion(dir: string): string {
  const hash = createHash('sha256');
  for (const file of modelFiles(dir)) hash.update(file).update(readFileSync(join(dir, file)));
  return hash.digest('hex').slice(0, 12);
}

function modelBytes(dir: string): number {
  return modelFiles(dir).reduce((n, f) => n + statSync(join(dir, f)).size, 0);
}

export default defineConfig({
  base: '/saffron-adulteration-detection/',
  plugins: [react(), tailwindcss()],
  define: {
    __MODEL_VERSION__: JSON.stringify(modelVersion(MODEL_DIR)),
    __MODEL_BYTES__: JSON.stringify(modelBytes(MODEL_DIR)),
    // the input check's image encoder (TinyCLIP ViT-8M/16), downloaded in parallel with the classifier
    __GATE_VERSION__: JSON.stringify(modelVersion(GATE_DIR)),
    __GATE_BYTES__: JSON.stringify(modelBytes(GATE_DIR)),
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
