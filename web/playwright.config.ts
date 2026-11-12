import { defineConfig, devices } from '@playwright/test';

// PREVIEW_PORT: only for machines where another app already holds 4173
const PORT = Number(process.env.PREVIEW_PORT ?? 4173);
const BASE = `http://localhost:${PORT}/saffron-adulteration-detection/`;

/**
 * E2E tests run against the production build (`npm run build` first). Firefox and WebKit run the core
 * flow; select projects with --project. Headless Chromium uses software WebGL (SwiftShader), so the first
 * model load and each inference are slower than on a real GPU; timeouts are generous accordingly.
 */
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 240_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: BASE,
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] }, grep: /@cross-browser/ },
    { name: 'webkit', use: { ...devices['Desktop Safari'] }, grep: /@cross-browser/ },
    { name: 'mobile', use: { ...devices['Pixel 7'] }, grep: /@cross-browser/ },
  ],
  webServer: {
    command: PORT === 4173 ? 'npm run preview' : `npx vite preview --port ${PORT} --strictPort`,
    url: BASE,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
