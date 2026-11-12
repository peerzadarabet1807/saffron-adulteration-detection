/**
 * Capture the README screenshots from the running production build (never from mock-ups).
 *
 *   npm run build && npm run preview      # in one terminal
 *   npm run screenshots                   # in another  -> ../docs/screenshots/*.png
 *
 * Uses the real GPU where the machine has one (headless Chromium otherwise falls back to software WebGL,
 * which makes the inference times shown in the UI unrealistically slow).
 */
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, devices, type Browser, type BrowserContextOptions, type Page } from '@playwright/test';

const BASE = process.env.BASE_URL ?? 'http://localhost:4173/saffron-adulteration-detection/';
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = join(ROOT, 'docs', 'screenshots');
const GPU_ARGS = ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist'];
const DESKTOP: BrowserContextOptions = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 };
/** result views: tall enough for the whole result card */
const DESKTOP_TALL: BrowserContextOptions = { viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 };

async function launch(): Promise<Browser> {
  try {
    return await chromium.launch({ channel: 'chromium', args: GPU_ARGS });
  } catch {
    return chromium.launch();
  }
}

async function settle(page: Page, ms = 900) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(ms); // let entrance animations finish
}

async function modelReady(page: Page) {
  await page
    .getByTestId('model-status')
    .and(page.locator('[data-state="ready"]'))
    .waitFor({ timeout: 180_000 });
}

/** Scroll an element to just below the sticky header. */
async function scrollToId(page: Page, id: string) {
  await page.evaluate((target) => {
    const el = document.getElementById(target);
    const header = document.querySelector('header')?.getBoundingClientRect().height ?? 64;
    if (el) {
      const top = el.getBoundingClientRect().top + window.scrollY - header - 16;
      window.scrollTo({ top, behavior: 'instant' });
    }
  }, id);
}

async function showResult(page: Page, source: string) {
  await page.locator(`[data-testid="result"][data-source="${source}"]`).waitFor({ timeout: 120_000 });
  await settle(page, 1400); // entrance animation and the app's own smooth scroll
  await scrollToId(page, 'result');
  await settle(page, 400);
}

async function shot(page: Page, name: string) {
  const path = join(OUT, name);
  await page.screenshot({ path });
  console.log('saved', path);
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const browser = await launch();

  // 01, 06: desktop 1440 x 900, light
  {
    const ctx = await browser.newContext({ ...DESKTOP, colorScheme: 'light' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await settle(page);
    await shot(page, '01-landing.png');
    await scrollToId(page, 'about');
    await settle(page);
    await shot(page, '06-about-model.png');
    await ctx.close();
  }

  // 03, 04, 05: results, light
  {
    const ctx = await browser.newContext({ ...DESKTOP_TALL, colorScheme: 'light' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);

    await page.locator('[data-testid="sample"][data-file="sample-09.jpg"]').click(); // C1, typical
    await showResult(page, 'sample-09.jpg');
    await shot(page, '03-result-pure.png');

    await page.locator('[data-testid="sample"][data-file="sample-06.jpg"]').click(); // C4, typical
    await showResult(page, 'sample-06.jpg');
    await shot(page, '04-result-counterfeit-gradcam.png');

    // a real photo that is not saffron (CC BY 4.0, see docs/ood/README.md): the input check turns it away
    await page.getByTestId('file-input').setInputFiles(join(ROOT, 'docs', 'ood', 'red_thread_yarn.jpg'));
    await page.getByTestId('input-check-notice').waitFor({ timeout: 120_000 });
    await page.waitForTimeout(900);
    await shot(page, '05-low-confidence.png');
    // "Analyse anyway": the classifier's answer, labelled as unchecked
    await page.getByTestId('analyse-anyway').click();
    await showResult(page, 'red_thread_yarn.jpg');
    await shot(page, '09-unchecked.png');
    await ctx.close();
  }

  // 10: the input check notice, dark
  {
    const ctx = await browser.newContext({ ...DESKTOP, colorScheme: 'dark' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await page.getByTestId('file-input').setInputFiles(join(ROOT, 'docs', 'ood', 'red_thread_yarn.jpg'));
    await page.getByTestId('input-check-notice').waitFor({ timeout: 120_000 });
    await page.waitForTimeout(900);
    await shot(page, '10-input-check-dark.png');
    await ctx.close();
  }

  // 02: model download in progress
  {
    const ctx = await browser.newContext({ ...DESKTOP, colorScheme: 'light' });
    // shards download in parallel; stagger them so the progress bar moves visibly
    await ctx.route(/\/(model|gate)\/group1-shard/, async (route) => {
      const n = Number(/shard(\d+)of/.exec(route.request().url())?.[1] ?? 1);
      await new Promise((r) => setTimeout(r, 700 * n));
      await route.continue();
    });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await page.waitForFunction(() => {
      const v = Number(document.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow'));
      return v >= 40 && v <= 60;
    });
    await shot(page, '02-model-loading.png');
    await ctx.close();
  }

  // 07: mobile
  {
    const ctx = await browser.newContext({
      ...devices['iPhone 13'],
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 2,
      colorScheme: 'light',
    });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await page.locator('[data-testid="sample"][data-file="sample-04.jpg"]').click(); // C5, typical
    await showResult(page, 'sample-04.jpg');
    await shot(page, '07-mobile.png');
    await ctx.close();
  }

  // 08: dark mode
  {
    const ctx = await browser.newContext({ ...DESKTOP_TALL, colorScheme: 'dark' });
    const page = await ctx.newPage();
    await page.goto(BASE);
    await modelReady(page);
    await page.locator('[data-testid="sample"][data-file="sample-03.jpg"]').click(); // C2, typical
    await showResult(page, 'sample-03.jpg');
    await shot(page, '08-dark-mode.png');
    await ctx.close();
  }

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
