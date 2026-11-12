import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Locator, type Page } from '@playwright/test';

interface Reference {
  id: string;
  trueClass: string;
  predicted: string;
  probabilities: number[];
}

const FIXTURES = fileURLToPath(new URL('../fixtures/', import.meta.url));
const CLASSES = ['C1', 'C2', 'C3', 'C4', 'C5'];
const references = JSON.parse(readFileSync(join(FIXTURES, 'sample-references.json'), 'utf8')) as Reference[];
const reference = (id: string) => references.find((r) => r.id === id)!;
/** Keras (Python decode) vs TF.js (browser decode, WebGL/WASM float math). */
const MAX_PROB_DIFF = 0.03;

async function waitForModel(page: Page) {
  await expect(page.getByTestId('model-status')).toHaveAttribute('data-state', 'ready', { timeout: 180_000 });
}

/** CC BY 4.0, 多多123, Wikimedia Commons (docs/ood/credits.json): red yarn, which the input check turns away. */
const OOD_FIXTURE = join(FIXTURES, '..', '..', '..', 'docs', 'ood', 'red_thread_yarn.jpg');

async function lastInputCheck(page: Page): Promise<{ score: number; passed: boolean } | null> {
  return page.evaluate(() =>
    (
      globalThis as unknown as {
        __saffronDebug: { lastInputCheck(): { score: number; passed: boolean } | null };
      }
    ).__saffronDebug.lastInputCheck(),
  );
}

/** The result panel for one input (an outgoing result can linger briefly during its exit animation). */
const resultFor = (page: Page, source: string) =>
  page.locator(`[data-testid="result"][data-source="${source}"]`);

async function readProbabilities(result: Locator): Promise<number[]> {
  const rows = result.getByTestId('prob-row');
  await expect(rows).toHaveCount(5);
  const byClass: Record<string, number> = {};
  for (const row of await rows.all()) {
    byClass[(await row.getAttribute('data-class'))!] = Number(await row.getAttribute('data-probability'));
  }
  return CLASSES.map((c) => byClass[c]);
}

async function expectMatchesReference(page: Page, ref: Reference, source = ref.id.replace('fixtures/', '')) {
  const result = resultFor(page, source);
  await expect(result).toBeVisible();
  await expect(result).toHaveAttribute('data-predicted', ref.predicted);
  const probs = await readProbabilities(result);
  const diff = Math.max(...probs.map((p, i) => Math.abs(p - ref.probabilities[i])));
  expect(diff, `${ref.id}: max |prob diff| vs Keras`).toBeLessThanOrEqual(MAX_PROB_DIFF);
  return diff;
}

async function tensorCount(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const debug = (globalThis as unknown as { __saffronDebug: { memory(): Promise<{ numTensors: number }> } })
      .__saffronDebug;
    return (await debug.memory()).numTensors;
  });
}

test.describe('saffron check', () => {
  test('classifies a held-out sample and explains it @cross-browser', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Is that saffron');
    await waitForModel(page);
    console.log('backend:', await page.getByTestId('backend').textContent());

    await page.getByTestId('sample').nth(3).click();
    const ref = reference('sample-04.jpg');
    const diff = await expectMatchesReference(page, ref);
    console.log(`sample-04: predicted ${ref.predicted}, max diff ${diff.toFixed(4)}`);
    await expect(page.getByTestId('verdict')).toBeVisible();
    await expect(resultFor(page, 'sample-04.jpg').getByTestId('sample-reveal')).toContainText(
      `Ground truth: ${ref.trueClass}`,
    );
    await expect(page.locator('#result-heading')).toBeFocused();
    await expect(page.getByTestId('cam-overlay')).toBeVisible();
  });

  test('every sample matches its Keras reference, without leaking tensors', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    const samples = page.getByTestId('sample');
    await expect(samples).toHaveCount(10);
    let baseline: number | null = null;
    for (let i = 0; i < 10; i++) {
      await samples.nth(i).click();
      const file = (await samples.nth(i).getAttribute('data-file'))!;
      await expect(resultFor(page, file).getByTestId('sample-reveal')).toBeVisible();
      const diff = await expectMatchesReference(page, reference(file));
      const check = await lastInputCheck(page);
      expect(check?.passed, `${file}: passes the input check`).toBe(true);
      await expect(page.getByTestId('unchecked-label')).toHaveCount(0);
      const tensors = await tensorCount(page);
      baseline ??= tensors;
      console.log(
        `${file}: ${reference(file).predicted} (true ${reference(file).trueClass}), diff ${diff.toFixed(4)}, input check ${check!.score.toFixed(4)}, tensors ${tensors}`,
      );
      expect(tensors, 'tensor count must not grow between predictions').toBe(baseline);
    }
  });

  test('flags the hard C2 sample as low confidence', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await page.locator('[data-testid="sample"][data-file="sample-02.jpg"]').click();
    await expectMatchesReference(page, reference('sample-02.jpg'));
    await expect(page.getByTestId('low-confidence')).toContainText('always picks one of its five classes');
  });

  test('applies EXIF orientation like PIL.ImageOps.exif_transpose', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await page.getByTestId('file-input').setInputFiles(join(FIXTURES, 'exif-rotated.jpg'));
    await expectMatchesReference(page, reference('fixtures/exif-rotated.jpg'));
  });

  test('decodes an iPhone HEIC photo @cross-browser', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await page.getByTestId('file-input').setInputFiles(join(FIXTURES, 'sample.heic'));
    // different HEVC decoders (libheif in pillow-heif vs heic2any / Safari) -> compare class and a looser bound
    const ref = reference('fixtures/sample.heic');
    const result = resultFor(page, 'sample.heic');
    await expect(result).toHaveAttribute('data-predicted', ref.predicted, { timeout: 120_000 });
    const probs = await readProbabilities(result);
    expect(Math.max(...probs.map((p, i) => Math.abs(p - ref.probabilities[i])))).toBeLessThanOrEqual(0.08);
  });

  test('rejects a file that is not an image @cross-browser', async ({ page }) => {
    await page.goto('./');
    await page.getByTestId('file-input').setInputFiles(join(FIXTURES, 'not-an-image.txt'));
    await expect(page.getByTestId('upload-error')).toContainText("doesn't look like a photo");
    await expect(page.getByTestId('result')).toHaveCount(0);
  });

  test('rejects photos over 25 MB', async ({ page }) => {
    await page.goto('./');
    const big = Buffer.alloc(26_000_000);
    big.set([0xff, 0xd8, 0xff, 0xe0]);
    await page
      .getByTestId('file-input')
      .setInputFiles({ name: 'huge.jpg', mimeType: 'image/jpeg', buffer: big });
    await expect(page.getByTestId('upload-error')).toContainText('under 25 MB');
  });

  test('WASM backend gives the same answer', async ({ page }) => {
    await page.goto('./?backend=wasm');
    await waitForModel(page);
    await expect(page.getByTestId('backend')).toContainText('WebAssembly');
    await page.locator('[data-testid="sample"][data-file="sample-01.jpg"]').click();
    await expectMatchesReference(page, reference('sample-01.jpg'));
  });

  test('turns away a photo that is not a saffron thread, and can analyse it anyway @cross-browser', async ({
    page,
  }) => {
    await page.goto('./');
    await waitForModel(page);
    await page.getByTestId('file-input').setInputFiles(OOD_FIXTURE);
    const notice = page.getByTestId('input-check-notice');
    await expect(notice).toBeVisible();
    await expect(
      page.getByRole('heading', { name: "This doesn't look like a single saffron thread on paper." }),
    ).toBeFocused();
    await expect(page.getByTestId('verdict')).toHaveCount(0);
    await expect(page.getByTestId('prob-row')).toHaveCount(0);
    await expect(page.getByTestId('live-region')).toContainText(
      "This doesn't look like a single saffron thread on paper.",
    );
    expect((await lastInputCheck(page))?.passed).toBe(false);

    await page.getByRole('link', { name: 'Photo tips' }).click();
    await expect(page.locator('#photo-guidelines')).toHaveAttribute('open', '');

    await page.getByTestId('analyse-anyway').click();
    await expect(page.getByTestId('result')).toBeVisible();
    await expect(page.getByTestId('unchecked-label')).toHaveText(
      'Unchecked input: this photo did not look like a single saffron thread on paper, so treat the result with extra caution.',
    );
    await expect(page.getByTestId('verdict')).toBeVisible();
    await expect(page.getByTestId('live-region')).toContainText('Unchecked input');

    await page.getByRole('button', { name: /Classify another/ }).click();
    await page.getByTestId('file-input').setInputFiles(OOD_FIXTURE);
    await expect(notice).toBeVisible();
    await page.getByRole('button', { name: 'Try another photo' }).click();
    await expect(notice).toHaveCount(0);
  });

  test('the input check notice works with the keyboard @cross-browser', async ({
    page,
    isMobile,
    browserName,
  }) => {
    test.skip(isMobile, 'no keyboard on the phone profile');
    test.skip(browserName === 'webkit' && process.platform === 'win32', 'WebKit on Windows ignores Tab');
    await page.goto('./');
    await waitForModel(page);
    await page.getByTestId('file-input').setInputFiles(OOD_FIXTURE);
    await expect(page.getByRole('heading', { name: /single saffron thread on paper/ })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Try another photo' })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Photo tips' })).toBeFocused();
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Analyse anyway' })).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('unchecked-label')).toBeVisible();
    await expect(page.locator('#result-heading')).toBeFocused();
  });

  test('repeated input checks do not leak tensors', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    await page.getByTestId('sample').first().click();
    await expect(page.getByTestId('result')).toBeVisible();
    const baseline = await tensorCount(page);
    for (let i = 0; i < 3; i++) {
      await page.getByRole('button', { name: /Classify another/ }).click();
      await page.getByTestId('file-input').setInputFiles(OOD_FIXTURE);
      await expect(page.getByTestId('input-check-notice')).toBeVisible();
      await page.getByTestId('analyse-anyway').click();
      await expect(page.getByTestId('unchecked-label')).toBeVisible();
    }
    expect(await tensorCount(page)).toBe(baseline);
  });

  test('caches the model in IndexedDB for the next visit', async ({ page }) => {
    await page.goto('./');
    await waitForModel(page);
    // saving to IndexedDB happens after warm-up; wait until it is listed
    await expect
      .poll(
        () =>
          page.evaluate(async () => {
            const dbs = (await indexedDB.databases?.()) ?? [];
            return dbs.some((d) => d.name === 'tensorflowjs');
          }),
        { timeout: 60_000 },
      )
      .toBe(true);
    await page.waitForTimeout(3000);
    await page.reload();
    await waitForModel(page);
    await expect(page.getByTestId('model-status')).toContainText('loaded from browser cache');
  });
});
