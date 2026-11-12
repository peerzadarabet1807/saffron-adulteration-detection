// Lighthouse scores of the production build, median of 3 runs, written to ../docs/lighthouse/summary.json.
//
//   npm run build && npm run preview     # in one terminal
//   node scripts/lighthouse.mjs          # in another
//
// excludingModel: the page with ?preload=0, so the models (47 MB classifier, 17 MB input check) are not fetched
// until the first classification (Lighthouse cannot block a Web Worker's requests). withModel: default
// behaviour, both models download in the background after page load.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const LIGHTHOUSE = 'lighthouse@13.5.0';
const BASE = process.env.BASE_URL ?? 'http://localhost:4173/saffron-adulteration-detection/';
const OUT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'docs', 'lighthouse');
const RUNS = 3;

function run(url, preset) {
  const file = join(tmpdir(), `lh-${process.pid}.json`);
  const args = [
    '--yes',
    LIGHTHOUSE,
    url,
    '--quiet',
    '--output=json',
    `--output-path=${file}`,
    '--chrome-flags=--headless=new --no-sandbox',
    '--only-categories=performance,accessibility,best-practices,seo',
  ];
  if (preset === 'desktop') args.push('--preset=desktop');
  execFileSync('npx', args, {
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: { ...process.env, CHROME_PATH: chromium.executablePath() },
  });
  const r = JSON.parse(readFileSync(file, 'utf8'));
  rmSync(file, { force: true });
  const score = (k) => Math.round(r.categories[k].score * 100);
  const audit = (k) => r.audits[k].numericValue;
  return {
    performance: score('performance'),
    accessibility: score('accessibility'),
    bestPractices: score('best-practices'),
    seo: score('seo'),
    fcp_s: +(audit('first-contentful-paint') / 1000).toFixed(2),
    lcp_s: +(audit('largest-contentful-paint') / 1000).toFixed(2),
    tbt_ms: Math.round(audit('total-blocking-time')),
    cls: +audit('cumulative-layout-shift').toFixed(3),
    version: r.lighthouseVersion,
  };
}

const median = (rows) => {
  const out = {};
  for (const k of Object.keys(rows[0]).filter((k) => k !== 'version')) {
    const v = rows.map((r) => r[k]).sort((a, b) => a - b);
    out[k] = v[Math.floor(v.length / 2)];
  }
  return out;
};

const results = {};
let version = '';
for (const [label, url] of [
  ['excludingModel', `${BASE}?preload=0`],
  ['withModel', BASE],
]) {
  for (const preset of ['mobile', 'desktop']) {
    const runs = Array.from({ length: RUNS }, () => run(url, preset));
    version = runs[0].version;
    runs.forEach((r) => delete r.version);
    results[`${label}_${preset}`] = { median: median(runs), runs };
    console.log(label, preset, JSON.stringify(results[`${label}_${preset}`].median));
  }
}

mkdirSync(OUT, { recursive: true });
const summary = {
  lighthouse: version,
  date: new Date().toISOString().slice(0, 10),
  browser: 'Chromium (Playwright build), headless',
  server: 'vite preview of the production build',
  note:
    'Median of 3 runs. excludingModel = page loaded with ?preload=0 (the models, a 47 MB classifier and a ' +
    "17 MB input check, are not fetched until the first classification; Lighthouse cannot block a Web Worker's " +
    'requests). withModel = default behaviour, both models download in the background after page load.',
  results,
};
writeFileSync(join(OUT, 'summary.json'), JSON.stringify(summary, null, 1) + '\n');
