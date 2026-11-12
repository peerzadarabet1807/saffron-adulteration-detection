/**
 * Input check parity: TF.js (Node, WASM backend) through the TypeScript preprocessing vs the original PyTorch
 * model (scripts/make_gate_references.py). Runs on committed files only: the 10 samples, the EXIF fixture and
 * the 7 original OOD photos.
 *
 * For every fixture: decode with sharp (EXIF orientation applied, as PIL.ImageOps.exif_transpose does), area
 * resize to 300 x 300 and then to 224 x 224 with web/src/lib/resize.ts, embed with the TF.js gate and apply the
 * rule in web/src/lib/inputCheck.ts. Reported: whether the 224 x 224 image equals Python's (SHA-256), cosine
 * with the PyTorch embedding, and the decision. Required: cosine >= 0.999 on every fixture whose image equals
 * Python's (that isolates the conversion), the same pass/fail decision on every fixture, no leaked tensors.
 * Results: ../results/input_check/parity_<variant>.json.
 *
 *   npm run test:parity                                     (the shipped gate, web/public/gate)
 *   PARITY_VARIANT=float16 npm run test:parity               (same, report written to results/)
 *   PARITY_GATE_DIR=<dir with model.json> npm run test:parity
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import sharp from 'sharp';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { checkEmbedding } from '../src/lib/inputCheck';
import { areaResizeToRGB, MODEL_SIZE } from '../src/lib/resize';
import { runGate, toGateInput } from '../src/worker/gate';
import { loadGraphModelFromDir } from './utils/nodeModel';

const ROOT = resolve(__dirname, '..', '..');
const GATE_DIR = resolve(process.env.PARITY_GATE_DIR ?? join(ROOT, 'web', 'public', 'gate'));
const VARIANT = process.env.PARITY_VARIANT ?? (process.env.PARITY_GATE_DIR ? basename(GATE_DIR) : 'shipped');
const MIN_COSINE = 0.999;

interface Reference {
  id: string;
  file: string;
  kind: 'saffron' | 'ood';
  gateSha256: string;
  score: number;
  passed: boolean;
  embedding: number[];
}

const ref = JSON.parse(
  readFileSync(join(ROOT, 'web', 'tests', 'fixtures', 'input_check', 'reference.json'), 'utf8'),
) as { model: string; version: string; images: Reference[] };

let gate: GraphModel;

beforeAll(async () => {
  setWasmPaths(join(ROOT, 'web', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/');
  await tf.setBackend('wasm');
  await tf.ready();
  gate = await loadGraphModelFromDir(GATE_DIR);
}, 120_000);

afterAll(() => gate?.dispose());

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

describe(`input check TF.js (${VARIANT}) matches the PyTorch reference`, () => {
  it('all fixtures', async () => {
    const before = tf.memory().numTensors;
    const rows = [];
    for (const r of ref.images) {
      await new Promise((tick) => setImmediate(tick)); // keep the vitest worker RPC responsive
      const { data, info } = await sharp(readFileSync(join(ROOT, r.file)))
        .rotate()
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const modelRgb = areaResizeToRGB(data, info.width, info.height, MODEL_SIZE, MODEL_SIZE, info.channels);
      const gateRgb = toGateInput(modelRgb);
      const identical = createHash('sha256').update(gateRgb).digest('hex') === r.gateSha256;
      const embedding = await runGate(gate, gateRgb);
      const check = checkEmbedding(embedding);
      rows.push({
        id: r.id,
        kind: r.kind,
        imageIdenticalToPython: identical,
        cosine: cosine(embedding, r.embedding),
        pythonScore: r.score,
        tfjsScore: check.score,
        scoreDiff: Math.abs(check.score - r.score),
        pythonPassed: r.passed,
        tfjsPassed: check.passed,
      });
    }
    const same = rows.filter((r) => r.imageIdenticalToPython);
    const summary = {
      description:
        'Input check image encoder as a TF.js graph model in Node (WASM backend), fed by sharp decode and the ' +
        'TypeScript area resizes, vs the original PyTorch model on the Python 224 x 224 image. Engineering ' +
        'check, not a model evaluation.',
      variant: VARIANT,
      model: ref.model,
      version: ref.version,
      backend: tf.getBackend(),
      images: rows.length,
      imageIdenticalToPython: same.length,
      minCosineIdenticalImages: Math.min(...same.map((r) => r.cosine)),
      minCosineAll: Math.min(...rows.map((r) => r.cosine)),
      maxScoreDiff: Math.max(...rows.map((r) => r.scoreDiff)),
      sameDecision: rows.filter((r) => r.pythonPassed === r.tfjsPassed).length,
      requiredMinCosine: MIN_COSINE,
      leakedTensors: tf.memory().numTensors - before,
    };
    console.log(JSON.stringify(summary, null, 1));
    const dir = join(ROOT, VARIANT === 'shipped' ? 'build' : 'results', 'input_check');
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, `parity_${VARIANT}.json`),
      JSON.stringify({ summary, perImage: rows }, null, 2) + '\n',
    );

    expect(same.length).toBeGreaterThan(0);
    expect(summary.minCosineIdenticalImages).toBeGreaterThanOrEqual(MIN_COSINE);
    expect(summary.sameDecision).toBe(rows.length);
    expect(summary.leakedTensors).toBe(0);
  }, 600_000);
});
