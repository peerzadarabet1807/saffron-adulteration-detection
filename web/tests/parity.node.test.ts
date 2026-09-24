/**
 * Keras <-> TF.js parity test (runs in Node on the TF.js WASM backend).
 *
 * Needs local build artefacts that are not in git:
 *   build/reference/<set>/NNN.npy + reference.json   (scripts/make_reference_predictions.py)
 * and a converted model (default web/public/model, override with PARITY_MODEL_DIR).
 *
 * Asserts, for every reference image: identical argmax, max |probability diff| <= 0.02 and
 * Pearson(CAM_tfjs, CAM_keras) >= 0.98 for the predicted class (after ReLU + min-max).
 *
 *   npm run test:parity
 *   PARITY_MODEL_DIR=../build/tfjs/uint8 npm run test:parity
 */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { extractClassMap, pearson, reluMinMax } from '../src/lib/cam';
import { runModel } from '../src/worker/inference';
import { readNpy } from './utils/npy';
import { loadGraphModelFromDir } from './utils/nodeModel';

const ROOT = resolve(__dirname, '..', '..');
const MODEL_DIR = resolve(process.env.PARITY_MODEL_DIR ?? join(ROOT, 'web', 'public', 'model'));
const REF_DIR = join(ROOT, 'build', 'reference');
const SETS = ['test', 'samples'].filter((s) => existsSync(join(REF_DIR, s, 'reference.json')));
const CLASSES = JSON.parse(readFileSync(join(ROOT, 'models', 'class_names.json'), 'utf8')) as string[];

const MAX_PROB_DIFF = 0.02;
const MIN_CAM_CORR = 0.98;

interface ReferenceRow {
  id: string;
  npy: string;
  trueClass: string | null;
  predicted: string;
  probabilities: number[];
  cam: number[];
}

let model: GraphModel;

beforeAll(async () => {
  setWasmPaths(join(ROOT, 'web', 'node_modules', '@tensorflow', 'tfjs-backend-wasm', 'dist') + '/');
  await tf.setBackend('wasm');
  await tf.ready();
  model = await loadGraphModelFromDir(MODEL_DIR);
}, 120_000);

afterAll(() => model?.dispose());

describe.skipIf(SETS.length === 0)('TF.js model matches Keras', () => {
  for (const set of SETS) {
    it(`${set} set`, async () => {
      const rows = JSON.parse(readFileSync(join(REF_DIR, set, 'reference.json'), 'utf8')) as ReferenceRow[];
      const before = tf.memory().numTensors;
      let agree = 0;
      let maxDiff = 0;
      let minCorr = 1;
      let correct = 0;
      const perImage = [];

      for (const row of rows) {
        await new Promise((r) => setImmediate(r)); // keep the vitest worker RPC responsive
        const { data } = readNpy(readFileSync(join(REF_DIR, set, row.npy)));
        const { probabilities, cam } = await runModel(model, data as Uint8Array);
        const pred = probabilities.indexOf(Math.max(...probabilities));
        const diff = Math.max(...row.probabilities.map((p, i) => Math.abs(p - probabilities[i])));
        const corr = pearson(reluMinMax(extractClassMap(cam, pred, CLASSES.length)), row.cam);
        agree += CLASSES[pred] === row.predicted ? 1 : 0;
        correct += row.trueClass && CLASSES[pred] === row.trueClass ? 1 : 0;
        maxDiff = Math.max(maxDiff, diff);
        minCorr = Math.min(minCorr, corr);
        perImage.push({
          id: row.id,
          keras: row.predicted,
          tfjs: CLASSES[pred],
          maxProbDiff: diff,
          camCorr: corr,
        });
      }

      const summary = {
        set,
        modelDir: MODEL_DIR,
        backend: tf.getBackend(),
        images: rows.length,
        argmaxAgreement: agree / rows.length,
        accuracy: correct / rows.length,
        maxProbDiff: maxDiff,
        minCamCorr: minCorr,
        leakedTensors: tf.memory().numTensors - before,
      };
      console.log(JSON.stringify(summary));
      mkdirSync(join(ROOT, 'build', 'parity'), { recursive: true });
      const tag = MODEL_DIR.split(/[\\/]/).slice(-2).join('_');
      writeFileSync(
        join(ROOT, 'build', 'parity', `${tag}_${set}.json`),
        JSON.stringify({ summary, perImage }, null, 1),
      );

      expect(summary.argmaxAgreement).toBe(1);
      expect(maxDiff).toBeLessThanOrEqual(MAX_PROB_DIFF);
      expect(minCorr).toBeGreaterThanOrEqual(MIN_CAM_CORR);
      expect(summary.leakedTensors).toBe(0);
    }, 600_000);
  }
});
