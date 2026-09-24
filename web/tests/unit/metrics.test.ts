import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { CLASS_ORDER } from '../../src/lib/classes';
import { correctCount, metrics, parity, testErrors } from '../../src/lib/metrics';

const root = join(__dirname, '..', '..', '..');

describe('metrics shown in the app', () => {
  it('are an exact copy of results/metrics.json from the training run (run `npm run sync:results`)', () => {
    const source = JSON.parse(readFileSync(join(root, 'results', 'metrics.json'), 'utf8'));
    expect(metrics).toEqual(source);
  });

  it('use the same class order as models/class_names.json', () => {
    expect(CLASS_ORDER).toEqual(JSON.parse(readFileSync(join(root, 'models', 'class_names.json'), 'utf8')));
  });

  it('derive the published test errors from the confusion matrix', () => {
    expect(correctCount()).toBe(Math.round(metrics.test.accuracy * metrics.test.n_images));
    expect(testErrors()).toEqual([{ trueClass: 'C4', predicted: 'C5', count: 2 }]);
  });

  it('report that the shipped TF.js variant passed the parity test', () => {
    expect(parity.variants[parity.shipped_variant].passes).toBe(true);
  });
});
