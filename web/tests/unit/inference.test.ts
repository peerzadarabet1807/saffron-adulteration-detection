/**
 * runModel must free every tensor it creates. Uses the CPU backend and a stand-in graph with the same
 * signature as the real model (image -> probabilities + cam), so it runs in CI without the 47 MB weights.
 */
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { beforeAll, describe, expect, it } from 'vitest';
import { runModel } from '../../src/worker/inference';

const fakeModel = {
  execute(inputs: Record<string, tf.Tensor>, outputs: string[]) {
    const x = inputs.image as tf.Tensor4D;
    expect(x.shape).toEqual([1, 300, 300, 3]);
    expect(x.dtype).toBe('float32');
    const pooled = tf.mean(x, [1, 2]) as tf.Tensor2D; // [1, 3]
    const weights = tf.mul(tf.ones([3, 5]), tf.reshape(tf.range(1, 6), [1, 5])) as tf.Tensor2D;
    const byName: Record<string, tf.Tensor> = {
      probabilities: tf.softmax(tf.div(tf.matMul(pooled, weights), 255)),
      cam: tf.tile(tf.mean(tf.avgPool(x, 30, 30, 'valid'), 3, true), [1, 1, 1, 5]), // [1, 10, 10, 5]
    };
    return outputs.map((name) => byName[name]);
  },
} as unknown as GraphModel;

beforeAll(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
});

describe('runModel', () => {
  it('returns probabilities and a 10 x 10 x 5 CAM as plain typed arrays', async () => {
    const rgb = Uint8Array.from({ length: 300 * 300 * 3 }, (_, i) => i % 256);
    const { probabilities, cam } = await runModel(fakeModel, rgb);
    expect(probabilities).toBeInstanceOf(Float32Array);
    expect(probabilities).toHaveLength(5);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    expect(cam).toHaveLength(10 * 10 * 5);
  });

  it('does not leak tensors across repeated predictions', async () => {
    const rgb = new Uint8Array(300 * 300 * 3).fill(200);
    await runModel(fakeModel, rgb); // allow any one-off backend allocations
    const before = tf.memory().numTensors;
    for (let i = 0; i < 5; i++) await runModel(fakeModel, rgb);
    expect(tf.memory().numTensors).toBe(before);
  });

  it('rejects inputs of the wrong size', async () => {
    await expect(runModel(fakeModel, new Uint8Array(10))).rejects.toThrow(/300x300x3/);
  });
});
