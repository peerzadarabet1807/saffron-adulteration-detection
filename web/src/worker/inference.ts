/**
 * Backend-agnostic model execution, shared by the Web Worker and the Node parity test.
 *
 * Contract of the exported graph (scripts/build_web_model.py):
 *   input  "image"          float32 [N, 300, 300, 3], raw RGB in [0, 255] (normalisation is inside the model)
 *   output "probabilities"  float32 [N, 5] softmax, order = models/class_names.json
 *   output "cam"            float32 [N, 10, 10, 5] class activation maps (pre-ReLU)
 */
import * as tf from '@tensorflow/tfjs-core';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { MODEL_SIZE } from '../lib/resize';

export const INPUT_NAME = 'image';
export const OUTPUT_NAMES = ['probabilities', 'cam'] as const;

export interface RawPrediction {
  probabilities: Float32Array;
  /** flat [10, 10, numClasses] (NHWC without the batch axis) */
  cam: Float32Array;
}

/** Run one 300x300 RGB image (bytes or floats, 0-255) through the model. Frees every tensor it creates. */
export async function runModel(model: GraphModel, rgb: Uint8Array | Float32Array): Promise<RawPrediction> {
  if (rgb.length !== MODEL_SIZE * MODEL_SIZE * 3) {
    throw new Error(`expected ${MODEL_SIZE}x${MODEL_SIZE}x3 pixels, got ${rgb.length} values`);
  }
  const [probT, camT] = tf.tidy(() => {
    const x = tf.tensor4d(Float32Array.from(rgb), [1, MODEL_SIZE, MODEL_SIZE, 3], 'float32');
    return model.execute({ [INPUT_NAME]: x }, [...OUTPUT_NAMES]) as tf.Tensor[];
  });
  try {
    const [probabilities, cam] = await Promise.all([probT.data(), camT.data()]);
    // copies: some backends hand out their internal buffers, and the worker transfers these to the UI
    return { probabilities: Float32Array.from(probabilities), cam: Float32Array.from(cam) };
  } finally {
    tf.dispose([probT, camT]);
  }
}
