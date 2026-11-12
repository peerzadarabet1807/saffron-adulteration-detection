/**
 * The input check's image encoder, shared by the Web Worker and the Node parity test.
 *
 * Contract of the exported graph (the poultry repository's scripts/build_gate_model.py, TinyCLIP ViT-8M/16
 * vision tower; the same weights are committed in web/public/gate/):
 *   input  "image"      float32 [N, 224, 224, 3], RGB 0..255 (normalised inside the graph)
 *   output "embedding"  float32 [N, 512], unit length
 *
 * The gate sees the classifier's 300 x 300 input area-resized to 224 x 224, exactly as
 * scripts/gate_extract.py does with cv2.INTER_AREA.
 */
import * as tf from '@tensorflow/tfjs-core';
import type { GraphModel } from '@tensorflow/tfjs-converter';
import { areaResizeToRGB, MODEL_SIZE } from '../lib/resize';

export const GATE_SIZE = 224;
export const GATE_INPUT = 'image';
export const GATE_OUTPUT = 'embedding';

/** The classifier's 300 x 300 RGB bytes -> the gate's 224 x 224 RGB bytes. */
export function toGateInput(modelRgb: Uint8Array): Uint8Array {
  if (modelRgb.length !== MODEL_SIZE * MODEL_SIZE * 3) {
    throw new Error(`expected ${MODEL_SIZE}x${MODEL_SIZE}x3 bytes, got ${modelRgb.length}`);
  }
  return areaResizeToRGB(modelRgb, MODEL_SIZE, MODEL_SIZE, GATE_SIZE, GATE_SIZE, 3);
}

/** Embed one 224 x 224 RGB image (bytes). Frees every tensor it creates. */
export async function runGate(model: GraphModel, gateRgb: Uint8Array): Promise<Float32Array> {
  if (gateRgb.length !== GATE_SIZE * GATE_SIZE * 3) {
    throw new Error(`expected ${GATE_SIZE}x${GATE_SIZE}x3 bytes, got ${gateRgb.length}`);
  }
  const out = tf.tidy(() => {
    const x = tf.tensor4d(Float32Array.from(gateRgb), [1, GATE_SIZE, GATE_SIZE, 3], 'float32');
    return model.execute({ [GATE_INPUT]: x }, GATE_OUTPUT) as tf.Tensor;
  });
  try {
    return Float32Array.from(await out.data());
  } finally {
    out.dispose();
  }
}
