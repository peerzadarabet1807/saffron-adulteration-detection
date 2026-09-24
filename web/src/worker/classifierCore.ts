/**
 * All TensorFlow.js work: backend selection (WebGL -> WASM -> CPU), model download with progress, IndexedDB
 * caching, warm-up, preprocessing and inference. Normally runs inside a Web Worker (classifier.worker.ts);
 * browsers without WebGL in workers run it on the main thread instead (see client.ts).
 */
import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import { loadGraphModel, type GraphModel } from '@tensorflow/tfjs-converter';
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreadedSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';
import { bitmapToRGBA, rgbToRGBA, toModelInput } from '../lib/pixels';
import { MODEL_SIZE } from '../lib/resize';
import { INPUT_NAME, OUTPUT_NAMES, runModel } from './inference';
import type {
  BackendName,
  ClassifierApi,
  ModelInfo,
  ProgressCallback,
  RunsIn,
  WorkerPrediction,
} from './types';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPlain,
  'tfjs-backend-wasm-simd.wasm': wasmSimd,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimd,
});

const CACHE_PREFIX = 'indexeddb://saffron-web-';
const CACHE_KEY = `${CACHE_PREFIX}${__MODEL_VERSION__}`;
const BACKENDS: BackendName[] = ['webgl', 'wasm', 'cpu'];

async function selectBackend(preferred?: BackendName): Promise<BackendName> {
  const order = preferred ? [preferred, ...BACKENDS.filter((b) => b !== preferred)] : BACKENDS;
  for (const name of order) {
    try {
      if (await tf.setBackend(name)) {
        await tf.ready();
        return name;
      }
    } catch {
      /* try the next backend */
    }
  }
  throw new Error('No TensorFlow.js backend is available in this browser.');
}

async function loadFromCache(): Promise<GraphModel | null> {
  try {
    const cached = await tf.io.listModels();
    if (cached[CACHE_KEY]) return await loadGraphModel(CACHE_KEY);
  } catch {
    /* IndexedDB unavailable (private browsing, blocked storage): fall back to the network */
  }
  return null;
}

async function saveToCache(m: GraphModel): Promise<void> {
  try {
    const cached = await tf.io.listModels();
    await Promise.all(
      Object.keys(cached)
        .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_KEY)
        .map((key) => tf.io.removeModel(key)),
    );
    await m.save(CACHE_KEY);
  } catch (e) {
    console.warn('Could not cache the model in IndexedDB', e);
  }
}

/**
 * First inference compiles every WebGL shader (ResNet50V2 has many), which is slow if done one by one.
 * On WebGL, a compile-only pass first lets the driver compile them in parallel (KHR_parallel_shader_compile);
 * then one real run primes everything else.
 */
async function warmUp(m: GraphModel, backend: BackendName): Promise<void> {
  if (backend === 'webgl') {
    const webgl = tf.backend() as unknown as {
      checkCompileCompletionAsync(): Promise<boolean[]>;
      getUniformLocations(): void;
    };
    const x = tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]);
    try {
      tf.env().set('ENGINE_COMPILE_ONLY', true);
      const outputs = m.execute({ [INPUT_NAME]: x }, [...OUTPUT_NAMES]) as tf.Tensor[];
      tf.env().set('ENGINE_COMPILE_ONLY', false);
      await webgl.checkCompileCompletionAsync();
      webgl.getUniformLocations();
      tf.dispose(outputs);
    } catch (e) {
      console.warn('Parallel shader compilation unavailable; compiling during warm-up instead', e);
    } finally {
      tf.env().set('ENGINE_COMPILE_ONLY', false);
      x.dispose();
    }
  }
  await runModel(m, new Uint8Array(MODEL_SIZE * MODEL_SIZE * 3));
}

export function createClassifierApi(runsIn: RunsIn): ClassifierApi {
  let model: GraphModel | null = null;
  let info: ModelInfo | null = null;

  async function predict(data: Uint8ClampedArray, width: number, height: number): Promise<WorkerPrediction> {
    if (!model || !info) throw new Error('The model is not loaded yet.');
    const t0 = performance.now();
    const rgb = toModelInput(data, width, height);
    const t1 = performance.now();
    const { probabilities, cam } = await runModel(model, rgb);
    const t2 = performance.now();
    const input = rgbToRGBA(rgb);
    const result: WorkerPrediction = {
      probabilities: Array.from(probabilities),
      cam,
      input,
      preprocessMs: t1 - t0,
      inferenceMs: t2 - t1,
      backend: info.backend,
    };
    return runsIn === 'worker' ? Comlink.transfer(result, [cam.buffer, input.buffer]) : result;
  }

  return {
    async init(modelUrl, onProgress: ProgressCallback, preferredBackend) {
      if (model && info) return info;
      const t0 = performance.now();
      const backend = await selectBackend(preferredBackend);

      let fromCache = false;
      model = await loadFromCache();
      if (model) {
        fromCache = true;
        onProgress({ phase: 'download', fraction: 1 });
      } else {
        model = await loadGraphModel(modelUrl, {
          onProgress: (fraction) => onProgress({ phase: 'download', fraction }),
        });
      }
      const loadMs = performance.now() - t0;

      onProgress({ phase: 'warmup', fraction: 0 });
      const t1 = performance.now();
      await warmUp(model, backend);
      const warmupMs = performance.now() - t1;
      onProgress({ phase: 'warmup', fraction: 1 });

      info = { backend, fromCache, loadMs, warmupMs, runsIn };
      // after warm-up, so caching never delays the first prediction
      if (!fromCache) void saveToCache(model);
      return info;
    },

    async classify(bitmap) {
      const { data, width, height } = bitmapToRGBA(bitmap);
      bitmap.close();
      return predict(data, width, height);
    },

    async classifyPixels(data, width, height) {
      return predict(data, width, height);
    },

    async memory() {
      return { numTensors: tf.memory().numTensors, numBytes: tf.memory().numBytes };
    },
  };
}
