/**
 * All TensorFlow.js work: backend selection (WebGL -> WASM -> CPU), download of both models in parallel with
 * progress (the classifier and the input check's image encoder), IndexedDB caching keyed by each model's
 * content hash, warm-up, preprocessing and inference. Normally runs inside a Web Worker (classifier.worker.ts);
 * browsers without WebGL in workers run it on the main thread instead (see client.ts).
 *
 * Per photo: preprocess once, run the input check, then the classifier only if the photo passed (or the user
 * chose "Analyse anyway").
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
import { checkEmbedding } from '../lib/inputCheck';
import { bitmapToRGBA, rgbToRGBA, toModelInput } from '../lib/pixels';
import { MODEL_SIZE } from '../lib/resize';
import { GATE_INPUT, GATE_OUTPUT, GATE_SIZE, runGate, toGateInput } from './gate';
import { INPUT_NAME, OUTPUT_NAMES, runModel } from './inference';
import type {
  BackendName,
  ClassifierApi,
  ClassifyOptions,
  ModelInfo,
  ProgressCallback,
  RunsIn,
  WorkerPrediction,
  WorkerResult,
} from './types';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPlain,
  'tfjs-backend-wasm-simd.wasm': wasmSimd,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimd,
});

interface CacheSlot {
  prefix: string;
  key: string;
}
const MODEL_CACHE: CacheSlot = {
  prefix: 'indexeddb://saffron-web-',
  key: `indexeddb://saffron-web-${__MODEL_VERSION__}`,
};
const GATE_CACHE: CacheSlot = {
  prefix: 'indexeddb://saffron-gate-',
  key: `indexeddb://saffron-gate-${__GATE_VERSION__}`,
};
const TOTAL_BYTES = __MODEL_BYTES__ + __GATE_BYTES__;
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

async function loadFromCache(slot: CacheSlot): Promise<GraphModel | null> {
  try {
    const cached = await tf.io.listModels();
    if (cached[slot.key]) return await loadGraphModel(slot.key);
  } catch {
    /* IndexedDB unavailable (private browsing, blocked storage): fall back to the network */
  }
  return null;
}

async function saveToCache(m: GraphModel, slot: CacheSlot): Promise<void> {
  try {
    const cached = await tf.io.listModels();
    await Promise.all(
      Object.keys(cached)
        .filter((key) => key.startsWith(slot.prefix) && key !== slot.key)
        .map((key) => tf.io.removeModel(key)),
    );
    await m.save(slot.key);
  } catch (e) {
    console.warn('Could not cache the model in IndexedDB', e);
  }
}

/** Cache first, then the network; reports this model's download fraction. */
async function loadModel(
  url: string,
  slot: CacheSlot,
  onFraction: (f: number) => void,
): Promise<{ model: GraphModel; fromCache: boolean }> {
  const cached = await loadFromCache(slot);
  if (cached) {
    onFraction(1);
    return { model: cached, fromCache: true };
  }
  const model = await loadGraphModel(url, { onProgress: onFraction });
  return { model, fromCache: false };
}

/**
 * First inference compiles every WebGL shader (ResNet50V2 has many), which is slow if done one by one.
 * On WebGL, a compile-only pass first lets the driver compile them in parallel (KHR_parallel_shader_compile);
 * then one real run primes everything else.
 */
async function compileShaders(m: GraphModel, size: number, input: string, outputs: string[]): Promise<void> {
  const webgl = tf.backend() as unknown as {
    checkCompileCompletionAsync(): Promise<boolean[]>;
    getUniformLocations(): void;
  };
  const x = tf.zeros([1, size, size, 3]);
  try {
    tf.env().set('ENGINE_COMPILE_ONLY', true);
    const out = m.execute({ [input]: x }, outputs) as tf.Tensor[];
    tf.env().set('ENGINE_COMPILE_ONLY', false);
    await webgl.checkCompileCompletionAsync();
    webgl.getUniformLocations();
    tf.dispose(out);
  } catch (e) {
    console.warn('Parallel shader compilation unavailable; compiling during warm-up instead', e);
  } finally {
    tf.env().set('ENGINE_COMPILE_ONLY', false);
    x.dispose();
  }
}

async function warmUp(m: GraphModel, gate: GraphModel, backend: BackendName): Promise<void> {
  if (backend === 'webgl') {
    await compileShaders(m, MODEL_SIZE, INPUT_NAME, [...OUTPUT_NAMES]);
    await compileShaders(gate, GATE_SIZE, GATE_INPUT, [GATE_OUTPUT]);
  }
  await runModel(m, new Uint8Array(MODEL_SIZE * MODEL_SIZE * 3));
  await runGate(gate, new Uint8Array(GATE_SIZE * GATE_SIZE * 3));
}

export function createClassifierApi(runsIn: RunsIn): ClassifierApi {
  let model: GraphModel | null = null;
  let gate: GraphModel | null = null;
  let info: ModelInfo | null = null;

  async function predict(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    options: ClassifyOptions,
  ): Promise<WorkerResult> {
    if (!model || !gate || !info) throw new Error('The model is not loaded yet.');
    const t0 = performance.now();
    const rgb = toModelInput(data, width, height);
    const t1 = performance.now();
    const inputCheck = checkEmbedding(await runGate(gate, toGateInput(rgb)));
    const t2 = performance.now();
    if (!inputCheck.passed && !options.force) {
      return { inputCheck, inputCheckMs: t2 - t1, prediction: null };
    }
    const { probabilities, cam } = await runModel(model, rgb);
    const t3 = performance.now();
    const input = rgbToRGBA(rgb);
    const prediction: WorkerPrediction = {
      probabilities: Array.from(probabilities),
      cam,
      input,
      preprocessMs: t1 - t0,
      inferenceMs: t3 - t2,
      backend: info.backend,
    };
    const result: WorkerResult = { inputCheck, inputCheckMs: t2 - t1, prediction };
    return runsIn === 'worker' ? Comlink.transfer(result, [cam.buffer, input.buffer]) : result;
  }

  return {
    async init(urls, onProgress: ProgressCallback, preferredBackend) {
      if (model && gate && info) return info;
      const t0 = performance.now();
      const backend = await selectBackend(preferredBackend);

      // both downloads at once; progress is the byte-weighted share of the two
      const fractions = { model: 0, gate: 0 };
      // If one download fails, the other may still be running: stop reporting its progress, or it would
      // overwrite the error state in the page, and swallow its later rejection.
      let failed = false;
      const report = () => {
        if (failed) return;
        onProgress({
          phase: 'download',
          fraction: (fractions.model * __MODEL_BYTES__ + fractions.gate * __GATE_BYTES__) / TOTAL_BYTES,
        });
      };
      const loads = [
        loadModel(urls.model, MODEL_CACHE, (f) => {
          fractions.model = f;
          report();
        }),
        loadModel(urls.gate, GATE_CACHE, (f) => {
          fractions.gate = f;
          report();
        }),
      ] as const;
      loads.forEach((l) => l.catch(() => undefined));
      let main: Awaited<(typeof loads)[0]>;
      let check: Awaited<(typeof loads)[1]>;
      try {
        [main, check] = await Promise.all(loads);
      } catch (e) {
        failed = true;
        throw e;
      }
      model = main.model;
      gate = check.model;
      const loadMs = performance.now() - t0;

      onProgress({ phase: 'warmup', fraction: 0 });
      const t1 = performance.now();
      await warmUp(model, gate, backend);
      const warmupMs = performance.now() - t1;
      onProgress({ phase: 'warmup', fraction: 1 });

      info = { backend, fromCache: main.fromCache && check.fromCache, loadMs, warmupMs, runsIn };
      // after warm-up, so caching never delays the first prediction
      if (!main.fromCache) void saveToCache(model, MODEL_CACHE);
      if (!check.fromCache) void saveToCache(gate, GATE_CACHE);
      return info;
    },

    async classify(bitmap, options) {
      const { data, width, height } = bitmapToRGBA(bitmap);
      bitmap.close();
      return predict(data, width, height, options);
    },

    async classifyPixels(data, width, height, options) {
      return predict(data, width, height, options);
    },

    async memory() {
      return { numTensors: tf.memory().numTensors, numBytes: tf.memory().numBytes };
    },
  };
}
