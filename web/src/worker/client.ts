import * as Comlink from 'comlink';
import { bitmapToRGBA } from '../lib/pixels';
import type {
  BackendName,
  ClassifierApi,
  ModelInfo,
  ProgressCallback,
  RunsIn,
  WorkerPrediction,
} from './types';

export const MODEL_URL = new URL(`${import.meta.env.BASE_URL}model/model.json`, window.location.origin).href;

/** Optional ?backend=webgl|wasm|cpu override (used by the e2e tests to exercise every backend). */
export function backendFromUrl(search = window.location.search): BackendName | undefined {
  const value = new URLSearchParams(search).get('backend');
  return value === 'webgl' || value === 'wasm' || value === 'cpu' ? value : undefined;
}

/** Can a Web Worker render WebGL? (Needs OffscreenCanvas with a WebGL context: Chrome, Firefox, Safari 17+.) */
export function workerCanUseWebGL(): boolean {
  try {
    if (typeof OffscreenCanvas === 'undefined') return false;
    const canvas = new OffscreenCanvas(1, 1);
    return !!(canvas.getContext('webgl2') ?? canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

function pageCanUseWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2') ?? canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

/**
 * Where to run TensorFlow.js. GPU inference in a worker keeps the page fully responsive. Where workers have
 * no WebGL but the page does (e.g. Safari 16), the GPU on the main thread is far faster than CPU/WASM in a
 * worker, at the cost of short UI pauses while a photo is preprocessed.
 */
export function chooseRunsIn(
  backend: BackendName | undefined,
  worker = workerCanUseWebGL(),
  page = pageCanUseWebGL(),
): RunsIn {
  const wantsWebGL = backend === undefined || backend === 'webgl';
  return wantsWebGL && !worker && page ? 'main-thread' : 'worker';
}

export class ClassifierClient {
  private worker: Worker | null = null;
  private remote: Comlink.Remote<ClassifierApi> | null = null;
  private local: Promise<ClassifierApi> | null = null;
  readonly runsIn: RunsIn;
  private readonly workerHasOffscreenCanvas = typeof OffscreenCanvas !== 'undefined';

  constructor(private readonly backend = backendFromUrl()) {
    this.runsIn = chooseRunsIn(backend);
    if (this.runsIn === 'worker') {
      this.worker = new Worker(new URL('./classifier.worker.ts', import.meta.url), { type: 'module' });
      this.remote = Comlink.wrap<ClassifierApi>(this.worker);
    } else {
      this.local = import('./classifierCore').then((m) => m.createClassifierApi('main-thread'));
    }
  }

  async init(onProgress: ProgressCallback): Promise<ModelInfo> {
    if (this.remote) return this.remote.init(MODEL_URL, Comlink.proxy(onProgress), this.backend);
    return (await this.local!).init(MODEL_URL, onProgress, this.backend ?? 'webgl');
  }

  /** Takes ownership of (and consumes) the bitmap. */
  async classify(bitmap: ImageBitmap): Promise<WorkerPrediction> {
    if (!this.remote) return (await this.local!).classify(bitmap);
    if (this.workerHasOffscreenCanvas) return this.remote.classify(Comlink.transfer(bitmap, [bitmap]));
    const { data, width, height } = bitmapToRGBA(bitmap);
    bitmap.close();
    return this.remote.classifyPixels(Comlink.transfer(data, [data.buffer]), width, height);
  }

  async memory() {
    return this.remote ? this.remote.memory() : (await this.local!).memory();
  }

  terminate() {
    this.remote?.[Comlink.releaseProxy]();
    this.worker?.terminate();
  }
}
