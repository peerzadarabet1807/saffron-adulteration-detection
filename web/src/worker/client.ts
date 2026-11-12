import * as Comlink from 'comlink';
import { bitmapToRGBA } from '../lib/pixels';
import { validateInputCheck } from '../lib/inputCheckValidate';
import type {
  BackendName,
  ClassifierApi,
  ClassifyOptions,
  ModelInfo,
  ProgressCallback,
  RunsIn,
  WorkerResult,
} from './types';

/** Boundary check: whatever the worker returns about the input check is checked before the UI sees it. */
function validateResult(r: WorkerResult): WorkerResult {
  validateInputCheck(r.inputCheck);
  if (!(r.inputCheckMs >= 0)) throw new Error('The input check returned an invalid timing.');
  if (r.prediction === null && r.inputCheck.passed) {
    throw new Error('The model returned no result for a photo that passed the input check.');
  }
  return r;
}

export const MODEL_URL = new URL(`${import.meta.env.BASE_URL}model/model.json`, window.location.origin).href;
export const GATE_URL = new URL(`${import.meta.env.BASE_URL}gate/model.json`, window.location.origin).href;
const URLS = { model: MODEL_URL, gate: GATE_URL };

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
    if (this.remote) return this.remote.init(URLS, Comlink.proxy(onProgress), this.backend);
    return (await this.local!).init(URLS, onProgress, this.backend ?? 'webgl');
  }

  /** Takes ownership of (and consumes) the bitmap. */
  async classify(bitmap: ImageBitmap, options: ClassifyOptions): Promise<WorkerResult> {
    return validateResult(await this.run(bitmap, options));
  }

  private async run(bitmap: ImageBitmap, options: ClassifyOptions): Promise<WorkerResult> {
    if (!this.remote) return (await this.local!).classify(bitmap, options);
    if (this.workerHasOffscreenCanvas) {
      return this.remote.classify(Comlink.transfer(bitmap, [bitmap]), options);
    }
    const { data, width, height } = bitmapToRGBA(bitmap);
    bitmap.close();
    return this.remote.classifyPixels(Comlink.transfer(data, [data.buffer]), width, height, options);
  }

  async memory() {
    return this.remote ? this.remote.memory() : (await this.local!).memory();
  }

  terminate() {
    this.remote?.[Comlink.releaseProxy]();
    this.worker?.terminate();
  }
}
