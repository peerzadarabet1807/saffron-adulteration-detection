import { useCallback, useEffect, useRef, useState } from 'react';
import { decodeImage, DecodeError, makeDisplayUrl } from '../lib/decode';
import { validateFile } from '../lib/fileValidation';
import { interpret, type Interpretation } from '../lib/prediction';
import type { SampleInfo } from '../lib/samples';
import { ClassifierClient } from '../worker/client';
import type { InputCheckResult, LoadProgress, ModelInfo, WorkerPrediction } from '../worker/types';

export type ModelState =
  | { status: 'loading'; phase: LoadProgress['phase']; fraction: number }
  | { status: 'ready'; info: ModelInfo }
  | { status: 'error'; message: string };

export type SourceKind = 'upload' | 'camera' | 'drop' | 'paste' | 'sample';

export interface ImageSource {
  kind: SourceKind;
  name: string;
  sample?: SampleInfo;
}

export interface ClassificationResult {
  id: number;
  source: ImageSource;
  /** oriented, downscaled JPEG object URL of the decoded photo */
  displayUrl: string;
  width: number;
  height: number;
  prediction: WorkerPrediction;
  interpretation: Interpretation;
  createdAt: Date;
  inputCheck: InputCheckResult;
  /** the input check failed and the user chose "Analyse anyway" */
  unchecked: boolean;
}

/** The input check failed: no verdict is shown unless the user chooses "Analyse anyway". */
export interface RejectedPhoto {
  id: number;
  source: ImageSource;
  displayUrl: string;
  width: number;
  height: number;
  inputCheck: InputCheckResult;
  /** kept so "Analyse anyway" can run the classifier on the same file */
  file: Blob;
}

export type JobStep = 'reading' | 'decoding' | 'waiting-for-model' | 'checking' | 'classifying';

export type JobState =
  | { status: 'idle' }
  | { status: 'working'; step: JobStep; source: ImageSource }
  | { status: 'rejected'; photo: RejectedPhoto }
  | { status: 'done'; result: ClassificationResult }
  | { status: 'error'; message: string; source?: ImageSource };

class UserFacingError extends Error {}

// One worker for the whole app (survives React StrictMode's double effects).
let client: ClassifierClient | null = null;
let initPromise: Promise<ModelInfo> | null = null;
let readyInfo: ModelInfo | null = null;
const progressListeners = new Set<(p: LoadProgress) => void>();

function getClient(): ClassifierClient {
  client ??= new ClassifierClient();
  return client;
}

// Read-only diagnostics for the e2e tests: tensor count inside the worker (memory leaks) and the last input
// check result.
let lastInputCheck: InputCheckResult | null = null;
(globalThis as { __saffronDebug?: unknown }).__saffronDebug = {
  memory: () => getClient().memory(),
  lastInputCheck: () => lastInputCheck,
};

function initModel(): Promise<ModelInfo> {
  initPromise ??= getClient()
    .init((p) => progressListeners.forEach((listener) => listener(p)))
    .then((info) => (readyInfo = info));
  return initPromise;
}

function retryInit(): Promise<ModelInfo> {
  client?.terminate();
  client = null;
  initPromise = null;
  return initModel();
}

function describeError(e: unknown): string {
  if (e instanceof UserFacingError || e instanceof DecodeError) return e.message;
  const detail = e instanceof Error ? e.message : String(e);
  return `Something went wrong while analysing the photo (${detail}). Please try again or use another image.`;
}

export function useClassifier() {
  const [model, setModel] = useState<ModelState>(() =>
    readyInfo ? { status: 'ready', info: readyInfo } : { status: 'loading', phase: 'download', fraction: 0 },
  );
  const [job, setJob] = useState<JobState>({ status: 'idle' });
  const jobId = useRef(0);
  const shownUrl = useRef<string | null>(null);

  const watchModel = useCallback((promise: Promise<ModelInfo>) => {
    promise.then(
      (info) => setModel({ status: 'ready', info }),
      (e: unknown) =>
        setModel({
          status: 'error',
          message: `The model could not be loaded: ${e instanceof Error ? e.message : String(e)}`,
        }),
    );
  }, []);

  useEffect(() => {
    // never let a late progress report replace an error (the user needs its Retry button)
    const onProgress = (p: LoadProgress) =>
      setModel((m) =>
        m.status === 'error' ? m : { status: 'loading', phase: p.phase, fraction: p.fraction },
      );
    progressListeners.add(onProgress);
    // The models (47 MB classifier, 17 MB input check) download once the page itself has loaded and painted,
    // so fonts, styles and the app are not competing with them. Classifying earlier starts the download
    // straight away (see classify).
    // ?preload=0 skips the background download until the first classification (used to measure page
    // performance without the model, since Lighthouse cannot block a worker's requests).
    let cancelled = new URLSearchParams(window.location.search).get('preload') === '0';
    const start = () => {
      if (!cancelled) watchModel(initModel());
    };
    const whenIdle = () => {
      if ('requestIdleCallback' in window) window.requestIdleCallback(start, { timeout: 1500 });
      else setTimeout(start, 200);
    };
    if (document.readyState === 'complete') whenIdle();
    else window.addEventListener('load', whenIdle, { once: true });
    return () => {
      cancelled = true;
      window.removeEventListener('load', whenIdle);
      progressListeners.delete(onProgress);
    };
  }, [watchModel]);

  useEffect(
    () => () => {
      if (shownUrl.current) URL.revokeObjectURL(shownUrl.current);
    },
    [],
  );

  const retryModel = useCallback(() => {
    setModel({ status: 'loading', phase: 'download', fraction: 0 });
    watchModel(retryInit());
  }, [watchModel]);

  /**
   * Validate, decode and classify one image. `input` may be a lazy loader (e.g. fetching a sample) so that
   * its failures surface as job errors too. Resolves with the result, or null if it failed or was superseded.
   */
  const classify = useCallback(
    async (input: Blob | (() => Promise<Blob>), source: ImageSource, force = false) => {
      const id = ++jobId.current;
      const isStale = () => id !== jobId.current;
      let displayUrl: string | null = null;
      setJob({ status: 'working', step: 'reading', source });
      try {
        const file = typeof input === 'function' ? await input() : input;
        const validation = await validateFile(file);
        if (!validation.ok) throw new UserFacingError(validation.message);
        if (isStale()) return null;

        setJob({ status: 'working', step: 'decoding', source });
        const bitmap = await decodeImage(file, validation.kind);
        const { width, height } = bitmap;
        displayUrl = await makeDisplayUrl(bitmap);
        if (isStale()) {
          bitmap.close();
          return null;
        }

        if (!readyInfo) setJob({ status: 'working', step: 'waiting-for-model', source });
        const modelReady = initModel();
        watchModel(modelReady);
        await modelReady;
        if (isStale()) {
          bitmap.close();
          return null;
        }
        setJob({ status: 'working', step: force ? 'classifying' : 'checking', source });
        const { inputCheck, prediction } = await getClient().classify(bitmap, { force });
        lastInputCheck = inputCheck;
        if (isStale()) return null;

        if (shownUrl.current) URL.revokeObjectURL(shownUrl.current);
        shownUrl.current = displayUrl;
        displayUrl = null; // now owned by the result or the notice
        if (!prediction) {
          setJob({
            status: 'rejected',
            photo: { id, source, displayUrl: shownUrl.current, width, height, inputCheck, file },
          });
          return null;
        }
        const result: ClassificationResult = {
          id,
          source,
          displayUrl: shownUrl.current,
          width,
          height,
          prediction,
          interpretation: interpret(prediction.probabilities),
          createdAt: new Date(),
          inputCheck,
          unchecked: !inputCheck.passed,
        };
        setJob({ status: 'done', result });
        return result;
      } catch (e) {
        if (!isStale()) setJob({ status: 'error', message: describeError(e), source });
        return null;
      } finally {
        if (displayUrl) URL.revokeObjectURL(displayUrl);
      }
    },
    [watchModel],
  );

  const reset = useCallback(() => {
    jobId.current++;
    setJob({ status: 'idle' });
  }, []);

  /** Run the classifier on the photo the input check turned away. */
  const analyseAnyway = useCallback(
    (photo: RejectedPhoto) => classify(photo.file, photo.source, true),
    [classify],
  );

  return { model, job, classify, reset, retryModel, analyseAnyway };
}
