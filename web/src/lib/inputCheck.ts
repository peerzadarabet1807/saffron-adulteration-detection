/**
 * The input check's decision rule:
 *
 *   score  = max over the five classes of cosine(image embedding, class prototype)
 *   passed = score >= threshold
 *
 * Prototypes and threshold come from models/input_check.json (scripts/build_input_check.py, synced to
 * src/data/). Runs in the worker next to the gate model; the UI only needs the copy in inputCheckCopy.ts.
 */
import config from '../data/input_check.json';

export interface InputCheckResult {
  /** cosine similarity with the nearest class prototype, -1..1 */
  score: number;
  threshold: number;
  passed: boolean;
  version: string;
}

export const INPUT_CHECK_THRESHOLD: number = config.threshold;
export const INPUT_CHECK_VERSION: string = config.version;
export const EMBEDDING_DIM: number = config.model.embedding_dim;

const PROTOTYPES: Float64Array[] = config.prototypes.vectors.map((v) => Float64Array.from(v));

if (PROTOTYPES.some((p) => p.length !== EMBEDDING_DIM)) {
  throw new Error('input_check.json: prototype length does not match embedding_dim');
}

/** Cosine similarity of one image embedding with the nearest class prototype. */
export function inputCheckScore(embedding: ArrayLike<number>): number {
  if (embedding.length !== EMBEDDING_DIM) {
    throw new Error(`expected a ${EMBEDDING_DIM}-d embedding, got ${embedding.length}`);
  }
  let norm = 0;
  for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  if (!Number.isFinite(norm) || norm === 0) throw new Error('embedding has no direction');
  let best = -Infinity;
  for (const p of PROTOTYPES) {
    let dot = 0;
    for (let i = 0; i < p.length; i++) dot += p[i] * embedding[i];
    best = Math.max(best, dot / norm);
  }
  return best;
}

export function checkEmbedding(embedding: ArrayLike<number>): InputCheckResult {
  const score = inputCheckScore(embedding);
  return {
    score,
    threshold: INPUT_CHECK_THRESHOLD,
    passed: score >= INPUT_CHECK_THRESHOLD,
    version: INPUT_CHECK_VERSION,
  };
}
