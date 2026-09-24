/** Typed access to the training run's metrics (web/src/data/*.json are copies of results/ and models/). */
import metricsJson from '../data/metrics.json';
import parityJson from '../data/tfjs_parity.json';
import { CLASS_ORDER, type ClassId } from './classes';

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
  roc_auc: number;
  average_precision: number;
}

export interface BackboneResult {
  backbone: string;
  val_macro_f1: number;
  val_accuracy: number;
  val_loss: number;
  params_M: number;
  train_min: number;
}

export interface Metrics {
  best_model: string;
  test: {
    n_images: number;
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    macro_roc_auc_ovr: number;
    per_class: Record<ClassId, PerClassMetrics>;
    confusion_matrix: number[][];
  };
  validation: { n_images: number; accuracy: number; macro_f1: number; loss: number };
  model_comparison: BackboneResult[];
  thesis_baseline: { val_accuracy: number; per_class_f1: Record<ClassId, number> };
  data: {
    n_valid_images: number;
    n_rejected: number;
    split_counts: Record<'train' | 'val' | 'test', Record<ClassId, number>>;
  };
  hyperparameters: { learning_rate: number; batch_size: number; dropout: number; img_size: number };
  environment: Record<string, string>;
}

export interface ParitySet {
  images: number;
  argmaxAgreement: number;
  accuracy: number;
  maxProbDiff: number;
  minCamCorr: number;
}

export interface Parity {
  shipped_variant: string;
  variants: Record<string, { test?: ParitySet; samples?: ParitySet; size_mb: number; passes: boolean }>;
}

export const metrics = metricsJson as unknown as Metrics;
export const parity = parityJson as unknown as Parity;

/** Test-set errors as (true, predicted, count) from the confusion matrix. */
export function testErrors(
  m: Metrics = metrics,
): { trueClass: ClassId; predicted: ClassId; count: number }[] {
  const out: { trueClass: ClassId; predicted: ClassId; count: number }[] = [];
  m.test.confusion_matrix.forEach((row, t) =>
    row.forEach((count, p) => {
      if (t !== p && count > 0) out.push({ trueClass: CLASS_ORDER[t], predicted: CLASS_ORDER[p], count });
    }),
  );
  return out;
}

export const correctCount = (m: Metrics = metrics) =>
  m.test.confusion_matrix.reduce((sum, row, i) => sum + row[i], 0);
