import { CLASS_INFO, CLASS_ORDER, type ClassId, type Group } from './classes';

/** Below this top-class probability the result is flagged as low confidence. */
export const MIN_TOP_PROBABILITY = 0.6;
/** Below this gap between the two most likely classes the result is flagged as low confidence. */
export const MIN_TOP2_MARGIN = 0.2;

export interface RankedClass {
  id: ClassId;
  probability: number;
}

export type LowConfidenceReason = 'low-top-probability' | 'small-margin';

export interface Interpretation {
  /** all classes, most likely first */
  ranked: RankedClass[];
  top: RankedClass;
  runnerUp: RankedClass;
  /** P(top) - P(runner-up) */
  margin: number;
  /** summed probability per group: pure = C1 + C2, counterfeit = C3 + C4 + C5 */
  groups: Record<Group, number>;
  /** the group holding more probability mass */
  verdict: Group;
  lowConfidence: boolean;
  reasons: LowConfidenceReason[];
}

export function groupProbabilities(probabilities: ArrayLike<number>): Record<Group, number> {
  const groups: Record<Group, number> = { pure: 0, counterfeit: 0 };
  CLASS_ORDER.forEach((id, i) => {
    groups[CLASS_INFO[id].group] += probabilities[i];
  });
  return groups;
}

export function lowConfidenceReasons(topProbability: number, margin: number): LowConfidenceReason[] {
  const reasons: LowConfidenceReason[] = [];
  if (topProbability < MIN_TOP_PROBABILITY) reasons.push('low-top-probability');
  if (margin < MIN_TOP2_MARGIN) reasons.push('small-margin');
  return reasons;
}

export function interpret(probabilities: ArrayLike<number>): Interpretation {
  if (probabilities.length !== CLASS_ORDER.length) {
    throw new Error(`expected ${CLASS_ORDER.length} probabilities, got ${probabilities.length}`);
  }
  const ranked = CLASS_ORDER.map((id, i) => ({ id, probability: probabilities[i] })).sort(
    (a, b) => b.probability - a.probability || CLASS_ORDER.indexOf(a.id) - CLASS_ORDER.indexOf(b.id),
  );
  const [top, runnerUp] = ranked;
  const margin = top.probability - runnerUp.probability;
  const groups = groupProbabilities(probabilities);
  const reasons = lowConfidenceReasons(top.probability, margin);
  return {
    ranked,
    top,
    runnerUp,
    margin,
    groups,
    verdict: groups.pure >= groups.counterfeit ? 'pure' : 'counterfeit',
    lowConfidence: reasons.length > 0,
    reasons,
  };
}

export const formatPercent = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;
