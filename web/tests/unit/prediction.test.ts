import { describe, expect, it } from 'vitest';
import {
  groupProbabilities,
  interpret,
  lowConfidenceReasons,
  MIN_TOP2_MARGIN,
  MIN_TOP_PROBABILITY,
} from '../../src/lib/prediction';

describe('groupProbabilities', () => {
  it('sums C1 + C2 as pure and C3 + C4 + C5 as counterfeit', () => {
    const g = groupProbabilities([0.1, 0.2, 0.3, 0.25, 0.15]);
    expect(g.pure).toBeCloseTo(0.3);
    expect(g.counterfeit).toBeCloseTo(0.7);
  });

  it('keeps the two groups summing to 1 for a softmax output', () => {
    const g = groupProbabilities([0.03, 0.55, 0.03, 0.01, 0.38]);
    expect(g.pure + g.counterfeit).toBeCloseTo(1, 10);
  });
});

describe('interpret', () => {
  it('ranks classes, finds the verdict and is confident for a clear winner', () => {
    const r = interpret([0.93, 0.03, 0.02, 0.01, 0.01]);
    expect(r.ranked.map((c) => c.id)).toEqual(['C1', 'C2', 'C3', 'C4', 'C5']);
    expect(r.top).toEqual({ id: 'C1', probability: 0.93 });
    expect(r.verdict).toBe('pure');
    expect(r.lowConfidence).toBe(false);
  });

  it('decides the verdict on group mass, not on the top class', () => {
    // top class is pure (C2 = 0.40) but the three counterfeit classes hold 0.55 together
    const r = interpret([0.05, 0.4, 0.2, 0.2, 0.15]);
    expect(r.top.id).toBe('C2');
    expect(r.verdict).toBe('counterfeit');
  });

  it('flags the real hard C2 test image (top 0.551, runner-up 0.383) as low confidence', () => {
    const r = interpret([0.033, 0.551, 0.026, 0.006, 0.383]);
    expect(r.top.id).toBe('C2');
    expect(r.runnerUp.id).toBe('C5');
    expect(r.margin).toBeCloseTo(0.168, 3);
    expect(r.reasons).toEqual(['low-top-probability', 'small-margin']);
  });

  it('rejects vectors of the wrong length', () => {
    expect(() => interpret([0.5, 0.5])).toThrow(/expected 5/);
  });
});

describe('lowConfidenceReasons', () => {
  it('uses strict thresholds of 0.60 (top) and 0.20 (margin)', () => {
    expect(MIN_TOP_PROBABILITY).toBe(0.6);
    expect(MIN_TOP2_MARGIN).toBe(0.2);
    expect(lowConfidenceReasons(0.6, 0.2)).toEqual([]);
    expect(lowConfidenceReasons(0.599, 0.5)).toEqual(['low-top-probability']);
    expect(lowConfidenceReasons(0.7, 0.199)).toEqual(['small-margin']);
  });

  it('for softmax outputs, a margin below 0.20 only happens together with a top below 0.60', () => {
    // top >= 0.6 leaves at most 0.4 for the runner-up, so the margin is then >= 0.2
    const r = interpret([0.0, 0.0, 0.0, 0.41, 0.59]);
    expect(r.reasons).toEqual(['low-top-probability', 'small-margin']);
    expect(interpret([0.0, 0.0, 0.0, 0.39, 0.61]).lowConfidence).toBe(false);
  });
});
