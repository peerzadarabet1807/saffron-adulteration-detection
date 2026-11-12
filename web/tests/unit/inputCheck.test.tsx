import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { InputCheckNotice } from '../../src/components/InputCheckNotice';
import config from '../../src/data/input_check.json';
import summary from '../../src/data/input_check_summary.json';
import type { RejectedPhoto } from '../../src/hooks/useClassifier';
import {
  checkEmbedding,
  EMBEDDING_DIM,
  INPUT_CHECK_THRESHOLD,
  INPUT_CHECK_VERSION,
  inputCheckScore,
} from '../../src/lib/inputCheck';
import { validateInputCheck } from '../../src/lib/inputCheckValidate';
import { INPUT_CHECK_COPY, rejectionAnnouncement, safeguardNote } from '../../src/lib/inputCheckCopy';
import { GATE_SIZE, toGateInput } from '../../src/worker/gate';

const protos = config.prototypes.vectors;

/** A vector orthogonal to every prototype (Gram-Schmidt against all of them). */
function orthogonalToPrototypes(seed = 1): number[] {
  let v = Array.from({ length: EMBEDDING_DIM }, (_, i) => (Math.sin(i * 12.9898 + seed) * 43758.5453) % 1);
  const basis: number[][] = [];
  for (const p of protos) {
    let b = [...p];
    for (const q of basis) {
      const d = b.reduce((a, x, i) => a + x * q[i], 0);
      b = b.map((x, i) => x - d * q[i]);
    }
    const n = Math.hypot(...b);
    basis.push(b.map((x) => x / n));
  }
  for (const q of basis) {
    const d = v.reduce((a, x, i) => a + x * q[i], 0);
    v = v.map((x, i) => x - d * q[i]);
  }
  return v;
}

describe('input check rule (models/input_check.json)', () => {
  it('loads the committed threshold, version and five prototypes', () => {
    expect(INPUT_CHECK_THRESHOLD).toBe(config.threshold);
    expect(INPUT_CHECK_THRESHOLD).toBe(summary.threshold);
    expect(INPUT_CHECK_VERSION).toBe(summary.version);
    expect(INPUT_CHECK_THRESHOLD).toBeGreaterThan(0);
    expect(INPUT_CHECK_THRESHOLD).toBeLessThan(1);
    expect(config.prototypes.classes).toEqual(['C1', 'C2', 'C3', 'C4', 'C5']);
    expect(protos).toHaveLength(5);
    for (const p of protos) expect(Math.hypot(...p)).toBeCloseTo(1, 6);
  });

  it('scores a prototype at 1 whatever its length, and passes it', () => {
    for (const p of protos) {
      expect(inputCheckScore(p)).toBeCloseTo(1, 6);
      expect(checkEmbedding(p.map((x) => x * 0.2)).passed).toBe(true);
    }
  });

  it('fails an embedding unlike every prototype', () => {
    const r = checkEmbedding(orthogonalToPrototypes());
    expect(r.score).toBeCloseTo(0, 9);
    expect(r.passed).toBe(false);
  });

  it('passes just above the threshold and fails just below it', () => {
    const o = orthogonalToPrototypes(2);
    const on = Math.hypot(...o);
    const at = (c: number) => protos[0].map((x, i) => c * x + Math.sqrt(1 - c * c) * (o[i] / on));
    expect(inputCheckScore(at(INPUT_CHECK_THRESHOLD))).toBeCloseTo(INPUT_CHECK_THRESHOLD, 6);
    expect(checkEmbedding(at(INPUT_CHECK_THRESHOLD + 1e-4)).passed).toBe(true);
    expect(checkEmbedding(at(INPUT_CHECK_THRESHOLD - 1e-4)).passed).toBe(false);
  });

  it.each([
    ['too short', new Array(10).fill(1)],
    ['zero', new Array(EMBEDDING_DIM).fill(0)],
    ['NaN', new Array(EMBEDDING_DIM).fill(NaN)],
  ])('rejects a %s embedding', (_, e) => {
    expect(() => inputCheckScore(e)).toThrow();
  });

  it('checks results at the worker boundary', () => {
    const ok = { score: 0.9, threshold: 0.8, passed: true, version: 'v' };
    expect(validateInputCheck(ok)).toBe(ok);
    expect(() => validateInputCheck({ ...ok, passed: false })).toThrow();
    expect(() => validateInputCheck({ ...ok, score: 1.5 })).toThrow();
    expect(() => validateInputCheck({ ...ok, score: NaN, passed: false })).toThrow();
    expect(() => validateInputCheck({ ...ok, version: '' })).toThrow();
  });

  it('area-resizes the 300 x 300 model input to the 224 x 224 gate input', () => {
    const flat = toGateInput(new Uint8Array(300 * 300 * 3).fill(77));
    expect(flat).toHaveLength(GATE_SIZE * GATE_SIZE * 3);
    expect(flat.every((v) => v === 77)).toBe(true);
    expect(() => toGateInput(new Uint8Array(10))).toThrow();
  });
});

describe('input check copy', () => {
  const all = [...Object.values(INPUT_CHECK_COPY), safeguardNote(), rejectionAnnouncement()];

  it('uses the agreed wording', () => {
    expect(INPUT_CHECK_COPY.heading).toBe("This doesn't look like a single saffron thread on paper.");
    expect(INPUT_CHECK_COPY.uncheckedLabel).toBe(
      'Unchecked input: this photo did not look like a single saffron thread on paper, so treat the result with extra caution.',
    );
    expect([INPUT_CHECK_COPY.retry, INPUT_CHECK_COPY.tips, INPUT_CHECK_COPY.anyway]).toEqual([
      'Try another photo',
      'Photo tips',
      'Analyse anyway',
    ]);
  });

  it('never hints at pure or counterfeit, never says diagnosis or certification, no em-dashes', () => {
    for (const text of all) {
      expect(text).not.toMatch(
        /\b(pure|purity|fake|counterfeit|adulterat\w*|genuine|authentic|diagnos\w*|certif\w*)\b/i,
      );
      expect(text).not.toContain('—');
    }
  });

  it('states the measured rates from the committed summary', () => {
    const note = safeguardNote();
    const p1 = (x: number) => `${(x * 100).toFixed(1)}%`;
    expect(note).toContain(p1(summary.false_rejection.test.rate));
    expect(note).toContain(p1(summary.false_acceptance.near.rate));
    expect(note).toMatch(/not a guarantee/);
  });
});

describe('InputCheckNotice', () => {
  const photo: RejectedPhoto = {
    id: 1,
    source: { kind: 'upload', name: 'yarn.jpg' },
    displayUrl: 'blob:x',
    width: 640,
    height: 480,
    inputCheck: { score: 0.61, threshold: 0.94, passed: false, version: 'v' },
    file: new Blob(['x'], { type: 'image/jpeg' }),
  };

  it('shows the heading, reason, tip and three actions, focuses the heading, and shows no verdict', async () => {
    const onAnother = vi.fn();
    const onPhotoTips = vi.fn();
    const onAnalyseAnyway = vi.fn();
    render(
      <InputCheckNotice
        photo={photo}
        onAnother={onAnother}
        onPhotoTips={onPhotoTips}
        onAnalyseAnyway={onAnalyseAnyway}
      />,
    );
    expect(screen.getByRole('heading', { name: INPUT_CHECK_COPY.heading })).toHaveFocus();
    expect(screen.getByText(/one thread on light paper/)).toBeInTheDocument();
    expect(screen.queryByTestId('verdict')).toBeNull();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Try another photo/ }));
    await user.click(screen.getByRole('link', { name: 'Photo tips' }));
    await user.click(screen.getByRole('button', { name: 'Analyse anyway' }));
    expect(onAnother).toHaveBeenCalledOnce();
    expect(onPhotoTips).toHaveBeenCalledOnce();
    expect(onAnalyseAnyway).toHaveBeenCalledOnce();
  });

  it('reaches every action with the keyboard, in order', async () => {
    render(
      <InputCheckNotice photo={photo} onAnother={vi.fn()} onPhotoTips={vi.fn()} onAnalyseAnyway={vi.fn()} />,
    );
    const user = userEvent.setup();
    await user.tab();
    expect(screen.getByRole('button', { name: /Try another photo/ })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('link', { name: 'Photo tips' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: 'Analyse anyway' })).toHaveFocus();
  });
});
