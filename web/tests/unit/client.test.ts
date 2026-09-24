import { describe, expect, it } from 'vitest';
import { backendFromUrl, chooseRunsIn } from '../../src/worker/client';

describe('where TensorFlow.js runs', () => {
  it('uses the worker whenever workers can use WebGL', () => {
    expect(chooseRunsIn(undefined, true, true)).toBe('worker');
  });

  it('falls back to WebGL on the page when workers lack WebGL (e.g. Safari 16)', () => {
    expect(chooseRunsIn(undefined, false, true)).toBe('main-thread');
    expect(chooseRunsIn('webgl', false, true)).toBe('main-thread');
  });

  it('keeps CPU/WASM work in the worker', () => {
    expect(chooseRunsIn('wasm', false, true)).toBe('worker');
    expect(chooseRunsIn('cpu', true, true)).toBe('worker');
    expect(chooseRunsIn(undefined, false, false)).toBe('worker');
  });

  it('reads the ?backend= override', () => {
    expect(backendFromUrl('?backend=wasm')).toBe('wasm');
    expect(backendFromUrl('?backend=nonsense')).toBeUndefined();
    expect(backendFromUrl('')).toBeUndefined();
  });
});
