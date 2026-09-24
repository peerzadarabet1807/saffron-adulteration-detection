import { describe, expect, it } from 'vitest';
import {
  bilinearUpsample,
  extractClassMap,
  heatmapAlpha,
  heatmapRGBA,
  jet,
  pearson,
  reluMinMax,
} from '../../src/lib/cam';

describe('extractClassMap', () => {
  it('reads one channel of a flat NHWC [1, H, W, C] tensor in row-major order', () => {
    const size = 2;
    const classes = 3;
    // value encodes (y, x, c) as 100y + 10x + c
    const cam = new Float32Array(size * size * classes);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++)
        for (let c = 0; c < classes; c++) cam[(y * size + x) * classes + c] = 100 * y + 10 * x + c;
    expect(Array.from(extractClassMap(cam, 2, classes, size))).toEqual([2, 12, 102, 112]);
  });
});

describe('reluMinMax', () => {
  it('clips negatives, then scales to [0, 1]', () => {
    expect(Array.from(reluMinMax([-3, 0, 2, 4]))).toEqual([0, 0, 0.5, 1]);
  });

  it('uses the minimum after ReLU, not zero, when every value is positive', () => {
    expect(Array.from(reluMinMax([2, 3, 4]))).toEqual([0, 0.5, 1]);
  });

  it('returns an all-zero map when there is no positive evidence', () => {
    expect(Array.from(reluMinMax([-1, -2, -0.5]))).toEqual([0, 0, 0]);
    expect(Array.from(reluMinMax([5, 5, 5]))).toEqual([0, 0, 0]);
  });
});

describe('bilinearUpsample', () => {
  it('produces the requested size and keeps a constant field constant', () => {
    const out = bilinearUpsample(new Float32Array(100).fill(0.7), 10, 10, 300, 300);
    expect(out).toHaveLength(300 * 300);
    expect(out.every((v) => Math.abs(v - 0.7) < 1e-6)).toBe(true);
  });

  it('uses half-pixel centres like cv2.INTER_LINEAR (2 -> 4 samples)', () => {
    // cv2.resize(np.array([[0, 1]], np.float32), (4, 1), interpolation=cv2.INTER_LINEAR) -> [0, .25, .75, 1]
    expect(Array.from(bilinearUpsample([0, 1], 2, 1, 4, 1))).toEqual([0, 0.25, 0.75, 1]);
  });

  it('stays within the range of the source values', () => {
    const src = Float32Array.from({ length: 100 }, (_, i) => ((i * 37) % 100) / 99);
    const out = bilinearUpsample(src, 10, 10, 64, 48);
    expect(Math.min(...out)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...out)).toBeLessThanOrEqual(1);
  });
});

describe('colormap', () => {
  it('maps the ends of the range to dark blue and dark red', () => {
    expect(jet(0)).toEqual([0, 0, 128]);
    expect(jet(1)).toEqual([128, 0, 0]);
    expect(jet(0.5)).toEqual([128, 255, 128]);
  });

  it('makes weak regions faint and strong regions opaque', () => {
    expect(heatmapAlpha(0)).toBeCloseTo(0.12);
    expect(heatmapAlpha(1)).toBe(1);
    const px = heatmapRGBA([0, 1]);
    expect(px).toHaveLength(8);
    expect(px[3]).toBeLessThan(40);
    expect(px[7]).toBe(255);
  });
});

describe('pearson', () => {
  it('is 1 for identical maps, -1 for inverted ones', () => {
    expect(pearson([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
    expect(pearson([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1);
  });

  it('treats two identical constant maps as a perfect match', () => {
    expect(pearson([0, 0, 0], [0, 0, 0])).toBe(1);
  });
});
