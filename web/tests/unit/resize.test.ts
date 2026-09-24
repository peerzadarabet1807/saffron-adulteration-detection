import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { rgbToRGBA, sourceCanvasSize, toModelInput, MAX_CANVAS_PIXELS } from '../../src/lib/pixels';
import { areaResizeToRGB, areaTaps, MODEL_SIZE, rgbBytesToFloat } from '../../src/lib/resize';

interface Fixture {
  opencv: string;
  cases: { src: [number, number]; dst: [number, number]; rgb: string }[];
}
const fixture = JSON.parse(
  readFileSync(join(__dirname, '..', 'fixtures', 'inter_area.json'), 'utf8'),
) as Fixture;

/** Must match `synthetic()` in scripts/make_resize_fixture.py. Returns RGBA. */
function synthetic(w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) out[p + c] = (x * 7 + y * 13 + c * 29 + ((x * y) % 17) * 5) % 256;
      out[p + 3] = 255;
    }
  return out;
}

describe('areaResizeToRGB reproduces cv2.INTER_AREA', () => {
  for (const { src, dst, rgb } of fixture.cases) {
    it(`${src.join('x')} -> ${dst.join('x')} (OpenCV ${fixture.opencv})`, () => {
      const expected = Uint8Array.from(Buffer.from(rgb, 'base64'));
      const actual = areaResizeToRGB(synthetic(src[0], src[1]), src[0], src[1], dst[0], dst[1], 4);
      expect(actual).toHaveLength(expected.length);
      let maxDiff = 0;
      let exact = 0;
      for (let i = 0; i < expected.length; i++) {
        const d = Math.abs(actual[i] - expected[i]);
        maxDiff = Math.max(maxDiff, d);
        if (d === 0) exact++;
      }
      expect(maxDiff).toBeLessThanOrEqual(1); // float rounding only
      expect(exact / expected.length).toBeGreaterThan(0.99);
    });
  }

  it('area weights for each output pixel sum to 1', () => {
    for (const [s, d] of [
      [3024, 300],
      [4032, 300],
      [97, 13],
    ]) {
      for (const taps of areaTaps(s, d)) {
        expect(taps.reduce((a, t) => a + t.weight, 0)).toBeCloseTo(1, 6);
      }
    }
  });

  it('refuses to upscale', () => {
    expect(() => areaResizeToRGB(new Uint8ClampedArray(4 * 4 * 4), 4, 4, 8, 8)).toThrow(/only downscales/);
  });
});

describe('model input preprocessing', () => {
  it('turns any RGBA frame into 300 x 300 x 3 uint8, squashing the full frame (no crop)', () => {
    const w = 403;
    const h = 302; // non-square, like a 4:3 phone photo
    const out = toModelInput(synthetic(w, h), w, h);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(out).toHaveLength(MODEL_SIZE * MODEL_SIZE * 3);
  });

  it('keeps raw 0-255 values as float32 (normalisation lives inside the model)', () => {
    const f = rgbBytesToFloat(Uint8Array.from([0, 128, 255]));
    expect(f).toBeInstanceOf(Float32Array);
    expect(Array.from(f)).toEqual([0, 128, 255]);
  });

  it('never shrinks a normal phone photo before the area filter', () => {
    expect(sourceCanvasSize(3024, 3024)).toEqual({ width: 3024, height: 3024 });
    expect(sourceCanvasSize(4032, 3024)).toEqual({ width: 4032, height: 3024 });
  });

  it('caps huge photos at the mobile canvas limit and never goes below the model size', () => {
    const big = sourceCanvasSize(8064, 6048);
    expect(big.width * big.height).toBeLessThanOrEqual(MAX_CANVAS_PIXELS);
    expect(big.width / big.height).toBeCloseTo(8064 / 6048, 2);
    expect(sourceCanvasSize(120, 90)).toEqual({ width: 300, height: 300 });
  });

  it('converts the model input back to opaque RGBA for display', () => {
    expect(Array.from(rgbToRGBA(Uint8Array.from([1, 2, 3, 4, 5, 6])))).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
  });
});
