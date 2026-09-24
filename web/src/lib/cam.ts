/**
 * Class activation map (CAM) post-processing.
 *
 * The model's `cam` output is [1, 10, 10, 5]: for every class c, sum_k w_k^c * A^k over the final
 * ResNet50V2 feature maps. For a GAP + linear head this equals Grad-CAM up to a constant factor, so after
 * ReLU and min-max normalisation it is exactly the Grad-CAM heatmap (see scripts/build_web_model.py).
 */

export const CAM_SIZE = 10;

/** Extract one class channel from a flat NHWC [1, H, W, C] tensor (row-major H x W result). */
export function extractClassMap(
  cam: ArrayLike<number>,
  classIndex: number,
  numClasses: number,
  size = CAM_SIZE,
): Float32Array {
  const out = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) out[i] = cam[i * numClasses + classIndex];
  return out;
}

/** ReLU followed by min-max normalisation to [0, 1]. A map with no positive evidence stays all zero. */
export function reluMinMax(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Math.max(0, values[i]);
    out[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo;
  if (!(range > 0)) return out.fill(0);
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) / range;
  return out;
}

/**
 * Bilinear upsampling with half-pixel centres (the convention of cv2.INTER_LINEAR and canvas smoothing),
 * edges clamped. Interpolating the scalar map *before* colour-mapping avoids the banding you get from
 * scaling an already coloured 10 x 10 image.
 */
export function bilinearUpsample(
  src: ArrayLike<number>,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Float32Array {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), sh - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), sw - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, sw - 1);
      const wx = fx - x0;
      const top = src[y0 * sw + x0] * (1 - wx) + src[y0 * sw + x1] * wx;
      const bottom = src[y1 * sw + x0] * (1 - wx) + src[y1 * sw + x1] * wx;
      out[y * dw + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return out;
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Classic "jet" colormap (same family as the notebook's cv2.COLORMAP_JET). v in [0, 1] -> [r, g, b]. */
export function jet(v: number): [number, number, number] {
  const t = clamp01(v);
  return [
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 3))),
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 2))),
    Math.round(255 * clamp01(1.5 - Math.abs(4 * t - 1))),
  ];
}

/** Colour-map a [0, 1] field into opaque RGBA pixels. */
export function heatmapRGBA(field: ArrayLike<number>): Uint8ClampedArray {
  const out = new Uint8ClampedArray(field.length * 4);
  for (let i = 0; i < field.length; i++) {
    const [r, g, b] = jet(field[i]);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  if (va === 0 || vb === 0) {
    let same = true;
    for (let i = 0; i < n && same; i++) same = Math.abs(a[i] - b[i]) < 1e-6;
    return same ? 1 : 0;
  }
  return cov / Math.sqrt(va * vb);
}
