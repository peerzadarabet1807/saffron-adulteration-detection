/**
 * Full-frame "squash" resize that reproduces OpenCV's cv2.INTER_AREA, which the model was trained with
 * (cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA): no crop, no letterbox).
 *
 * Browser resamplers (canvas smoothing, createImageBitmap's resizeQuality) differ between engines, so the
 * area filter is implemented here instead. It follows OpenCV's computeResizeAreaTab / resizeArea_: every
 * destination pixel is the overlap-weighted mean of the source pixels it covers, computed separably.
 */

export const MODEL_SIZE = 300;

interface Tap {
  src: number;
  weight: number;
}

/** Per-destination-index list of (source index, weight): a port of OpenCV's computeResizeAreaTab. */
export function areaTaps(srcSize: number, dstSize: number): Tap[][] {
  const scale = srcSize / dstSize;
  const taps: Tap[][] = [];
  for (let dx = 0; dx < dstSize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, srcSize - fsx1);
    let sx2 = Math.floor(fsx2);
    let sx1 = Math.ceil(fsx1);
    sx2 = Math.min(sx2, srcSize - 1);
    sx1 = Math.min(sx1, sx2);
    const row: Tap[] = [];
    if (sx1 - fsx1 > 1e-3) row.push({ src: sx1 - 1, weight: (sx1 - fsx1) / cellWidth });
    for (let sx = sx1; sx < sx2; sx++) row.push({ src: sx, weight: 1 / cellWidth });
    if (fsx2 - sx2 > 1e-3) {
      row.push({ src: sx2, weight: Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth });
    }
    taps.push(row);
  }
  return taps;
}

/** Round half to even, like cvRound / saturate_cast<uchar>. */
function rint(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const toByte = (v: number) => (v <= 0 ? 0 : v >= 255 ? 255 : rint(v));

/**
 * Area-resize interleaved pixels (RGBA or RGB) to `dw x dh` RGB bytes.
 * @param channels channels per source pixel (4 for canvas ImageData, 3 for RGB)
 */
export function areaResizeToRGB(
  src: ArrayLike<number>,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
  channels = 4,
): Uint8Array {
  if (sw < dw || sh < dh) throw new Error(`area resize only downscales (${sw}x${sh} -> ${dw}x${dh})`);
  const out = new Uint8Array(dw * dh * 3);
  const sx = sw / dw;
  const sy = sh / dh;
  const isx = Math.round(sx);
  const isy = Math.round(sy);

  if (Math.abs(sx - isx) < Number.EPSILON && Math.abs(sy - isy) < Number.EPSILON) {
    // Integer factors: OpenCV's resizeAreaFast (plain block mean).
    const area = isx * isy;
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        for (let c = 0; c < 3; c++) {
          let sum = 0;
          for (let j = 0; j < isy; j++) {
            const rowBase = ((y * isy + j) * sw + x * isx) * channels + c;
            for (let i = 0; i < isx; i++) sum += src[rowBase + i * channels];
          }
          out[(y * dw + x) * 3 + c] = area === 4 ? (sum + 2) >> 2 : toByte(Math.fround(sum * Math.fround(1 / area)));
        }
      }
    }
    return out;
  }

  const xTaps = areaTaps(sw, dw);
  const yTaps = areaTaps(sh, dh);
  // Horizontal pass for every source row, then weighted vertical accumulation (float32 like OpenCV).
  const acc = new Float32Array(dw * 3);
  const rowCache = new Map<number, Float32Array>();

  const horizontal = (sy0: number): Float32Array => {
    const cached = rowCache.get(sy0);
    if (cached) return cached;
    const res = new Float32Array(dw * 3);
    const base = sy0 * sw * channels;
    for (let x = 0; x < dw; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (const { src: s, weight } of xTaps[x]) {
        const p = base + s * channels;
        r += src[p] * weight;
        g += src[p + 1] * weight;
        b += src[p + 2] * weight;
      }
      res[x * 3] = r;
      res[x * 3 + 1] = g;
      res[x * 3 + 2] = b;
    }
    rowCache.set(sy0, res);
    return res;
  };

  for (let y = 0; y < dh; y++) {
    acc.fill(0);
    for (const { src: s, weight } of yTaps[y]) {
      const row = horizontal(s);
      for (let i = 0; i < acc.length; i++) acc[i] += row[i] * weight;
    }
    // A source row can be shared by two destination rows at most; drop rows we will not need again.
    const lastSrc = yTaps[y][yTaps[y].length - 1].src;
    for (const key of rowCache.keys()) if (key < lastSrc) rowCache.delete(key);
    for (let i = 0; i < acc.length; i++) out[y * dw * 3 + i] = toByte(acc[i]);
  }
  return out;
}

/** RGB bytes -> float32 values in [0, 255] (the model normalises internally; do NOT scale here). */
export function rgbBytesToFloat(rgb: Uint8Array): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) out[i] = rgb[i];
  return out;
}
