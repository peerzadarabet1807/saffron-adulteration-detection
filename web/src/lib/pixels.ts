/**
 * ImageBitmap -> model input. Runs in the worker (OffscreenCanvas) or, as a fallback, on the main thread.
 *
 * Training preprocessing: PIL decode -> EXIF transpose -> RGB -> cv2.resize(..., (300, 300), INTER_AREA),
 * i.e. the full frame squashed to 300 x 300 with an area filter, no crop and no letterbox.
 */
import { areaResizeToRGB, MODEL_SIZE } from './resize';

/** Mobile Safari refuses canvases above 16.7 M pixels; larger photos are pre-shrunk on the canvas. */
export const MAX_CANVAS_PIXELS = 16_777_216;

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;
type Any2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

export function createCanvas(width: number, height: number): AnyCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** Size of the canvas the bitmap is drawn onto before the area filter (never below the model size). */
export function sourceCanvasSize(width: number, height: number): { width: number; height: number } {
  const scale = Math.min(1, Math.sqrt(MAX_CANVAS_PIXELS / (width * height)));
  return {
    width: Math.max(MODEL_SIZE, Math.floor(width * scale)),
    height: Math.max(MODEL_SIZE, Math.floor(height * scale)),
  };
}

/** Decoded pixels at (up to) native resolution, as RGBA. */
export function bitmapToRGBA(bitmap: ImageBitmap): {
  data: Uint8ClampedArray;
  width: number;
  height: number;
} {
  const { width, height } = sourceCanvasSize(bitmap.width, bitmap.height);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as Any2D | null;
  if (!ctx) throw new Error('2D canvas is not available');
  if (width !== bitmap.width || height !== bitmap.height) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
  }
  ctx.drawImage(bitmap, 0, 0, width, height);
  return { data: ctx.getImageData(0, 0, width, height).data, width, height };
}

/** RGBA pixels of any size -> 300 x 300 x 3 bytes, exactly like cv2.INTER_AREA. */
export function toModelInput(data: Uint8ClampedArray, width: number, height: number): Uint8Array {
  return areaResizeToRGB(data, width, height, MODEL_SIZE, MODEL_SIZE, 4);
}

/** 300 x 300 RGB bytes -> RGBA for display ("what the model sees"). */
export function rgbToRGBA(rgb: Uint8Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    out[j] = rgb[i];
    out[j + 1] = rgb[i + 1];
    out[j + 2] = rgb[i + 2];
    out[j + 3] = 255;
  }
  return out;
}
