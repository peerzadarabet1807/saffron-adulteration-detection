/**
 * Browser image decoding that matches the training pipeline:
 *  - EXIF orientation applied (PIL ImageOps.exif_transpose),
 *  - NO colour management: the training photos are Display-P3 tagged and PIL used the raw values, so
 *    `colorSpaceConversion: "none"` (a browser would otherwise convert P3 -> sRGB and shift the reds),
 *  - HEIC/HEIF (the iPhone default) decoded natively where possible (Safari), otherwise via a lazily
 *    loaded heic2any.
 */
import { readJpegInfo, swapsAxes } from './exif';
import type { ImageKind } from './fileValidation';
import { createCanvas } from './pixels';

const DECODE_OPTIONS: ImageBitmapOptions = {
  imageOrientation: 'from-image',
  colorSpaceConversion: 'none',
  premultiplyAlpha: 'none',
};

export const MIN_SIDE_PX = 64;

export class DecodeError extends Error {}

export async function decodeImage(file: Blob, kind: ImageKind): Promise<ImageBitmap> {
  let source: Blob = file;
  if (kind === 'heic') {
    try {
      return checkSize(await createImageBitmap(file, DECODE_OPTIONS));
    } catch (e) {
      if (e instanceof DecodeError) throw e;
      source = await heicToPng(file);
    }
  }
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(source, DECODE_OPTIONS);
  } catch {
    bitmap = await decodeWithImageElement(source);
  }
  if (kind === 'jpeg') bitmap = await ensureExifOrientation(bitmap, file);
  return checkSize(bitmap);
}

function checkSize(bitmap: ImageBitmap): ImageBitmap {
  if (Math.min(bitmap.width, bitmap.height) < MIN_SIDE_PX) {
    const { width, height } = bitmap;
    bitmap.close();
    throw new DecodeError(`This image is only ${width} × ${height} px. Please use a full-size photo.`);
  }
  return bitmap;
}

async function heicToPng(file: Blob): Promise<Blob> {
  try {
    const { default: heic2any } = await import('heic2any');
    const out = await heic2any({ blob: file, toType: 'image/png' });
    return Array.isArray(out) ? out[0] : out;
  } catch {
    throw new DecodeError('This HEIC photo could not be decoded. Please export it as JPEG and try again.');
  }
}

async function decodeWithImageElement(blob: Blob): Promise<ImageBitmap> {
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return await createImageBitmap(img);
  } catch {
    throw new DecodeError('This file could not be decoded as an image.');
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Safety net for engines that ignore `imageOrientation: "from-image"`: if the JPEG asks for a 90/270 degree
 * rotation and the decoded bitmap still has the stored (un-rotated) size, rotate it here. (Undetectable for
 * square images and 180 degree flips, where modern engines apply EXIF anyway.)
 */
async function ensureExifOrientation(bitmap: ImageBitmap, file: Blob): Promise<ImageBitmap> {
  const info = readJpegInfo(await file.slice(0, 256 * 1024).arrayBuffer());
  if (!info || !swapsAxes(info.orientation) || info.width === info.height) return bitmap;
  if (bitmap.width !== info.width || bitmap.height !== info.height) return bitmap; // already applied
  const canvas = createCanvas(bitmap.height, bitmap.width);
  const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
  const w = bitmap.width;
  const h = bitmap.height;
  // EXIF orientation 5-8 transforms (drawn into a canvas of size h x w)
  const transforms: Record<number, [number, number, number, number, number, number]> = {
    5: [0, 1, 1, 0, 0, 0],
    6: [0, 1, -1, 0, h, 0],
    7: [0, -1, -1, 0, h, w],
    8: [0, -1, 1, 0, 0, w],
  };
  ctx.setTransform(...transforms[info.orientation]);
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return createImageBitmap(canvas);
}

/** A downscaled, already-oriented JPEG of the decoded image for display and the result card. */
export async function makeDisplayUrl(bitmap: ImageBitmap, maxSide = 1400): Promise<string> {
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas is not available');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, width, height);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  if (!blob) throw new Error('Could not render the image preview');
  return URL.createObjectURL(blob);
}
