/**
 * Minimal JPEG header reader: EXIF orientation and the stored (un-rotated) frame size.
 * Used to double-check that the browser applied the EXIF rotation when decoding (the training pipeline ran
 * PIL's ImageOps.exif_transpose on every photo).
 */
export interface JpegInfo {
  orientation: number;
  width: number;
  height: number;
}

export function readJpegInfo(buf: ArrayBuffer): JpegInfo | null {
  const view = new DataView(buf);
  if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return null;
  let orientation = 1;
  let width = 0;
  let height = 0;
  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    if (view.getUint8(offset) !== 0xff) break;
    const marker = view.getUint8(offset + 1);
    if (marker === 0xd9 || marker === 0xda) break; // end of image / start of scan
    const length = view.getUint16(offset + 2);
    const segment = offset + 4;
    if (marker === 0xe1 && length >= 8 && view.getUint32(segment) === 0x45786966) {
      orientation = readOrientation(view, segment + 6) ?? orientation;
    } else if (isStartOfFrame(marker)) {
      height = view.getUint16(segment + 1);
      width = view.getUint16(segment + 3);
    }
    offset += 2 + length;
  }
  return width && height ? { orientation, width, height } : null;
}

/** SOF0-SOF15 except DHT (C4), JPG (C8) and DAC (CC), which share the range but carry no frame size. */
const isStartOfFrame = (m: number) => m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc;

function readOrientation(view: DataView, tiff: number): number | null {
  if (tiff + 8 > view.byteLength) return null;
  const little = view.getUint16(tiff) === 0x4949;
  const ifd = tiff + view.getUint32(tiff + 4, little);
  if (ifd + 2 > view.byteLength) return null;
  const entries = view.getUint16(ifd, little);
  for (let i = 0; i < entries; i++) {
    const entry = ifd + 2 + i * 12;
    if (entry + 12 > view.byteLength) return null;
    if (view.getUint16(entry, little) === 0x0112) return view.getUint16(entry + 8, little);
  }
  return null;
}

/** True when an orientation value swaps width and height (90 / 270 degree rotations). */
export const swapsAxes = (orientation: number) => orientation >= 5 && orientation <= 8;
