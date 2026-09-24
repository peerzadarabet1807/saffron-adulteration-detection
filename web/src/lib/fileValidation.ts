/** Upload validation by file signature ("magic bytes"), not by the (often wrong or empty) MIME type. */

export const MAX_FILE_BYTES = 25_000_000; // 25 MB

export type ImageKind = 'jpeg' | 'png' | 'webp' | 'heic';

export type ValidationResult =
  | { ok: true; kind: ImageKind }
  | { ok: false; reason: 'empty' | 'too-large' | 'unsupported-image' | 'not-an-image'; message: string };

const HEIF_BRANDS = new Set(['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'hevm', 'hevs', 'mif1', 'msf1']);

const ascii = (bytes: Uint8Array, start: number, end: number) =>
  String.fromCharCode(...bytes.subarray(start, end));

/** Identify a supported image format from the first bytes of a file. */
export function sniffImageKind(head: Uint8Array): ImageKind | null {
  if (head.length >= 3 && head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return 'jpeg';
  if (head.length >= 8 && ascii(head, 0, 8) === '\x89PNG\r\n\x1a\n') return 'png';
  if (head.length >= 12 && ascii(head, 0, 4) === 'RIFF' && ascii(head, 8, 12) === 'WEBP') return 'webp';
  if (head.length >= 12 && ascii(head, 4, 8) === 'ftyp' && HEIF_BRANDS.has(ascii(head, 8, 12))) return 'heic';
  return null;
}

async function readHead(file: Blob, n = 32): Promise<Uint8Array> {
  return new Uint8Array(await file.slice(0, n).arrayBuffer());
}

/** Decimal units (1 MB = 1,000,000 bytes), trailing ".0" dropped: "25 MB", "47.2 MB", "312 KB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${Math.round(bytes)} B`;
  if (bytes < 1_000_000) return `${Math.round(bytes / 1000)} KB`;
  return `${(bytes / 1_000_000).toFixed(1).replace(/\.0$/, '')} MB`;
}

export async function validateFile(file: Blob & { name?: string }): Promise<ValidationResult> {
  const name = file.name ? `“${file.name}”` : 'This file';
  if (file.size === 0) {
    return { ok: false, reason: 'empty', message: `${name} is empty. Please choose a photo.` };
  }
  if (file.size > MAX_FILE_BYTES) {
    return {
      ok: false,
      reason: 'too-large',
      message: `${name} is ${formatBytes(file.size)}. Please use a photo under ${formatBytes(MAX_FILE_BYTES)}.`,
    };
  }
  const kind = sniffImageKind(await readHead(file));
  if (kind) return { ok: true, kind };
  if (file.type.startsWith('image/')) {
    return {
      ok: false,
      reason: 'unsupported-image',
      message: `${name} is an image format this app can't read. Please use JPEG, PNG, WebP or HEIC.`,
    };
  }
  return {
    ok: false,
    reason: 'not-an-image',
    message: `${name} doesn't look like a photo. Please choose a JPEG, PNG, WebP or HEIC image of a saffron thread.`,
  };
}
