// @vitest-environment node
// (Node's native File/Blob; jsdom's Blob has no arrayBuffer())
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { formatBytes, MAX_FILE_BYTES, sniffImageKind, validateFile } from '../../src/lib/fileValidation';

const fixtures = join(__dirname, '..', 'fixtures');
const samples = join(__dirname, '..', '..', 'public', 'samples');
const file = (bytes: Uint8Array | string, name: string, type = '') =>
  new File([typeof bytes === 'string' ? bytes : new Uint8Array(bytes)], name, { type });

describe('sniffImageKind', () => {
  it('recognises JPEG, PNG, WebP and HEIC by their signatures', () => {
    expect(sniffImageKind(readFileSync(join(samples, 'sample-01.jpg')).subarray(0, 32))).toBe('jpeg');
    expect(sniffImageKind(readFileSync(join(fixtures, 'sample.heic')).subarray(0, 32))).toBe('heic');
    expect(sniffImageKind(readFileSync(join(samples, 'thumbs', 'sample-01.webp')).subarray(0, 32))).toBe(
      'webp',
    );
    expect(sniffImageKind(Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))).toBe('png');
  });

  it('returns null for anything else', () => {
    expect(sniffImageKind(new TextEncoder().encode('GIF89a......'))).toBeNull();
    expect(sniffImageKind(new TextEncoder().encode('%PDF-1.7'))).toBeNull();
    expect(sniffImageKind(new Uint8Array())).toBeNull();
  });
});

describe('validateFile', () => {
  it('accepts a real photo even when the browser gives no MIME type', async () => {
    const jpeg = readFileSync(join(samples, 'sample-02.jpg'));
    await expect(validateFile(file(jpeg, 'IMG_0042', ''))).resolves.toEqual({ ok: true, kind: 'jpeg' });
  });

  it('accepts HEIC from an iPhone', async () => {
    const heic = readFileSync(join(fixtures, 'sample.heic'));
    await expect(validateFile(file(heic, 'IMG_1234.HEIC', 'image/heic'))).resolves.toEqual({
      ok: true,
      kind: 'heic',
    });
  });

  it('rejects text files with a friendly message', async () => {
    const res = await validateFile(file('hello', 'notes.txt', 'text/plain'));
    expect(res).toMatchObject({ ok: false, reason: 'not-an-image' });
    if (!res.ok) expect(res.message).toMatch(/doesn.t look like a photo/);
  });

  it('rejects files that lie about being an image', async () => {
    const res = await validateFile(file('not really a jpeg', 'fake.jpg', 'image/jpeg'));
    expect(res).toMatchObject({ ok: false, reason: 'unsupported-image' });
  });

  it('rejects empty files and files over 25 MB before reading them', async () => {
    await expect(validateFile(file(new Uint8Array(), 'empty.jpg'))).resolves.toMatchObject({
      reason: 'empty',
    });
    const huge = {
      size: MAX_FILE_BYTES + 1,
      type: 'image/jpeg',
      name: 'huge.jpg',
      slice: () => new Blob(),
    } as unknown as File;
    const res = await validateFile(huge);
    expect(res).toMatchObject({ ok: false, reason: 'too-large' });
    if (!res.ok) expect(res.message).toContain('under 25 MB');
  });
});

describe('formatBytes', () => {
  it('uses decimal units and drops a trailing .0', () => {
    expect(formatBytes(25_000_000)).toBe('25 MB');
    expect(formatBytes(47_189_000)).toBe('47.2 MB');
    expect(formatBytes(64_700)).toBe('65 KB');
    expect(formatBytes(512)).toBe('512 B');
  });
});
