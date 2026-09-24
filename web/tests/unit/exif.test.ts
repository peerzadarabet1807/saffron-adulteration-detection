import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { readJpegInfo, swapsAxes } from '../../src/lib/exif';

const buf = (path: string) => {
  const b = readFileSync(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

describe('readJpegInfo', () => {
  it('reads EXIF orientation 6 and the stored frame size', () => {
    const info = readJpegInfo(buf(join(__dirname, '..', 'fixtures', 'exif-rotated.jpg')));
    expect(info).toEqual({ orientation: 6, width: 600, height: 600 });
    expect(swapsAxes(info!.orientation)).toBe(true);
  });

  it('defaults to orientation 1 for upright images', () => {
    const info = readJpegInfo(buf(join(__dirname, '..', '..', 'public', 'samples', 'sample-03.jpg')));
    expect(info?.orientation).toBe(1);
    expect(info?.width).toBeGreaterThan(0);
  });

  it('returns null for non-JPEG data', () => {
    expect(readJpegInfo(new TextEncoder().encode('not a jpeg').buffer as ArrayBuffer)).toBeNull();
  });
});
