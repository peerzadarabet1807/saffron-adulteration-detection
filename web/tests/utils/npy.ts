/** Minimal reader for NumPy .npy files (format 1.0/2.0, uint8 or little-endian float32, C order). */
export interface NpyArray {
  shape: number[];
  data: Uint8Array | Float32Array;
}

export function readNpy(buf: Uint8Array): NpyArray {
  const magic = String.fromCharCode(...buf.subarray(0, 6));
  if (magic !== '\x93NUMPY') throw new Error('not a .npy file');
  const major = buf[6];
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const offset = (major === 1 ? 10 : 12) + headerLen;
  const header = new TextDecoder().decode(buf.subarray(major === 1 ? 10 : 12, offset));

  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  if (fortran !== 'False') throw new Error('Fortran-ordered arrays are not supported');

  // Copy: a Node Buffer's .slice() is a view on a shared pool, so its .buffer is not just this array.
  const body = new Uint8Array(buf.byteLength - offset);
  body.set(buf.subarray(offset));
  if (descr === '|u1') return { shape, data: new Uint8Array(body.buffer) };
  if (descr === '<f4') return { shape, data: new Float32Array(body.buffer) };
  throw new Error(`unsupported dtype ${descr}`);
}
