// Guard for the committed TF.js model: every weight shard <= 4 MiB, and report the total download.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'model');
const MAX_SHARD = 4 * 1024 * 1024;

const manifest = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8'));
const listed = manifest.weightsManifest.flatMap((g) => g.paths);
const files = readdirSync(dir);
let total = 0;
let ok = true;
for (const f of files) {
  const size = statSync(join(dir, f)).size;
  total += size;
  if (f.endsWith('.bin') && size > MAX_SHARD) {
    console.error(`shard too large: ${f} ${size} bytes`);
    ok = false;
  }
}
for (const p of listed) {
  if (!files.includes(p)) {
    console.error(`model.json references missing shard ${p}`);
    ok = false;
  }
}
console.log(
  `${listed.length} shards, total ${(total / 1e6).toFixed(1)} MB, largest shard limit ${MAX_SHARD} bytes`,
);
process.exit(ok ? 0 : 1);
