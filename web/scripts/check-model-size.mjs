// Guard for the committed TF.js models (the classifier in public/model, the input check in public/gate):
// every weight shard <= 4 MiB, every shard listed in model.json present, and report each download.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const pub = join(dirname(fileURLToPath(import.meta.url)), '..', 'public');
const MAX_SHARD = 4 * 1024 * 1024;
// the input check's budget (INPUT_CHECK_PROMPT section 4.2): about 25 MB
const BUDGET = { model: Infinity, gate: 25e6 };

let ok = true;
for (const name of ['model', 'gate']) {
  const dir = join(pub, name);
  const manifest = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8'));
  const listed = manifest.weightsManifest.flatMap((g) => g.paths);
  const files = readdirSync(dir).filter((f) => f === 'model.json' || f.endsWith('.bin'));
  let total = 0;
  for (const f of files) {
    const size = statSync(join(dir, f)).size;
    total += size;
    if (f.endsWith('.bin') && size > MAX_SHARD) {
      console.error(`${name}: shard too large: ${f} ${size} bytes`);
      ok = false;
    }
  }
  for (const p of listed) {
    if (!files.includes(p)) {
      console.error(`${name}: model.json references missing shard ${p}`);
      ok = false;
    }
  }
  if (total > BUDGET[name]) {
    console.error(`${name}: ${total} bytes is over the ${BUDGET[name]} byte budget`);
    ok = false;
  }
  console.log(
    `${name}: ${listed.length} shards, total ${(total / 1e6).toFixed(1)} MB, largest shard limit ${MAX_SHARD} bytes`,
  );
}
process.exit(ok ? 0 : 1);
