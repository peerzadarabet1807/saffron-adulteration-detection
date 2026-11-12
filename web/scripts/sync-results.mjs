// Copy the training run's metrics (the single source of truth in ../results) into the web app.
// The About section renders these files; nothing in the UI hard-codes a metric.
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const root = join(web, '..');
const files = [
  [join(root, 'results', 'metrics.json'), join(web, 'src', 'data', 'metrics.json')],
  [join(root, 'models', 'tfjs_parity.json'), join(web, 'src', 'data', 'tfjs_parity.json')],
  [join(root, 'models', 'class_names.json'), join(web, 'src', 'data', 'class_names.json')],
  [join(root, 'models', 'input_check.json'), join(web, 'src', 'data', 'input_check.json')],
  [
    join(root, 'results', 'input_check', 'summary.json'),
    join(web, 'src', 'data', 'input_check_summary.json'),
  ],
];

mkdirSync(join(web, 'src', 'data'), { recursive: true });
for (const [from, to] of files) {
  if (!existsSync(from)) {
    console.error(`missing ${from}`);
    process.exit(1);
  }
  copyFileSync(from, to);
  console.log(`synced ${from.slice(root.length + 1)} -> web/${to.slice(web.length + 1)}`);
}
