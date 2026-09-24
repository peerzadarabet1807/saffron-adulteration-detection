/** Render a shareable PNG summarising one result (photo, verdict, probabilities, timestamp). */
import { CLASS_INFO } from './classes';
import { formatPercent, type Interpretation } from './prediction';

const W = 1200;
const H = 675;
const C = {
  paper: '#fbf7f0',
  ink: '#1e1a16',
  ink2: '#4b433a',
  ink3: '#6b6155',
  line: '#e7ddcd',
  track: '#efe6d8',
  pure: '#bf8209',
  fake: '#b3322e',
  crimson: '#8b1e1e',
  gold: '#e3a018',
  warn: '#9a3412',
};
const SERIF = '"Fraunces Variable", Georgia, serif';
const SANS = '"Inter Variable", system-ui, sans-serif';

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('could not load the photo for the result card'));
    img.src = url;
  });
}

export interface ResultCardInput {
  displayUrl: string;
  interpretation: Interpretation;
  createdAt: Date;
  pageUrl: string;
}

export async function renderResultCard({ displayUrl, interpretation, createdAt, pageUrl }: ResultCardInput) {
  await document.fonts?.ready;
  const img = await loadImage(displayUrl);
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas is not available');

  ctx.fillStyle = C.paper;
  ctx.fillRect(0, 0, W, H);

  // photo, cover-fitted into the left square
  const side = H;
  const scale = Math.max(side / img.width, side / img.height);
  const sw = side / scale;
  const sh = side / scale;
  ctx.drawImage(img, (img.width - sw) / 2, (img.height - sh) / 2, sw, sh, 0, 0, side, side);

  const x = side + 56;
  const right = W - 56;
  let y = 76;
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = C.crimson;
  ctx.font = `600 22px ${SERIF}`;
  ctx.fillText('Saffron Check', x, y);

  const pure = interpretation.verdict === 'pure';
  y += 70;
  ctx.fillStyle = pure ? C.pure : C.fake;
  ctx.fillRect(x, y - 44, 6, 56);
  ctx.fillStyle = C.ink;
  ctx.font = `600 48px ${SERIF}`;
  ctx.fillText(pure ? 'Likely pure' : 'Likely counterfeit', x + 22, y);
  y += 34;
  ctx.fillStyle = C.ink2;
  ctx.font = `400 20px ${SANS}`;
  const top = interpretation.top;
  ctx.fillText(
    `Most likely: ${top.id} · ${CLASS_INFO[top.id].name} (${formatPercent(top.probability)})`,
    x,
    y,
  );

  if (interpretation.lowConfidence) {
    y += 32;
    ctx.fillStyle = C.warn;
    ctx.font = `600 18px ${SANS}`;
    ctx.fillText('⚠ Low confidence: retake the photo before relying on this', x, y);
  }

  y += 44;
  const barW = right - x;
  for (const { id, probability } of interpretation.ranked) {
    ctx.fillStyle = C.ink;
    ctx.font = `600 17px ${SANS}`;
    ctx.fillText(id, x, y);
    ctx.fillStyle = C.ink2;
    ctx.font = `400 17px ${SANS}`;
    ctx.fillText(CLASS_INFO[id].name, x + 36, y);
    ctx.textAlign = 'right';
    ctx.fillStyle = C.ink;
    ctx.font = `600 17px ${SANS}`;
    ctx.fillText(formatPercent(probability), right, y);
    ctx.textAlign = 'left';
    ctx.fillStyle = C.track;
    ctx.fillRect(x, y + 9, barW, 10);
    ctx.fillStyle = CLASS_INFO[id].group === 'pure' ? C.pure : C.fake;
    ctx.fillRect(x, y + 9, Math.max(2, barW * probability), 10);
    y += 50;
  }

  ctx.fillStyle = C.ink3;
  ctx.font = `400 15px ${SANS}`;
  ctx.fillText(`${createdAt.toLocaleString()} · ResNet50V2, run in the browser`, x, H - 64);
  ctx.fillText('Research prototype. Not a certified food-safety or laboratory test.', x, H - 42);
  ctx.fillStyle = C.crimson;
  ctx.fillText(pageUrl.replace(/^https?:\/\//, ''), x, H - 20);

  return new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed'))), 'image/png'),
  );
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function resultCardFilename(createdAt: Date) {
  const pad = (n: number) => String(n).padStart(2, '0');
  const d = createdAt;
  return `saffron-check-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}.png`;
}
