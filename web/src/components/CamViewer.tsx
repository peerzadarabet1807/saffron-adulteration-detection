import { useCallback, useId, useMemo, useState } from 'react';
import { bilinearUpsample, CAM_SIZE, extractClassMap, heatmapRGBA, reluMinMax } from '../lib/cam';
import { CLASS_INFO, CLASS_ORDER, type ClassId } from '../lib/classes';
import { MODEL_SIZE } from '../lib/resize';

const HEATMAP_SIZE = 300;
/** CSS version of the jet colormap in lib/cam.ts, for the legend */
const JET_GRADIENT =
  'linear-gradient(90deg, rgb(0 0 128), rgb(0 0 255) 12.5%, rgb(0 255 255) 37.5%, rgb(255 255 0) 62.5%, rgb(255 0 0) 87.5%, rgb(128 0 0))';

interface Props {
  displayUrl: string;
  width: number;
  height: number;
  /** flat [10, 10, 5] class activation maps */
  cam: Float32Array;
  /** 300 x 300 RGBA model input */
  input: Uint8ClampedArray;
  predicted: ClassId;
  alt: string;
}

/** Callback ref that paints `pixels` whenever the canvas mounts or the pixels change. */
function usePaintedCanvas(pixels: Uint8ClampedArray, size: number) {
  return useCallback(
    (canvas: HTMLCanvasElement | null) => {
      canvas?.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(pixels), size, size), 0, 0);
    },
    [pixels, size],
  );
}

/** Render with `key={result.id}` so the view state resets for every new result. */
export function CamViewer({ displayUrl, width, height, cam, input, predicted, alt }: Props) {
  const [view, setView] = useState<'photo' | 'model'>('photo');
  const [showCam, setShowCam] = useState(true);
  const [opacity, setOpacity] = useState(0.8);
  const [camClass, setCamClass] = useState<ClassId>(predicted);
  const ids = { opacity: useId(), select: useId(), caption: useId() };

  const { heatmap, empty } = useMemo(() => {
    const map = reluMinMax(extractClassMap(cam, CLASS_ORDER.indexOf(camClass), CLASS_ORDER.length, CAM_SIZE));
    const empty = map.every((v) => v === 0);
    const field = bilinearUpsample(map, CAM_SIZE, CAM_SIZE, HEATMAP_SIZE, HEATMAP_SIZE);
    return { heatmap: heatmapRGBA(field), empty };
  }, [cam, camClass]);

  const heatmapRef = usePaintedCanvas(heatmap, HEATMAP_SIZE);
  const inputRef = usePaintedCanvas(input, MODEL_SIZE);
  const aspect = view === 'photo' ? `${width} / ${height}` : '1 / 1';

  return (
    <figure className="space-y-3" aria-describedby={ids.caption}>
      <div className="flex flex-wrap items-center gap-2">
        <div
          role="group"
          aria-label="Image shown"
          className="inline-flex rounded-full border border-line bg-surface-2 p-0.5 text-sm"
        >
          {(
            [
              ['photo', 'Your photo'],
              ['model', 'What the model sees'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              aria-pressed={view === value}
              onClick={() => setView(value)}
              className={`rounded-full px-3 py-1.5 font-medium transition-colors ${
                view === value
                  ? 'bg-surface text-ink shadow-sm ring-1 ring-line'
                  : 'text-ink-2 hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-pressed={showCam}
          onClick={() => setShowCam((v) => !v)}
          data-testid="cam-toggle"
          className={`rounded-full border px-3 py-1.5 text-sm font-medium transition-colors ${
            showCam
              ? 'border-crimson bg-crimson text-on-crimson'
              : 'border-line-strong text-ink-2 hover:text-ink'
          }`}
        >
          Grad-CAM {showCam ? 'on' : 'off'}
        </button>
      </div>

      <div
        className="relative mx-auto w-full overflow-hidden rounded-2xl bg-surface-2 ring-1 ring-line"
        style={{
          aspectRatio: aspect,
          maxHeight: '32rem',
          maxWidth: `calc(32rem * ${view === 'photo' ? width / height : 1})`,
        }}
      >
        {view === 'photo' ? (
          <img src={displayUrl} alt={alt} className="absolute inset-0 size-full object-fill" />
        ) : (
          <canvas
            ref={inputRef}
            width={MODEL_SIZE}
            height={MODEL_SIZE}
            role="img"
            aria-label="The 300 by 300 pixel image the model actually receives: your whole photo squashed to a square, without cropping."
            className="absolute inset-0 size-full [image-rendering:auto]"
          />
        )}
        <canvas
          ref={heatmapRef}
          width={HEATMAP_SIZE}
          height={HEATMAP_SIZE}
          aria-hidden="true"
          data-testid="cam-overlay"
          className="pointer-events-none absolute inset-0 size-full transition-opacity duration-300"
          style={{ opacity: showCam && !empty ? opacity : 0 }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 text-sm">
        <label htmlFor={ids.select} className="flex items-center gap-2 text-ink-2">
          Explain class
          <select
            id={ids.select}
            value={camClass}
            onChange={(e) => setCamClass(e.target.value as ClassId)}
            className="rounded-lg border border-line-strong bg-surface px-2 py-1.5 text-ink"
          >
            {CLASS_ORDER.map((id) => (
              <option key={id} value={id}>
                {id} · {CLASS_INFO[id].name}
                {id === predicted ? ' (predicted)' : ''}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor={ids.opacity} className="flex items-center gap-2 text-ink-2">
          Opacity
          <input
            id={ids.opacity}
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            disabled={!showCam}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-28 accent-[var(--crimson)]"
          />
        </label>
        <div className="flex items-center gap-2 text-xs text-ink-3" aria-hidden="true">
          less
          <span className="h-2 w-24 rounded-full" style={{ background: JET_GRADIENT }} />
          more influence
        </div>
      </div>

      <figcaption id={ids.caption} className="text-sm text-ink-2">
        {empty ? (
          <>The model found no positive evidence for {camClass} in this photo, so the heatmap is empty.</>
        ) : (
          <>
            <strong className="font-semibold text-ink">How to read this:</strong> warm colours mark the
            regions that raised the model&apos;s score for {camClass} (
            {CLASS_INFO[camClass].name.toLowerCase()}). A trustworthy explanation sits on the thread. For the
            pure classes the model often looks at the paper instead (see{' '}
            <a href="#limitations" className="font-medium text-crimson-ink underline underline-offset-2">
              Limitations
            </a>
            ). The heatmap is a 10 × 10 map, smoothly upsampled.
          </>
        )}
      </figcaption>
    </figure>
  );
}
