import { useEffect, useRef } from 'react';
import type { RejectedPhoto } from '../hooks/useClassifier';
import { INPUT_CHECK_COPY, safeguardNote } from '../lib/inputCheckCopy';
import { AlertIcon, RefreshIcon } from './icons';

interface Props {
  photo: RejectedPhoto;
  onAnother: () => void;
  onPhotoTips: () => void;
  onAnalyseAnyway: () => void;
}

/** Soft block: the input check turned the photo away, so no verdict, bars or heatmap are shown. */
export function InputCheckNotice({ photo, onAnother, onPhotoTips, onAnalyseAnyway }: Props) {
  const heading = useRef<HTMLHeadingElement>(null);

  // Move focus to the notice once it appears, as for a result.
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    document
      .getElementById('result')
      ?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }, [photo.id]);

  return (
    <div
      className="card overflow-hidden"
      data-testid="input-check-notice"
      data-score={photo.inputCheck.score.toFixed(4)}
    >
      <div className="grid gap-6 p-5 sm:grid-cols-[minmax(0,11rem)_minmax(0,1fr)] sm:p-6">
        <figure className="space-y-2">
          <img
            src={photo.displayUrl}
            alt={`What you submitted (${photo.source.name || 'your upload'}), shown upright.`}
            className="max-h-52 w-full rounded-xl border border-line bg-surface-2 object-contain"
          />
          <figcaption className="text-xs text-ink-3">
            {photo.width} × {photo.height} px
          </figcaption>
        </figure>

        <div className="space-y-4">
          <div className="rounded-2xl border border-warn-line bg-warn-soft px-4 py-4">
            <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-warn-ink">
              <AlertIcon /> Input check
            </p>
            <h2
              id="result-heading"
              ref={heading}
              tabIndex={-1}
              className="mt-1 font-serif text-2xl font-semibold leading-tight text-warn-ink outline-none"
            >
              {INPUT_CHECK_COPY.heading}
            </h2>
            <p className="mt-2 text-sm text-ink-2">{INPUT_CHECK_COPY.why}</p>
            <p className="mt-2 text-sm text-ink-2">
              <strong className="font-semibold text-ink">Tip:</strong> {INPUT_CHECK_COPY.tip}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button type="button" className="btn btn-primary" onClick={onAnother}>
              <RefreshIcon /> {INPUT_CHECK_COPY.retry}
            </button>
            <a
              href="#photo-guidelines"
              className="btn btn-secondary"
              onClick={(e) => {
                e.preventDefault();
                onPhotoTips();
              }}
            >
              {INPUT_CHECK_COPY.tips}
            </a>
            <button
              type="button"
              className="rounded px-1 py-2 text-sm font-medium text-ink-2 underline underline-offset-4 hover:text-ink"
              onClick={onAnalyseAnyway}
              data-testid="analyse-anyway"
            >
              {INPUT_CHECK_COPY.anyway}
            </button>
          </div>

          <p className="text-sm text-ink-3">{safeguardNote()}</p>
        </div>
      </div>
    </div>
  );
}
