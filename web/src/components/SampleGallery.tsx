import type { SamplesState } from '../hooks/useSamples';
import type { ClassId } from '../lib/classes';
import { SAMPLES_BASE, type SampleInfo } from '../lib/samples';
import { CheckIcon, CrossIcon } from './icons';

export interface Revealed {
  predicted: ClassId;
}

interface Props {
  samples: SamplesState;
  revealed: Record<string, Revealed>;
  activeFile?: string;
  busy: boolean;
  onPick: (sample: SampleInfo, index: number) => void;
}

export function SampleGallery({ samples, revealed, activeFile, busy, onPick }: Props) {
  return (
    <section aria-labelledby="samples-heading" className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 id="samples-heading" className="font-serif text-xl font-semibold">
          No saffron at hand? Try a held-out test image
        </h2>
        <p className="text-sm text-ink-3">labels are revealed after classification</p>
      </div>
      <p className="mt-1 max-w-3xl text-sm text-ink-2">
        Ten photos from the test split: the model never saw them during training, tuning or model selection.
        Per class, one typical image and the hardest one (including a real mistake).
      </p>

      {samples.status === 'loading' && (
        <ul className="mt-4 grid grid-cols-5 gap-2 sm:gap-3" aria-busy="true" aria-label="Loading samples">
          {Array.from({ length: 10 }, (_, i) => (
            <li key={i} className="aspect-square animate-pulse rounded-xl bg-surface-2" />
          ))}
        </ul>
      )}
      {samples.status === 'error' && (
        <p role="alert" className="mt-4 text-sm text-fake-ink">
          {samples.message}
        </p>
      )}
      {samples.status === 'ready' && (
        <ul className="mt-4 grid grid-cols-5 gap-2 sm:gap-3" data-testid="samples">
          {samples.samples.map((sample, i) => {
            const shown = revealed[sample.file];
            const correct = shown && shown.predicted === sample.trueClass;
            const active = activeFile === sample.file;
            return (
              <li key={sample.file}>
                <button
                  type="button"
                  onClick={() => onPick(sample, i)}
                  disabled={busy}
                  data-testid="sample"
                  data-file={sample.file}
                  aria-label={`Classify sample ${i + 1}${shown ? `, true class ${sample.trueClass}` : ''}`}
                  aria-current={active ? 'true' : undefined}
                  className={`group relative block w-full overflow-hidden rounded-xl ring-1 transition disabled:cursor-wait ${
                    active ? 'ring-2 ring-crimson' : 'ring-line hover:ring-line-strong'
                  }`}
                >
                  <img
                    src={`${SAMPLES_BASE}${sample.thumb}`}
                    alt=""
                    width={200}
                    height={200}
                    loading="lazy"
                    decoding="async"
                    className="aspect-square w-full object-cover transition-transform duration-300 group-hover:scale-[1.04]"
                  />
                  <span className="absolute left-1.5 top-1.5 rounded-md bg-surface/90 px-1.5 py-0.5 text-[11px] font-semibold text-ink shadow-sm">
                    {i + 1}
                  </span>
                  {shown && (
                    <span
                      className={`absolute inset-x-1.5 bottom-1.5 flex items-center justify-center gap-1 rounded-md px-1 py-0.5 text-[11px] font-semibold ${
                        correct ? 'bg-pure-soft text-pure-ink' : 'bg-fake-soft text-fake-ink'
                      }`}
                    >
                      {correct ? <CheckIcon /> : <CrossIcon />}
                      {sample.trueClass}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
