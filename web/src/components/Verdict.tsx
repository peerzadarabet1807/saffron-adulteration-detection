import { CLASS_INFO } from '../lib/classes';
import { MIN_TOP2_MARGIN, MIN_TOP_PROBABILITY, formatPercent, type Interpretation } from '../lib/prediction';
import { AlertIcon, ShieldAlertIcon, ShieldCheckIcon } from './icons';

export function VerdictBanner({ interpretation }: { interpretation: Interpretation }) {
  const pure = interpretation.verdict === 'pure';
  const share = pure ? interpretation.groups.pure : interpretation.groups.counterfeit;
  return (
    <div
      data-testid="verdict"
      data-verdict={interpretation.verdict}
      className={`flex items-start gap-4 rounded-2xl border-l-4 px-5 py-4 ${
        pure ? 'border-pure bg-pure-soft' : 'border-fake bg-fake-soft'
      }`}
    >
      {pure ? (
        <ShieldCheckIcon className="mt-1 shrink-0 text-3xl text-pure-ink" />
      ) : (
        <ShieldAlertIcon className="mt-1 shrink-0 text-3xl text-fake-ink" />
      )}
      <div>
        <p
          className={`text-xs font-semibold uppercase tracking-[0.14em] ${pure ? 'text-pure-ink' : 'text-fake-ink'}`}
        >
          Verdict
        </p>
        <p className="font-serif text-2xl font-semibold leading-tight text-ink sm:text-3xl">
          {pure ? 'Likely pure' : 'Likely counterfeit'}
        </p>
        <p className="mt-1 text-sm text-ink-2">
          {formatPercent(share)} of the probability is on the{' '}
          {pure ? 'two pure classes (C1 + C2)' : 'three counterfeit classes (C3 + C4 + C5)'}.
        </p>
      </div>
    </div>
  );
}

export function TopClass({ interpretation }: { interpretation: Interpretation }) {
  const { top } = interpretation;
  const info = CLASS_INFO[top.id];
  return (
    <div data-testid="top-class" data-class={top.id}>
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-3">Most likely class</p>
      <p className="mt-1 flex flex-wrap items-baseline gap-x-3">
        <span className="font-serif text-xl font-semibold text-ink">
          {top.id} · {info.name}
        </span>
        <span className="text-lg font-semibold text-ink-2">{formatPercent(top.probability)}</span>
      </p>
      <p className="mt-0.5 text-sm text-ink-2">{info.description}</p>
    </div>
  );
}

export function LowConfidenceWarning({ interpretation }: { interpretation: Interpretation }) {
  if (!interpretation.lowConfidence) return null;
  const { top, runnerUp, margin, reasons } = interpretation;
  return (
    <div
      role="note"
      data-testid="low-confidence"
      className="flex gap-3 rounded-2xl border border-warn-line bg-warn-soft px-4 py-3 text-sm text-ink-2"
    >
      <AlertIcon className="mt-0.5 shrink-0 text-lg text-warn-ink" />
      <div>
        <p className="font-semibold text-warn-ink">Low confidence: treat this result with caution</p>
        <p className="mt-1">
          {reasons.includes('low-top-probability') &&
            `The top class has only ${formatPercent(top.probability)} probability (below ${formatPercent(MIN_TOP_PROBABILITY, 0)}). `}
          {reasons.includes('small-margin') &&
            `It leads ${runnerUp.id} (${formatPercent(runnerUp.probability)}) by just ${(margin * 100).toFixed(1)} points (below ${(MIN_TOP2_MARGIN * 100).toFixed(0)}). `}
          Try retaking the photo following the guidelines. Remember that the model{' '}
          <strong className="text-ink">always picks one of its five classes</strong>, even for a photo that
          isn&apos;t saffron at all.
        </p>
      </div>
    </div>
  );
}
