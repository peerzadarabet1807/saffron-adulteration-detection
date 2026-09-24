import { motion } from 'framer-motion';
import { CLASS_INFO, GROUP_LABEL, type Group } from '../lib/classes';
import { formatPercent, type RankedClass } from '../lib/prediction';

const GROUP_FILL: Record<Group, string> = { pure: 'var(--pure)', counterfeit: 'var(--fake)' };

export function GroupLegend() {
  return (
    <ul className="flex items-center gap-4 text-xs text-ink-2" aria-label="Legend">
      {(Object.keys(GROUP_FILL) as Group[]).map((group) => (
        <li key={group} className="flex items-center gap-1.5">
          <span
            className="size-2.5 rounded-[3px]"
            style={{ background: GROUP_FILL[group] }}
            aria-hidden="true"
          />
          {GROUP_LABEL[group]}
        </li>
      ))}
    </ul>
  );
}

/** Horizontal bars, most likely class first, coloured by group (pure / counterfeit) and always labelled. */
export function ProbabilityBars({ ranked }: { ranked: RankedClass[] }) {
  return (
    <figure className="space-y-3">
      <figcaption className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-semibold text-ink">Probability for each class</span>
        <GroupLegend />
      </figcaption>
      <ol className="space-y-3" aria-label="Class probabilities, most likely first">
        {ranked.map(({ id, probability }, i) => {
          const info = CLASS_INFO[id];
          return (
            <li key={id} data-testid="prob-row" data-class={id} data-probability={probability.toFixed(6)}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="min-w-0">
                  <span className="font-semibold text-ink">{id}</span>{' '}
                  <span className="text-ink-2">{info.name}</span>{' '}
                  <span className="whitespace-nowrap text-xs text-ink-3">· {GROUP_LABEL[info.group]}</span>
                </span>
                <span className="tabular shrink-0 font-semibold text-ink">{formatPercent(probability)}</span>
              </div>
              <div className="mt-1.5 h-3 w-full rounded-r-[4px] bg-track" aria-hidden="true">
                <motion.div
                  className="h-full origin-left rounded-r-[4px]"
                  style={{
                    width: `${Math.max(probability * 100, 0.6)}%`,
                    background: GROUP_FILL[info.group],
                  }}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 0.7, delay: 0.1 + i * 0.06, ease: [0.2, 0.8, 0.2, 1] }}
                />
              </div>
            </li>
          );
        })}
      </ol>
    </figure>
  );
}
