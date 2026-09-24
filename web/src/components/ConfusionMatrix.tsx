import { CLASS_INFO, CLASS_ORDER } from '../lib/classes';
import { testErrors } from '../lib/metrics';

/**
 * Test confusion matrix as an accessible table. Cell fill is a single-hue sequential ramp (surface -> crimson)
 * scaled to the row total; the count is always printed, with ink chosen for contrast against the fill.
 */
export function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  return (
    <div className="card p-5">
      <h3 className="font-serif text-xl font-semibold">Confusion matrix (test)</h3>
      <p className="text-sm text-ink-3">rows: true class · columns: predicted class</p>
      <div className="mt-4 overflow-x-auto">
        <table
          className="mx-auto border-separate border-spacing-[3px] text-sm tabular"
          data-testid="confusion-matrix"
        >
          <thead>
            <tr>
              <td />
              {CLASS_ORDER.map((id) => (
                <th key={id} scope="col" className="w-12 pb-1 text-center text-xs font-semibold text-ink-2">
                  {id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, t) => {
              const total = row.reduce((a, b) => a + b, 0) || 1;
              return (
                <tr key={CLASS_ORDER[t]}>
                  <th scope="row" className="pr-2 text-right text-xs font-semibold text-ink-2">
                    <abbr title={CLASS_INFO[CLASS_ORDER[t]].name} className="no-underline">
                      {CLASS_ORDER[t]}
                    </abbr>
                  </th>
                  {row.map((count, p) => {
                    const share = count / total;
                    const strong = share >= 0.75; // white text only on the deep end of the ramp
                    const label = `True ${CLASS_ORDER[t]}, predicted ${CLASS_ORDER[p]}: ${count} image${count === 1 ? '' : 's'}`;
                    return (
                      <td
                        key={p}
                        title={label}
                        aria-label={label}
                        className={`h-11 w-12 rounded-[4px] text-center font-semibold ${
                          strong ? 'text-white' : count ? 'text-ink' : 'text-ink-3'
                        } ${t !== p && count > 0 ? 'ring-2 ring-fake ring-offset-1 ring-offset-surface' : ''}`}
                        style={{
                          background:
                            count === 0
                              ? 'var(--surface-2)'
                              : `color-mix(in oklab, var(--fake) ${Math.round(18 + 82 * share)}%, var(--surface))`,
                        }}
                      >
                        {count}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-sm text-ink-2">
        {testErrors()
          .map(
            (e) =>
              `${e.count} × ${CLASS_INFO[e.trueClass].name.toLowerCase()} (${e.trueClass}) predicted as ${CLASS_INFO[e.predicted].name.toLowerCase()} (${e.predicted})`,
          )
          .join('; ')}
        . C4 vs C5 was also the thesis CNN&apos;s main confusion. One of these mistakes is among the samples
        above, made with high confidence: the probabilities are not calibrated.
      </p>
    </div>
  );
}
