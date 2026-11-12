import { INPUT_CHECK_SUMMARY as IC } from '../lib/inputCheckCopy';
import { REPO_URL } from '../lib/links';

interface Rate {
  k: number;
  n: number;
  rate: number;
  ci95: number[];
}

const p1 = (x: number) => `${(x * 100).toFixed(1)}%`;

const cell = (r: Rate) => (
  <>
    <strong className="text-ink">{p1(r.rate)}</strong> ({r.k} of {r.n})
    <span className="block text-xs text-ink-3">
      {p1(r.ci95[0])} to {p1(r.ci95[1])}
    </span>
  </>
);

/** The About section's "Input check" entry; every number comes from results/input_check/summary.json. */
export function InputCheckAbout() {
  const fa = IC.false_acceptance;
  const counts = Object.values(IC.prototype_photos_per_class);
  const v = IC.ood_test_confident_answer;
  const rows: [string, string, Rate][] = [
    ['Saffron-thread photos turned away', 'held-out test photos, all five classes', IC.false_rejection.test],
    [
      'Other photos let through',
      'look-alikes: red thread, fabric, chili threads, hair, flowers, jars…',
      fa.near,
    ],
    ['', 'unrelated: people, documents, screenshots, scenes, objects', fa.far],
    ['Grey zone let through', 'safflower, corn silk, loose saffron not on paper', fa.grey],
  ];
  return (
    <div className="card p-5 sm:p-6" id="input-check" data-testid="input-check-about">
      <h3 className="font-serif text-2xl font-semibold">Input check</h3>
      <div className="mt-3 space-y-3 text-sm text-ink-2 sm:text-[0.95rem]">
        <p>
          Before the classifier runs, a small second model checks whether the photo looks like the training
          photos: one thread on light paper. It is the image encoder of TinyCLIP ViT-8M/16 (Microsoft, MIT
          licence, {(__GATE_BYTES__ / 1e6).toFixed(1)} MB download), comparing the photo with the average of
          the {Math.min(...counts)} to {Math.max(...counts)} training photos of each class. It runs in your
          browser like the classifier. It says nothing about whether a thread is pure or counterfeit: it
          passes counterfeit threads as readily as pure ones, because both are what the classifier was built
          for.
        </p>
        <p>
          If the photo is not similar enough, no result is shown; you can still choose Analyse anyway, and the
          result is then labelled as unchecked. The threshold was set on the validation photos alone. Measured
          on the test photos and on openly licensed photos of other things that played no part in that choice
          (95% intervals):
        </p>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[30rem] text-left text-sm tabular-nums">
          <caption className="sr-only">Error rates of the input check on the test sets</caption>
          <tbody>
            {rows.map(([what, set, r]) => (
              <tr key={set} className="border-b border-line align-top">
                <th scope="row" className="py-2 pr-3 font-semibold text-ink">
                  {what}
                </th>
                <td className="py-2 pr-3 text-ink-2">{set}</td>
                <td className="py-2">{cell(r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-sm text-ink-3">
        A safeguard, not a guarantee. With {IC.false_rejection.test.n} test photos the intervals are wide, and
        the check compares with photos taken on a few kinds of paper with one phone, so threads on other paper
        or in other light may be turned away more often (not measured: no such photos were available). All{' '}
        {IC.samples.n} sample photos below pass. Post-hoc: on {v.classifier_alone.n} test photos of other
        things, the classifier alone would give a confident answer for {p1(v.classifier_alone.rate)}; with the
        check in front, for {p1(v.after_input_check.rate)}.
      </p>
      <a
        className="mt-3 inline-block text-sm font-medium text-ink underline underline-offset-4"
        href={`${REPO_URL}/blob/main/results/input_check/evaluation.md`}
      >
        The full evaluation
      </a>
    </div>
  );
}
