import { CLASS_INFO, CLASS_ORDER, GROUP_LABEL } from '../lib/classes';
import { COLAB_URL, MODEL_CARD_URL, NOTEBOOK_URL, REPO_URL } from '../lib/links';
import { correctCount, metrics, parity, testErrors } from '../lib/metrics';
import { ConfusionMatrix } from './ConfusionMatrix';
import { ExternalIcon } from './icons';
import { InputCheckAbout } from './InputCheckAbout';
import { Limitations } from './Limitations';

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const f3 = (v: number) => v.toFixed(3);
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;

function StatTile({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="card p-5">
      <p className="text-sm text-ink-2">{label}</p>
      <p className="mt-1 font-sans text-4xl font-semibold tracking-tight text-ink">{value}</p>
      <p className="mt-1 text-sm text-ink-3">{detail}</p>
    </div>
  );
}

export function AboutModel() {
  const { test, validation, thesis_baseline: thesis, model_comparison: comparison } = metrics;
  const errors = testErrors();
  const shipped = parity.variants[parity.shipped_variant];
  const rejected = Object.entries(parity.variants).filter(([name]) => name !== parity.shipped_variant);
  const best = comparison[0];
  const runnerUp = comparison[1];
  const valGap = Math.round((best.val_macro_f1 - runnerUp.val_macro_f1) * validation.n_images);
  const splitSize = (split: 'train' | 'val' | 'test') =>
    Object.values(metrics.data.split_counts[split]).reduce((a, b) => a + b, 0);

  return (
    <section id="about" aria-labelledby="about-heading" className="scroll-mt-20 space-y-10">
      <div className="max-w-3xl">
        <p className="eyebrow">02 · About the model</p>
        <h2 id="about-heading" className="mt-2 font-serif text-3xl font-semibold sm:text-4xl">
          ResNet50V2, fine-tuned on {splitSize('train')} photos, tested on {splitSize('test')} it never saw
        </h2>
        <p className="mt-3 text-ink-2">
          {metrics.data.n_valid_images} usable photos were split {splitSize('train')} / {splitSize('val')} /{' '}
          {splitSize('test')} (train / validation / test), stratified by class. ImageNet transfer learning:
          three backbones (EfficientNetB3, ResNet50V2, DenseNet121) were tuned and fine-tuned, the best one
          was chosen on validation macro-F1, and the held-out test set was evaluated <em>once</em>. It began
          as a B.Tech thesis at IIT Kharagpur, whose from-scratch CNN reached about {pct(thesis.val_accuracy)}{' '}
          validation accuracy.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" data-testid="stat-tiles">
        <StatTile
          label="Test accuracy"
          value={pct(test.accuracy)}
          detail={`${correctCount()} of ${test.n_images} held-out images`}
        />
        <StatTile label="Test macro-F1" value={f3(test.macro_f1)} detail="every class weighted equally" />
        <StatTile label="Macro ROC-AUC" value={f3(test.macro_roc_auc_ovr)} detail="one-vs-rest, test set" />
        <StatTile
          label="Test errors"
          value={String(test.n_images - correctCount())}
          detail={errors.map((e) => `${e.count}× ${e.trueClass} → ${e.predicted}`).join(', ')}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="card overflow-hidden">
          <h3 className="px-5 pt-5 font-serif text-xl font-semibold">Per-class results on the test set</h3>
          <p className="px-5 text-sm text-ink-3">{test.per_class.C1.support} images per class</p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[34rem] text-sm">
              <thead>
                <tr className="border-y border-line bg-surface-2 text-left text-xs uppercase tracking-wider text-ink-3">
                  <th scope="col" className="px-5 py-2 font-semibold">
                    Class
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Precision
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Recall
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    F1
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Thesis F1
                  </th>
                  <th scope="col" className="px-5 py-2 text-right font-semibold">
                    Change
                  </th>
                </tr>
              </thead>
              <tbody className="tabular">
                {CLASS_ORDER.map((id) => {
                  const m = test.per_class[id];
                  const base = thesis.per_class_f1[id];
                  return (
                    <tr key={id} className="border-b border-line last:border-0">
                      <th scope="row" className="px-5 py-2.5 text-left font-normal">
                        <span className="font-semibold text-ink">{id}</span>{' '}
                        <span className="text-ink-2">{CLASS_INFO[id].name}</span>
                        <span className="block text-xs text-ink-3">{GROUP_LABEL[CLASS_INFO[id].group]}</span>
                      </th>
                      <td className="px-3 py-2.5 text-right">{f3(m.precision)}</td>
                      <td className="px-3 py-2.5 text-right">{f3(m.recall)}</td>
                      <td className="px-3 py-2.5 text-right font-semibold text-ink">{f3(m.f1)}</td>
                      <td className="px-3 py-2.5 text-right text-ink-3">{base.toFixed(2)}</td>
                      <td className="px-5 py-2.5 text-right font-medium text-ink-2">{signed(m.f1 - base)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="border-t border-line px-5 py-3 text-xs text-ink-3">
            Thesis figures come from the thesis&apos;s own split and protocol (validation accuracy about{' '}
            {pct(thesis.val_accuracy)}), so the comparison is indicative, not strictly like-for-like.
          </p>
        </div>

        <ConfusionMatrix matrix={test.confusion_matrix} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="card overflow-hidden">
          <h3 className="px-5 pt-5 font-serif text-xl font-semibold">Backbone comparison (validation)</h3>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[30rem] text-sm">
              <thead>
                <tr className="border-y border-line bg-surface-2 text-left text-xs uppercase tracking-wider text-ink-3">
                  <th scope="col" className="px-5 py-2 font-semibold">
                    Backbone
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Macro-F1
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Log-loss
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-semibold">
                    Params
                  </th>
                  <th scope="col" className="px-5 py-2 text-right font-semibold">
                    Train time
                  </th>
                </tr>
              </thead>
              <tbody className="tabular">
                {comparison.map((row) => (
                  <tr key={row.backbone} className="border-b border-line last:border-0">
                    <th scope="row" className="px-5 py-2.5 text-left font-semibold text-ink">
                      {row.backbone}
                      {row.backbone === metrics.best_model && (
                        <span className="ml-2 rounded-full bg-pure-soft px-2 py-0.5 text-xs font-semibold text-pure-ink">
                          selected
                        </span>
                      )}
                    </th>
                    <td className="px-3 py-2.5 text-right">{f3(row.val_macro_f1)}</td>
                    <td className="px-3 py-2.5 text-right">{f3(row.val_loss)}</td>
                    <td className="px-3 py-2.5 text-right">{row.params_M.toFixed(1)} M</td>
                    <td className="px-5 py-2.5 text-right">{row.train_min.toFixed(1)} min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-line px-5 py-3 text-sm text-ink-2">
            A near-tie: on {validation.n_images} validation images, {f3(best.val_macro_f1)} vs{' '}
            {f3(runnerUp.val_macro_f1)} is a difference of {valGap === 1 ? 'one image' : `${valGap} images`}.
            The validation set also drove early stopping and the hyperparameter search, so these scores are
            optimistic; the test set is the number to trust.
          </p>
        </div>

        <div className="card p-5">
          <h3 className="font-serif text-xl font-semibold">Same model, in your browser</h3>
          {shipped?.test && (
            <p className="mt-2 text-sm text-ink-2">
              The TensorFlow.js model ({parity.shipped_variant} weights, {shipped.size_mb.toFixed(0)} MB) was
              checked against the original Keras model on all {shipped.test.images} test images: the same top
              class for {pct(shipped.test.argmaxAgreement, 0)} of them, probabilities within{' '}
              {shipped.test.maxProbDiff.toFixed(3)}, and Grad-CAM maps correlating at r ≥{' '}
              {shipped.test.minCamCorr.toFixed(4)}.
              {rejected.map(([name, v]) =>
                v.test ? (
                  <span key={name}>
                    {' '}
                    Smaller {name} weights ({v.size_mb.toFixed(0)} MB) were rejected: probabilities drifted by
                    up to {v.test.maxProbDiff.toFixed(2)}.
                  </span>
                ) : null,
              )}
            </p>
          )}
          <p className="mt-3 text-sm text-ink-2">
            The Grad-CAM heatmap is computed in the same forward pass. For a global-average-pool + linear
            head, Grad-CAM weights are just the classifier weights, so no gradients are needed in the browser.
          </p>
          <ul className="mt-4 flex flex-wrap gap-2 text-sm">
            {[
              ['Training notebook', NOTEBOOK_URL],
              ['Open in Colab', COLAB_URL],
              ['Model card', MODEL_CARD_URL],
              ['Source code', REPO_URL],
            ].map(([label, href]) => (
              <li key={label}>
                <a
                  href={href}
                  className="inline-flex items-center gap-1.5 rounded-full border border-line-strong px-3 py-1.5 font-medium text-ink hover:border-ink-3"
                >
                  {label} <ExternalIcon className="text-ink-3" />
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <InputCheckAbout />

      <Limitations />
    </section>
  );
}
