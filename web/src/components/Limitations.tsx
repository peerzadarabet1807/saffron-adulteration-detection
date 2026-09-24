import { metrics } from '../lib/metrics';
import { AlertIcon } from './icons';

const FIGURES = `${import.meta.env.BASE_URL}figures/`;

export function Limitations() {
  const perClass = metrics.test.per_class.C1.support;
  const recallStep = ((1 / perClass) * 100).toFixed(1);
  return (
    <section
      id="limitations"
      aria-labelledby="limitations-heading"
      className="scroll-mt-20 rounded-[1.25rem] border border-warn-line/70 bg-surface p-5 sm:p-7"
    >
      <div className="flex items-center gap-3">
        <AlertIcon className="shrink-0 text-2xl text-warn-ink" />
        <h3 id="limitations-heading" className="font-serif text-2xl font-semibold">
          Limitations: read these before trusting a result
        </h3>
      </div>

      <div className="mt-5 grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="space-y-5 text-sm text-ink-2 sm:text-[0.95rem]">
          <div>
            <h4 className="font-semibold text-ink">
              1. The pure classes are partly recognised by their background
            </h4>
            <p className="mt-1">
              Grad-CAM shows that for C3, C4 and C5 the model looks at the thread itself. For the two pure
              classes (C1, C2) its attention is often diffuse and spread over the{' '}
              <strong>background paper</strong>. The reason is in the photos: C1 was shot on a distinct
              beige/grey paper, C2 mostly on bluish or vignetted white paper, while C3–C5 share a similar
              white, vignetted background. This is an acquisition confound, so the perfect C1/C2 test scores
              are likely helped by background cues, not by thread appearance alone.
            </p>
            <p className="mt-2">
              <span className="font-medium text-ink">Next steps:</span> re-photograph every class on one
              standard background under fixed lighting; add background-randomisation augmentation or segment
              and crop the thread; evaluate on an external test set collected under different conditions.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-ink">2. A small test set</h4>
            <p className="mt-1">
              {metrics.test.n_images} test images, {perClass} per class: one error moves a class&apos;s recall
              by about {recallStep} points. Treat per-class differences of that size as noise.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-ink">3. Optimistic validation scores</h4>
            <p className="mt-1">
              The validation set was reused for early stopping, the hyperparameter search and model selection.
              The held-out test score is the number to quote.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-ink">4. Narrow capture conditions</h4>
            <p className="mt-1">
              Every training photo is a single thread on light paper, taken with one phone, of saffron from
              one region (Pampore, Kashmir). Other cameras, lighting, backgrounds or origins may behave
              differently.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-ink">5. A closed set of five answers</h4>
            <p className="mt-1">
              There is no &ldquo;not saffron&rdquo; or &ldquo;mixed sample&rdquo; class, so the model{' '}
              <strong>always</strong> answers with one of the five classes, even for a red thread or a blank
              page. Its probabilities are not calibrated, and it can be confidently wrong.
            </p>
          </div>
        </div>

        <figure className="space-y-2">
          <a
            href={`${FIGURES}gradcam_overview.webp`}
            className="block overflow-hidden rounded-xl ring-1 ring-line"
          >
            <img
              src={`${FIGURES}gradcam_overview.webp`}
              width={1100}
              height={1705}
              loading="lazy"
              decoding="async"
              alt="Grad-CAM overview from the training notebook: twelve held-out test images, each above its heatmap overlay. For the C1 and C2 images the warm regions spread across the paper; for C3, C4 and C5 they sit on the thread."
              className="w-full"
            />
          </a>
          <figcaption className="text-xs text-ink-3">
            Grad-CAM on held-out test images, from the training notebook. Top rows (C1, C2): attention spread
            over the paper. Lower rows (C3–C5): attention on the thread.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}
