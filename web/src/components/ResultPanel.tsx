import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import type { ClassificationResult, JobState, JobStep } from '../hooks/useClassifier';
import { CLASS_INFO } from '../lib/classes';
import { downloadBlob, renderResultCard, resultCardFilename } from '../lib/resultCard';
import { CamViewer } from './CamViewer';
import { CheckIcon, CrossIcon, DownloadIcon, RefreshIcon } from './icons';
import { ProbabilityBars } from './ProbabilityBars';
import { LowConfidenceWarning, TopClass, VerdictBanner } from './Verdict';

const STEP_LABEL: Record<JobStep, string> = {
  reading: 'Checking the file…',
  decoding: 'Decoding the photo…',
  'waiting-for-model': 'Waiting for the model to finish loading…',
  classifying: 'Running the model…',
};

interface Props {
  job: JobState;
  onAnother: () => void;
}

export function ResultPanel({ job, onAnother }: Props) {
  if (job.status === 'idle' || job.status === 'error') return null;
  return (
    <section id="result" aria-labelledby="result-heading" className="scroll-mt-24">
      <AnimatePresence mode="wait">
        {job.status === 'working' ? (
          <motion.div
            key="working"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="card flex items-center gap-4 p-6"
            data-testid="working"
          >
            <span className="relative flex size-10 shrink-0 items-center justify-center" aria-hidden="true">
              <span className="absolute inset-0 animate-spin rounded-full border-2 border-line border-t-crimson" />
            </span>
            <div>
              <h2 id="result-heading" className="font-serif text-xl font-semibold">
                Analysing {job.source.kind === 'sample' ? 'the sample' : 'your photo'}
              </h2>
              <p className="text-sm text-ink-2">{STEP_LABEL[job.step]}</p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key={job.result.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <Result result={job.result} onAnother={onAnother} />
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

function Result({ result, onAnother }: { result: ClassificationResult; onAnother: () => void }) {
  const heading = useRef<HTMLHeadingElement>(null);
  const [cardState, setCardState] = useState<'idle' | 'busy' | 'error'>('idle');
  const { interpretation, prediction, source } = result;

  useEffect(() => {
    // move focus to the result so keyboard and screen-reader users land on it
    heading.current?.focus({ preventScroll: true });
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    document
      .getElementById('result')
      ?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  }, [result.id]);

  const download = async () => {
    setCardState('busy');
    try {
      const blob = await renderResultCard({
        displayUrl: result.displayUrl,
        interpretation,
        createdAt: result.createdAt,
        pageUrl: window.location.origin + window.location.pathname,
      });
      downloadBlob(blob, resultCardFilename(result.createdAt));
      setCardState('idle');
    } catch {
      setCardState('error');
    }
  };

  const sourceLabel =
    source.kind === 'sample' ? 'held-out test sample' : source.name ? `“${source.name}”` : 'your photo';

  return (
    <div
      className="card overflow-hidden"
      data-testid="result"
      data-predicted={interpretation.top.id}
      data-source={source.sample?.file ?? source.name}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line px-5 py-4 sm:px-6">
        <h2
          ref={heading}
          id="result-heading"
          tabIndex={-1}
          className="font-serif text-2xl font-semibold outline-none"
        >
          Result
        </h2>
        <p className="text-sm text-ink-3">
          {sourceLabel} · {prediction.backend.toUpperCase()} · {Math.round(prediction.inferenceMs)} ms
          inference
        </p>
      </div>

      <div className="grid gap-8 p-5 sm:p-6 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        {/* on phones the verdict comes first; on wide screens the image sits on the left */}
        <div className="order-2 lg:order-1">
          <CamViewer
            key={result.id}
            displayUrl={result.displayUrl}
            width={result.width}
            height={result.height}
            cam={prediction.cam}
            input={prediction.input}
            predicted={interpretation.top.id}
            alt={`The photo that was classified (${sourceLabel}).`}
          />
        </div>

        <div className="order-1 space-y-6 lg:order-2">
          <VerdictBanner interpretation={interpretation} />
          <TopClass interpretation={interpretation} />
          <LowConfidenceWarning interpretation={interpretation} />
          {source.sample && <SampleReveal result={result} />}
          <ProbabilityBars ranked={interpretation.ranked} />

          <div className="flex flex-wrap gap-3 border-t border-line pt-5">
            <button type="button" className="btn btn-primary" onClick={onAnother}>
              <RefreshIcon /> Classify another
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={download}
              disabled={cardState === 'busy'}
            >
              <DownloadIcon /> {cardState === 'busy' ? 'Preparing…' : 'Download result card'}
            </button>
          </div>
          {cardState === 'error' && (
            <p role="alert" className="text-sm text-fake-ink">
              The result card could not be created in this browser.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SampleReveal({ result }: { result: ClassificationResult }) {
  const sample = result.source.sample!;
  const correct = sample.trueClass === result.interpretation.top.id;
  return (
    <div
      data-testid="sample-reveal"
      data-correct={correct}
      className="rounded-2xl border border-line bg-surface-2 px-4 py-3 text-sm"
    >
      <p className="flex flex-wrap items-center gap-2 font-semibold text-ink">
        <span>
          Ground truth: {sample.trueClass} · {CLASS_INFO[sample.trueClass].name}
        </span>
        <span
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs ${
            correct ? 'bg-pure-soft text-pure-ink' : 'bg-fake-soft text-fake-ink'
          }`}
        >
          {correct ? <CheckIcon /> : <CrossIcon />}
          {correct ? 'Correct' : 'Misclassified'}
        </span>
      </p>
      <p className="mt-1 text-ink-2">{sample.description}</p>
      {!correct && (
        <p className="mt-1 text-ink-2">
          This is one of the model&apos;s real mistakes on the held-out test set, shown on purpose.
        </p>
      )}
    </div>
  );
}
