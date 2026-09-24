import { LazyMotion, MotionConfig, domAnimation } from 'framer-motion';
import { useCallback, useState } from 'react';
import { AboutModel } from './components/AboutModel';
import { Footer } from './components/Footer';
import { Header } from './components/Header';
import { ModelStatus } from './components/ModelStatus';
import { PhotoGuidelines } from './components/PhotoGuidelines';
import { ResultPanel } from './components/ResultPanel';
import { SampleGallery, type Revealed } from './components/SampleGallery';
import { UploadCard } from './components/UploadCard';
import { useClassifier, type JobState, type SourceKind } from './hooks/useClassifier';
import { useSamples } from './hooks/useSamples';
import { useTheme } from './hooks/useTheme';
import { CLASS_INFO } from './lib/classes';
import { formatPercent } from './lib/prediction';
import { fetchSampleBlob, type SampleInfo } from './lib/samples';

/** Text for the polite live region, so screen readers hear progress and the outcome. */
function announcement(job: JobState): string {
  switch (job.status) {
    case 'working':
      return 'Analysing the photo…';
    case 'error':
      return job.message;
    case 'done': {
      const { verdict, top, lowConfidence } = job.result.interpretation;
      return `Result: likely ${verdict}. Most likely class ${top.id}, ${CLASS_INFO[top.id].name}, ${formatPercent(top.probability)}.${
        lowConfidence ? ' Low confidence.' : ''
      }`;
    }
    default:
      return '';
  }
}

export default function App() {
  const { theme, toggle } = useTheme();
  const { model, job, classify, reset, retryModel } = useClassifier();
  const samples = useSamples();
  const [guidelinesOpen, setGuidelinesOpen] = useState(true);
  const [revealed, setRevealed] = useState<Record<string, Revealed>>({});
  const [activeSample, setActiveSample] = useState<string>();

  const busy = job.status === 'working';
  const lastInferenceMs = job.status === 'done' ? job.result.prediction.inferenceMs : undefined;

  const run = useCallback(
    async (input: Blob | (() => Promise<Blob>), kind: SourceKind, name: string, sample?: SampleInfo) => {
      setGuidelinesOpen(false);
      setActiveSample(sample?.file);
      const result = await classify(input, { kind, name, sample });
      // reveal a sample's true label once it has been classified
      if (result && sample) {
        setRevealed((r) => ({ ...r, [sample.file]: { predicted: result.interpretation.top.id } }));
      }
    },
    [classify],
  );

  const onFile = useCallback((file: File, kind: SourceKind) => void run(file, kind, file.name), [run]);

  const onSample = useCallback(
    (sample: SampleInfo, index: number) =>
      void run(() => fetchSampleBlob(sample), 'sample', `Sample ${index + 1}`, sample),
    [run],
  );

  const onAnother = useCallback(() => {
    reset();
    setActiveSample(undefined);
    const upload = document.getElementById('upload');
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    upload?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
    upload?.querySelector<HTMLButtonElement>('button')?.focus({ preventScroll: true });
  }, [reset]);

  return (
    <LazyMotion features={domAnimation} strict>
      <MotionConfig reducedMotion="user">
        <a
          href="#main"
          className="sr-only z-50 rounded-full bg-crimson px-4 py-2 text-on-crimson focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
        >
          Skip to content
        </a>
        <div id="top" />
        <Header theme={theme} onToggleTheme={toggle} />

        <main id="main" className="mx-auto max-w-6xl space-y-16 px-4 pt-10 sm:px-6 sm:pt-14">
          <section id="classify" aria-labelledby="hero-heading" className="scroll-mt-20 space-y-8">
            <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-12">
              <div className="space-y-6 lg:pt-4">
                <p className="eyebrow">01 · Classify a thread</p>
                <h1
                  id="hero-heading"
                  className="font-serif text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl"
                >
                  Is that saffron <em className="font-medium text-crimson-ink">real</em>?
                </h1>
                <p className="max-w-xl text-lg text-ink-2">
                  Photograph a single thread and a ResNet50V2 model, running entirely in your browser, tells
                  you which of five classes it most resembles: two kinds of pure saffron or three common
                  counterfeits, with a Grad-CAM map of where it looked.
                </p>
                <PhotoGuidelines open={guidelinesOpen} onToggle={setGuidelinesOpen} />
              </div>
              <UploadCard
                busy={busy}
                error={job.status === 'error' ? job.message : undefined}
                onFile={onFile}
                status={<ModelStatus model={model} lastInferenceMs={lastInferenceMs} onRetry={retryModel} />}
              />
            </div>

            <ResultPanel job={job} onAnother={onAnother} />

            <SampleGallery
              samples={samples}
              revealed={revealed}
              activeFile={activeSample}
              busy={busy}
              onPick={onSample}
            />
          </section>

          <AboutModel />
        </main>

        <Footer />
        <p aria-live="polite" className="sr-only" data-testid="live-region">
          {announcement(job)}
        </p>
      </MotionConfig>
    </LazyMotion>
  );
}
