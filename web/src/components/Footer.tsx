import { REPO_URL } from '../lib/links';

export function Footer() {
  return (
    <footer className="mt-20 border-t border-line pb-[env(safe-area-inset-bottom)]">
      <div className="mx-auto max-w-6xl space-y-4 px-4 py-10 text-sm text-ink-3 sm:px-6">
        <p className="font-semibold text-ink" data-testid="disclaimer">
          Research prototype. Not a certified food-safety or laboratory test.
        </p>
        <p className="max-w-3xl">
          Built by Peerzada Mohammad Rabet from his B.Tech thesis in Agricultural and Food Engineering at IIT
          Kharagpur, supervised by Dr. Mamoni Banerjee (RMSOEE). Photos taken by the author from saffron
          collected in Pampore, Kashmir.
        </p>
        <p>
          <a href={REPO_URL} className="font-medium text-crimson-ink underline underline-offset-2">
            Source code
          </a>{' '}
          (MIT) · TensorFlow.js {`·`} Model version <code className="text-xs">{__MODEL_VERSION__}</code>
        </p>
      </div>
    </footer>
  );
}
