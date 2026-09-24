import type { Theme } from '../hooks/useTheme';
import { REPO_URL } from '../lib/links';
import { GitHubIcon, MoonIcon, SaffronMark, SunIcon } from './icons';

interface Props {
  theme: Theme;
  onToggleTheme: () => void;
}

export function Header({ theme, onToggleTheme }: Props) {
  const next = theme === 'dark' ? 'light' : 'dark';
  return (
    <header className="sticky top-0 z-30 border-b border-line/80 bg-paper/85 pt-[env(safe-area-inset-top)] backdrop-blur supports-[backdrop-filter]:bg-paper/70">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <a href="#top" className="flex items-center gap-2.5 rounded-md">
          <SaffronMark className="text-[2rem]" />
          <span className="flex flex-col leading-none">
            <span className="font-serif text-lg font-semibold tracking-tight">Saffron Check</span>
            <span className="hidden text-xs text-ink-3 sm:block">
              adulteration detection, in your browser
            </span>
          </span>
        </a>
        <nav aria-label="Main" className="ml-auto flex items-center gap-1 text-sm font-medium">
          <a
            href="#classify"
            className="hidden rounded-full px-3 py-2 text-ink-2 hover:text-ink md:inline-block"
          >
            Classify
          </a>
          <a
            href="#about"
            className="hidden rounded-full px-3 py-2 text-ink-2 hover:text-ink md:inline-block"
          >
            About the model
          </a>
          <a
            href="#limitations"
            className="hidden rounded-full px-3 py-2 text-ink-2 hover:text-ink md:inline-block"
          >
            Limitations
          </a>
          <a
            href={REPO_URL}
            className="grid size-10 place-items-center rounded-full text-ink-2 hover:bg-surface-2 hover:text-ink"
            aria-label="Source code on GitHub"
          >
            <GitHubIcon className="text-xl" />
          </a>
          <button
            type="button"
            onClick={onToggleTheme}
            className="grid size-10 place-items-center rounded-full text-ink-2 hover:bg-surface-2 hover:text-ink"
            aria-label={`Switch to ${next} theme`}
            title={`Switch to ${next} theme`}
          >
            {theme === 'dark' ? <SunIcon className="text-xl" /> : <MoonIcon className="text-xl" />}
          </button>
        </nav>
      </div>
    </header>
  );
}
