import { ChevronIcon } from './icons';

interface Props {
  open: boolean;
  onToggle: (open: boolean) => void;
}

const TIPS = [
  { title: 'One thread', text: 'A single saffron thread, roughly centred in the frame.' },
  { title: 'Plain light paper', text: 'White or light paper underneath; nothing else in the shot.' },
  { title: 'Even daylight', text: 'Diffuse light, no flash, no heavy shadows or coloured lamps.' },
  {
    title: 'Fill the frame sensibly',
    text: 'Hold the phone parallel to the paper, 15–25 cm away, in focus.',
  },
];

/** Shown open before the first result; collapsible afterwards. */
export function PhotoGuidelines({ open, onToggle }: Props) {
  return (
    <details
      id="photo-guidelines"
      className="group card scroll-mt-24 overflow-hidden"
      open={open}
      onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="flex cursor-pointer list-none items-center gap-3 px-5 py-4 [&::-webkit-details-marker]:hidden">
        <span className="eyebrow">Photo guidelines</span>
        <span className="text-sm text-ink-3">how the training photos were taken</span>
        <ChevronIcon className="ml-auto text-lg text-ink-3 transition-transform group-open:rotate-180" />
      </summary>
      <div className="border-t border-line px-5 pb-5 pt-4">
        <ol className="grid gap-3 sm:grid-cols-2">
          {TIPS.map((tip, i) => (
            <li key={tip.title} className="flex gap-3">
              <span className="font-serif text-lg leading-6 text-gold-ink" aria-hidden="true">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="text-sm">
                <span className="block font-semibold text-ink">{tip.title}</span>
                <span className="text-ink-2">{tip.text}</span>
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-4 rounded-xl bg-surface-2 px-4 py-3 text-sm text-ink-2">
          The model was trained <strong className="text-ink">only</strong> on photos like these: one thread on
          light paper, shot with one phone. Photos of mixed samples, jars, powder or other backgrounds are
          outside what it has learned, and it will still pick one of its five classes.
        </p>
      </div>
    </details>
  );
}
