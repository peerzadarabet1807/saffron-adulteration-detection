/**
 * Every sentence the app shows about the input check. The rates come from results/input_check/summary.json
 * (scripts/build_input_check.py, synced to src/data/); nothing here hard-codes a number. The check only asks
 * whether the photo looks like the training photos; no sentence here may hint at pure or counterfeit.
 */
import summary from '../data/input_check_summary.json';
import { formatPercent } from './prediction';

const p1 = (x: number) => formatPercent(x, 1);

export const INPUT_CHECK_SUMMARY = summary;

export const INPUT_CHECK_COPY = {
  heading: "This doesn't look like a single saffron thread on paper.",
  why: 'The model only knows photos of one thread on light paper, so its answer for this photo would not mean anything.',
  tip: 'Lay one thread on plain white or light paper and photograph it in even daylight, with the phone parallel to the paper.',
  retry: 'Try another photo',
  tips: 'Photo tips',
  anyway: 'Analyse anyway',
  uncheckedLabel:
    'Unchecked input: this photo did not look like a single saffron thread on paper, so treat the result with extra caution.',
} as const;

/** Shown under the notice: the check's own error rates, so nobody takes it as a guarantee. */
export function safeguardNote(): string {
  return (
    'The check is a safeguard, not a guarantee. In testing it turned away ' +
    `${p1(summary.false_rejection.test.rate)} of held-out saffron-thread photos and let through ` +
    `${p1(summary.false_acceptance.near.rate)} of look-alike photos such as red thread or chili threads. ` +
    'If your photo does show one thread on paper, use Analyse anyway.'
  );
}

/** Text for the polite live region when the check turns a photo away. */
export function rejectionAnnouncement(): string {
  return `${INPUT_CHECK_COPY.heading} ${INPUT_CHECK_COPY.why} No result is shown. Options: ${INPUT_CHECK_COPY.retry}, ${INPUT_CHECK_COPY.tips} or ${INPUT_CHECK_COPY.anyway}.`;
}
