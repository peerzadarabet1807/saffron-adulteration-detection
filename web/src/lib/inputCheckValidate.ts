/**
 * Boundary check for input check results coming back from the worker. Kept apart from inputCheck.ts so the page
 * bundle does not pull in the prototype vectors (models/input_check.json), which only the worker needs.
 */
import type { InputCheckResult } from './inputCheck';

/** Throws if a result is malformed or its decision does not match its score (the worker boundary). */
export function validateInputCheck(c: InputCheckResult): InputCheckResult {
  const bad = (what: string): never => {
    throw new Error(`The input check returned an invalid result (${what}).`);
  };
  if (typeof c.score !== 'number' || !Number.isFinite(c.score) || c.score < -1.0001 || c.score > 1.0001) {
    bad('score outside [-1, 1]');
  }
  if (typeof c.threshold !== 'number' || !Number.isFinite(c.threshold)) bad('threshold');
  if (c.passed !== c.score >= c.threshold) bad('decision does not match its score');
  if (typeof c.version !== 'string' || !c.version) bad('version');
  return c;
}
