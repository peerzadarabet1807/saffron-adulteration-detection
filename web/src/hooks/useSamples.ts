import { useEffect, useState } from 'react';
import { loadSamples, type SampleInfo } from '../lib/samples';

export type SamplesState =
  { status: 'loading' } | { status: 'ready'; samples: SampleInfo[] } | { status: 'error'; message: string };

export function useSamples(): SamplesState {
  const [state, setState] = useState<SamplesState>({ status: 'loading' });
  useEffect(() => {
    let alive = true;
    loadSamples().then(
      (samples) => alive && setState({ status: 'ready', samples }),
      (e: unknown) =>
        alive && setState({ status: 'error', message: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}
