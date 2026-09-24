import type { ClassId } from './classes';

export interface SampleInfo {
  file: string;
  thumb: string;
  trueClass: ClassId;
  kind: 'typical' | 'hardest';
  description: string;
  /** original file in the dataset (the dataset itself is not published) */
  source: string;
}

export const SAMPLES_BASE = `${import.meta.env.BASE_URL}samples/`;

export async function loadSamples(): Promise<SampleInfo[]> {
  const res = await fetch(`${SAMPLES_BASE}manifest.json`);
  if (!res.ok) throw new Error(`Could not load the sample images (HTTP ${res.status}).`);
  const manifest = (await res.json()) as { samples: SampleInfo[] };
  return manifest.samples;
}

export async function fetchSampleBlob(sample: SampleInfo): Promise<Blob> {
  const res = await fetch(`${SAMPLES_BASE}${sample.file}`);
  if (!res.ok) throw new Error(`Could not load ${sample.file} (HTTP ${res.status}).`);
  return res.blob();
}
