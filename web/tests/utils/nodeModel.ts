/** Load a TF.js graph model from the local filesystem (Node), without tfjs-node. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as tf from '@tensorflow/tfjs-core';
import { loadGraphModel, type GraphModel } from '@tensorflow/tfjs-converter';

export async function loadGraphModelFromDir(dir: string): Promise<GraphModel> {
  const handler: tf.io.IOHandler = {
    async load() {
      const json = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8')) as tf.io.ModelJSON;
      const paths = json.weightsManifest.flatMap((group) => group.paths);
      const buffers = paths.map((p) => readFileSync(join(dir, p)));
      const total = buffers.reduce((n, b) => n + b.byteLength, 0);
      const weightData = new Uint8Array(total);
      let offset = 0;
      for (const b of buffers) {
        weightData.set(b, offset);
        offset += b.byteLength;
      }
      return tf.io.getModelArtifactsForJSONSync(
        json,
        json.weightsManifest.flatMap((group) => group.weights),
        weightData.buffer,
      );
    },
  };
  return loadGraphModel(handler);
}
