/** Web Worker entry: the whole TensorFlow.js pipeline runs here, off the UI thread. */
import * as Comlink from 'comlink';
import { createClassifierApi } from './classifierCore';

Comlink.expose(createClassifierApi('worker'));
