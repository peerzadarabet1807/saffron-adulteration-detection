/// <reference types="vite/client" />

/** Content hash of public/model (vite.config.ts); used as the IndexedDB cache key. */
declare const __MODEL_VERSION__: string;
/** Total bytes of public/model (model.json + weight shards). */
declare const __MODEL_BYTES__: number;
/** Content hash of public/gate, the input check's image encoder (IndexedDB cache key). */
declare const __GATE_VERSION__: string;
/** Total bytes of public/gate (model.json + weight shards). */
declare const __GATE_BYTES__: number;
