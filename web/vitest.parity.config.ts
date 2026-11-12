import { defineConfig } from 'vitest/config';

// Parity test: Node environment, local build artefacts, long timeouts. Not part of `npm test` / CI.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/parity.node.test.ts', 'tests/parity.gate.node.test.ts'],
    testTimeout: 600_000,
    hookTimeout: 120_000,
    pool: 'forks',
  },
});
