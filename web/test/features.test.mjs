import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { canonicalize, featurize, fnv1a, INPUT_DIM } from '../dist/features.js';
import { Classifier, validateManifest } from '../dist/index.js';

const fixtures = JSON.parse(await readFile(new URL('../../data/parity.json', import.meta.url)));
const manifest = JSON.parse(await readFile(new URL('../../artifacts/web/model.json', import.meta.url)));

test('all Python feature contract probes match in TypeScript', () => {
  for (const row of fixtures) {
    assert.equal(canonicalize(row.input), row.canonical, `normalization: ${row.input}`);
    assert.equal(fnv1a(row.input), row.hash, `hash: ${row.input}`);
    const expected = new Float32Array(INPUT_DIM);
    for (const [i, value] of row.features) expected[i] = value;
    const actual = featurize(row.input);
    assert.equal(actual.length, expected.length);
    for (let i = 0; i < INPUT_DIM; i++) {
      assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6,
                `feature ${i}: ${JSON.stringify(row.input)}: ${actual[i]} != ${expected[i]}`);
    }
  }
});

test('reject empty input and whitespace', () => {
  for (const input of ['', ' ', '\t\n', '\u0085\u001c']) assert.throws(() => featurize(input));
});

test('validate model contract before GPU allocation', () => {
  assert.doesNotThrow(() => validateManifest(manifest));
  for (const patch of [{ inputDim: 1 }, { hiddenDim: 1000000 }, { hiddenDim: -1 },
                       { weightBytes: 1 }, { featureVersion: 'other' }, { threshold: 0.9 },
                       { bits: 1 }, { labels: ['person_name', 'search_query'] }]) {
    assert.throws(() => validateManifest({ ...manifest, ...patch }));
  }
  assert.throws(() => Classifier.fromWeights(manifest, new ArrayBuffer(4)), /byte length/);
  const nanWeights = new ArrayBuffer(manifest.weightBytes);
  assert.throws(() => Classifier.fromWeights(manifest, nanWeights), /scale/);
});

test('packed decoder reproduces ONNX scores across the contract probes', async () => {
  const { decodePackedWeights } = await import('../dist/weights.js');
  const file = await readFile(new URL('../../artifacts/web/weights.bin', import.meta.url));
  const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const weights = decodePackedWeights(manifest, buffer);
  const hidden = manifest.hiddenDim;
  const b1 = INPUT_DIM * hidden;
  const w2 = b1 + hidden;
  for (const row of fixtures) {
    const x = featurize(row.input);
    let logit = weights[w2 + hidden];
    for (let j = 0; j < hidden; j++) {
      let value = weights[b1 + j];
      for (let i = 0; i < INPUT_DIM; i++) value += weights[j * INPUT_DIM + i] * x[i];
      logit += Math.max(value, 0) * weights[w2 + j];
    }
    const score = 1 / (1 + Math.exp(-logit));
    assert.ok(Math.abs(score - row.personProbability) < 2e-6, `packed score: ${row.input}`);
  }
});
