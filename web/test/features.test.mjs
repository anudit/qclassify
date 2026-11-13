import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { canonicalize, featurize, fnv1a, WORD_DIM, GLOBAL_DIM, MAX_WORDS } from '../dist/features.js';
import { Classifier, validateManifest, decodePackedWeights, referenceProbability } from '../dist/index.js';

const fixtures = JSON.parse(await readFile(new URL('../../data/parity.json', import.meta.url)));
const manifest = JSON.parse(await readFile(new URL('../../artifacts/web/model.json', import.meta.url)));
const file = await readFile(new URL('../../artifacts/web/weights.bin', import.meta.url));
const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
const weights = decodePackedWeights(manifest, buffer);

const dense = (pairs, n) => { const v = new Float32Array(n); for (const [i, x] of pairs) v[i] = x; return v; };
const close = (actual, expected, what) => {
  assert.equal(actual.length, expected.length, what);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6, `${what} [${i}]: ${actual[i]} != ${expected[i]}`);
  }
};

test('all Python feature contract probes match in TypeScript', () => {
  for (const row of fixtures) {
    assert.equal(canonicalize(row.input), row.canonical, `normalization: ${row.input}`);
    assert.equal(fnv1a(row.input), row.hash, `hash: ${row.input}`);
    const f = featurize(row.input, weights.lexicon);
    assert.equal(f.count, row.words.length, `word count: ${row.input}`);
    assert.ok(f.count >= 1 && f.count <= MAX_WORDS);
    row.words.forEach((w, k) => close(f.words.subarray(k * WORD_DIM, (k + 1) * WORD_DIM), dense(w, WORD_DIM),
                                      `word ${k} of ${JSON.stringify(row.input)}`));
    close(f.global, dense(row.global, GLOBAL_DIM), `global of ${JSON.stringify(row.input)}`);
  }
});

test('reject empty input, whitespace and a missing lexicon', () => {
  for (const input of ['', ' ', '\t\n', '\u0085\u001c']) assert.throws(() => featurize(input, weights.lexicon));
  assert.throws(() => featurize('Jane Doe'));
});

test('validate model contract before GPU allocation', () => {
  assert.doesNotThrow(() => validateManifest(manifest));
  for (const patch of [{ wordDim: 1 }, { hiddenDim: 1000000 }, { hiddenDim: -1 }, { globalHiddenDim: 0 },
                       { weightBytes: 1 }, { featureVersion: 'other' }, { threshold: 0.9 }, { lexBins: 8 },
                       { bits: 1 }, { labels: ['person_name', 'search_query'] }, { format: 'qclassify-webgl-v2' }]) {
    assert.throws(() => validateManifest({ ...manifest, ...patch }));
  }
  assert.throws(() => Classifier.fromWeights(manifest, new ArrayBuffer(4)), /byte length/);
  const bad = new Uint8Array(manifest.weightBytes);
  bad.fill(0x77, 0, 2048);  // valid lexicon, zero scales
  assert.throws(() => Classifier.fromWeights(manifest, bad.buffer), /scale/);
  assert.throws(() => decodePackedWeights(manifest, new Uint8Array(manifest.weightBytes).fill(0xff).buffer), /lexicon/);
});

test('packed decoder + JS reference reproduce ONNX scores across the contract probes', () => {
  for (const row of fixtures) {
    const f = featurize(row.input, weights.lexicon);
    const score = referenceProbability(weights, f.words, f.count, f.global);
    assert.ok(Math.abs(score - row.personProbability) < 2e-6, `packed score: ${row.input}: ${score} vs ${row.personProbability}`);
  }
});
