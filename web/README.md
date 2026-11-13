# @omnid/qclassify

A tiny person-name versus search-query neural classifier. Zero runtime
dependencies; WebGL 2 shaders run the whole network. Packed weights ship in the
package. No model service, ONNX runtime, WASM binary, or WebGPU API is needed.

```ts
import { Classifier } from '@omnid/qclassify';

const classifier = await Classifier.create();
const result = classifier.predict('Aarav Sharma');
// { input, label: 'person_name', personProbability, searchProbability }

const results = classifier.predictBatch([
  'Aarav Sharma',
  'open source inference engines',
  'Ada Lovelace biography',
]);
classifier.dispose();
```

Each word is a 2048-feature row (up to 16 words) plus a 1044-feature global row; `create()` imports the bundled model module and decodes its packed weights once.
It does not fetch external model assets. Keep one classifier for the lifetime of
the search box. `predict` and `predictBatch` are synchronous and return model
scores, not calibrated guarantees. The boundary is 0.5. Input is lowercased internally, so casing never changes a score. Empty input throws; an
empty batch returns `[]`. Inputs are clipped to 256 normalized Unicode code points.

## HHQ / Next.js

Use it from a client component or effect; call `create()` after mounting and
`dispose()` when the component unmounts. Avoid initializing WebGL during SSR.

```ts
const prediction = classifier.predict(query);
const intent = prediction.label === 'person_name' ? 'person' : 'topic';
```

This maps to HHQ's `intent: 'person' | 'topic'` contract. Use real traffic to
choose whether uncertain scores should trigger both search paths. Scores are uncalibrated and the training data is synthetic, so check
real traffic for mononyms and lowercase input.

## Custom weights

For retrained models, host both exports and provide their URLs. The loader checks
the SHA-256 digest, manifest, byte length, scales, and finite weights.

```ts
const classifier = await Classifier.create({
  manifestUrl: '/qclassify/model.json',
  weightsUrl: '/qclassify/weights.bin',
});
```

This fetch-based path requires WebCrypto (HTTPS or localhost). To use already
loaded trusted assets, call `Classifier.fromWeights(manifest, arrayBuffer)`;
this validates structure and values but leaves checksum verification to you.

`@omnid/qclassify/features` exports the identical TypeScript feature extractor. Since 0.3.0 it is
`featurize(text, lexicon)`, returning `{ words, count, global }`; get the lexicon from
`decodePackedWeights(manifest, buffer).lexicon` (or `decodeLexicon(bytes)` on the first 2,048 blob bytes).
`@omnid/qclassify/weights` exports the packed-weight decoder and manifest validator.
Neither subpath initializes a GPU or loads the bundled model.

## Runtime details

WebGL 2 is required. Float weights and features use core `R32F` textures; the
intermediate and output floats are encoded losslessly into core `RGBA8` targets.
No floating-point render-target extension is needed. The library owns its canvas
and context; an optional `canvas` argument must not be shared with another
renderer. Both `HTMLCanvasElement` and `OffscreenCanvas` are supported.

Batches of up to 256 inputs (and up to 4,096 words) use three draw calls (per-word hidden layer, global
hidden layer, then word scoring + soft-min/mean pooling + output) and one `readPixels` call. Larger
batches are chunked. Capacity grows and GPU buffers/textures are reused. Context
loss and calls after disposal throw; create another classifier after context
loss. For large jobs, run this API inside a worker with `OffscreenCanvas`.

The weight blob (`qclassify-webgl-v3`) is: the 4,096-slot 4-bit name lexicon (2,048 bytes), the word layer and
global layer in symmetric groupwise low-bit quantization (float16 scale per 32 weights, packed codes), then a
float32 tail of biases, output weights, pooling weights and shape weights. The exporter picks the widest
3/4/5/6/8-bit width that fits the 25,000-byte budget and keeps validation accuracy/loss within tolerance
(5-bit, 23,456 bytes). The test split is evaluated after selection. The embedded hex representation is
chosen by measured Brotli delivery size. See the root `artifacts/web/packing.json` and bundle reports for exact
results rather than counting only weight bytes.

## Build and package

From this directory:

```sh
npm ci
npm test
npm run size
npm pack
```

Only compiled ESM, declarations, README, and license enter the tarball. The
runtime has no dependencies or peers; TypeScript and esbuild are dev tools.
`npm run size` measures reachable minified ESM including the model, raw, gzip,
and Brotli sizes. Compression settings are recorded in the script.

For a local HHQ dependency, build first, then install the resulting tarball in
that application's repository. Publish from here with `npm publish --access public`
when you are ready to release the `@omnid/qclassify` name.

The authored dataset, held-out metrics, and model limitations are documented in
the root repository. Short ambiguous strings cannot reveal actual user intent.
