# @omnid/qclassify

A tiny person-name versus search-query neural classifier. Zero runtime
dependencies; WebGL 2 shaders run both neural layers. Packed weights ship in the
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

`create()` imports the bundled model module and decodes its packed weights once.
It does not fetch external model assets. Keep one classifier for the lifetime of
the search box. `predict` and `predictBatch` are synchronous and return model
scores, not calibrated guarantees. The boundary is 0.5. Empty input throws; an
empty batch returns `[]`. Inputs are clipped to 256 normalized Unicode code points.

## HHQ / Next.js

Use it from a client component or effect; call `create()` after mounting and
`dispose()` when the component unmounts. Avoid initializing WebGL during SSR.

```ts
const prediction = classifier.predict(query);
const intent = prediction.label === 'person_name' ? 'person' : 'topic';
```

This maps to HHQ's `intent: 'person' | 'topic'` contract. Use real traffic to
choose whether uncertain scores should trigger both search paths. This package
has not been installed into HHQ or published to npm yet.

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

`@omnid/qclassify/features` exports the identical TypeScript feature extractor.
`@omnid/qclassify/weights` exports the packed-weight decoder and manifest validator.
Neither subpath initializes a GPU or loads the bundled model.

## Runtime details

WebGL 2 is required. Float weights and features use core `R32F` textures; the
intermediate and output floats are encoded losslessly into core `RGBA8` targets.
No floating-point render-target extension is needed. The library owns its canvas
and context; an optional `canvas` argument must not be shared with another
renderer. Both `HTMLCanvasElement` and `OffscreenCanvas` are supported.

Batches of up to 256 inputs use two draw calls and one `readPixels` call. Larger
batches are chunked. Capacity grows and GPU buffers/textures are reused. Context
loss and calls after disposal throw; create another classifier after context
loss. For large jobs, run this API inside a worker with `OffscreenCanvas`.

The weights use symmetric groupwise low-bit quantization: float16 scales per 32
weights, packed codes, float32 hidden/output biases, and float16 output weights.
The exporter compares 2/3/4/5/6/8-bit formats and selects the smallest that keeps
validation accuracy within 0.2 percentage points and loss within 0.01 of FP32.
The test split is evaluated after selection. The embedded hex representation is
chosen by measured Brotli delivery size; larger uncompressed source can compress
better than base64. See the root `artifacts/web/packing.json` and bundle reports for exact
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
