# qclassify

A tiny neural classifier for **bare personal name** versus **search/topic input**.
Python 3.12 + uv for MPS training; ONNX export; a zero-runtime-dependency WebGL 2
TypeScript library at `web`, packaged as **`@omnid/qclassify`**.

```sh
uv sync --python 3.12
uv run qclassify-train --device mps
uv run qclassify-export-web
uv run python scripts/make_parity_fixtures.py
uv run qclassify-predict "Aarav Sharma" "Hugging Face" "Ada Lovelace biography"
```

`qclassify-train` defaults to MPS and fails explicitly if Metal is unavailable.
For a non-Mac machine, select `--device cpu`. NumPy computes preprocessing and
export references; PyTorch performs training and backpropagation on the selected
device. Best-checkpoint selection and early stopping use validation loss only.
MPS can differ slightly across hardware/driver versions despite fixed seeds.

## Data and model

[The dataset](data/examples.json) contains **5,226 explicitly authored seed
examples**, with `input` and `output` fields: **2,550 `person_name`** and **2,676
`search_query`**. The original text sources, split manifest, and generated
training-only casing variants are all under `data/`. See [data/README.md](data/README.md)
for authorship, label policy, HHQ context, and limitations. No scraped dataset or
template expansion is needed to train the seed corpus.

The network is `1044 → 8 ReLU → 1 sigmoid`: 1,024 signed character n-gram hash
features, 16 shape features, and four small fixed English intent hints. The
hints indicate request starts, artifact/search words such as `biography`,
technology terms, and URLs; they are learned inputs, not a routing override.
There is no learned vocabulary, tokenizer download, embedding table, or external
weight file for the ONNX graph. The versioned feature extractor is shared
between Python and TypeScript and truncates to 256 normalized code points.

| Artifact | Size / held-out result |
| --- | --- |
| Full FP32 ONNX, graph and weights included | **33,864 bytes** |
| All FP32 learned weights | **33,476 bytes**, 8,369 parameters |
| Packed browser weights | **4,750 bytes**, 4-bit groupwise |
| Original model test accuracy | **91.78%**, 523 authored test examples |
| Packed model test accuracy | **91.20%**, same test split |
| Complete reachable minified browser ESM | **22,798 bytes raw** |
| ESM including embedded weights, compressed per module | **8,428 bytes Brotli / 10,350 bytes gzip** |

Exact results live in `artifacts/metrics.json`, `artifacts/web/packing.json`, and
`artifacts/bundle-size.json`. The bundle measure includes preprocessing, decoding,
shaders, GPU lifecycle, and dynamically imported model code. It excludes HTTP
headers and consuming-framework overhead. Rebundling may change compressed size.

The exporter enforces **the complete ONNX file < 100,000 bytes**, stricter than
100 KiB. Packing selects the smallest tested 2/3/4/5/6/8-bit representation meeting
validation accuracy/loss tolerances, then evaluates the selected model on test.
It writes both a packed binary and an equivalent dequantized ONNX. The latter
is useful for parity checks and has the same packed-model predictions; the
compact codes are what ship to the browser.

These are synthetic-corpus metrics. Names and search topics overlap, and a bare
name does not reveal actual intent. Random splits can share related names,
products, and phrasing. Check independent real HHQ queries before relying on
routing, especially mononyms, handles, prefixes, and languages underrepresented
in the seed data. Scores are uncalibrated. Case augmentation does not make the
classifier case-invariant.

## Browser package

```sh
cd web
npm ci
npm test
npm run size
npm pack
```

```ts
import { Classifier } from '@omnid/qclassify';
const classifier = await Classifier.create();
const result = classifier.predict('Aarav Sharma');
const intent = result.label === 'person_name' ? 'person' : 'topic';
classifier.dispose();
```

Create the classifier after mounting a client component in HHQ. Reuse it across
inputs. The model is embedded in the npm library; creating it does not download
an external model. ESM, declarations and license are included, with zero runtime
dependencies. [web/README.md](web/README.md) documents batches, custom models,
workers, context loss, and package preparation. No HHQ files were modified.

The runtime was informed by lexer's packed-weight/resource lifecycle approach
in `/Users/anudit/Documents/GitHub/lexer/lex/src/runtime.js` and `weights-codec.js`.
It is implemented here for WebGL 2 using two fragment-shader passes and one
readback per batch, with no WebGPU dependency.

## Verification

```sh
uv run pytest -q
cd web && npm test
# From the repository root, after building web:
uv run python scripts/check_browser.py
```

Python checks dataset/split integrity, actual learning, ONNX size, and NumPy/ONNX
equivalence. Node checks Unicode preprocessing and packed decoding against frozen
Python/ONNX probes. The headless Chrome check executes the actual shaders,
compares scores to packed ONNX, and exercises batching, growth, repeated use,
embedded/fetched models and disposal. It records its GPU renderer and numerical
error in `artifacts/browser-validation.json`; Chrome must be installed. In a
restricted runner, Metal access and binding a localhost port may require running
outside its sandbox. For CI without a GPU, `--software` explicitly uses SwiftShader.

Useful references: [uv Python versions](https://docs.astral.sh/uv/concepts/python-versions/),
[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), and
[WebGL 2 specification](https://registry.khronos.org/webgl/specs/latest/2.0/).
