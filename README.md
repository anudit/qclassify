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

[The dataset](data/examples.json) contains **5,909 explicitly authored seed
examples**, with `input` and `output` fields: **2,915 `person_name`** and **2,994
`search_query`**. The original text sources, split manifest, and generated
training-only casing variants are all under `data/`. See [data/README.md](data/README.md)
for authorship, label policy, HHQ context, and limitations. No scraped dataset or
template expansion is needed to train the seed corpus.

The 0.3.0 network scores **each word separately** and pools the word scores, so one
query-looking word ("biography", "papers") can flip an input to search while a
name-looking word cannot be outvoted by its own n-gram fragments:

- **Word branch:** each of up to 16 words becomes a 2048-wide vector (2,028 signed
  2-4-gram hash bins of `^word$`, 10 shape/intent flags, and a name-vs-word
  **lexicon** value). A shared `2048 → 13 ReLU → 1` layer scores every word.
- **Pooling:** soft-min (`-log Σ exp(-s)`) plus mean of the word scores, with learned weights.
- **Global branch:** a 1024-bin hash of the whole text plus 20 shape features,
  `1044 → 4 ReLU → 1`, plus a linear term on the shape features. Then sigmoid.

The lexicon is 4,096 hashed 4-bit slots holding `clip(name_zipf - word_zipf, ±3.5)`:
how much more common a token is as a given/family name than as an ordinary word.
It is built by `scripts/build_lexicon.py` from US Census 2010 surnames, SSA baby
names (via the hadley/data-baby-names mirror), Wikidata given/family-name labels,
and [wordfreq](https://github.com/rspeer/wordfreq) word frequencies. wordfreq data
is **CC-BY-SA 4.0** (Robyn Speer et al.); the derived `src/qclassify/lexicon.bin` is
shipped under that attribution. There is no tokenizer download or vocabulary file;
the lexicon travels inside the packed weight blob. The versioned feature extractor
(`nfkc-lower-words2to4-lex4096-global1024-v5`) is shared between Python and
TypeScript and truncates to 256 normalized code points.

| Artifact | Size / held-out result |
| --- | --- |
| Full FP32 ONNX, graph and weights included | **124,767 bytes** (limit 134,000) |
| All FP32 learned weights | **123,432 bytes**, 30,858 parameters |
| Packed browser weights, lexicon included | **23,456 bytes**, 5-bit groupwise (budget 25,000) |
| Original model test accuracy | **90.94%**, 2,428 test examples (authored + mined) |
| Packed model test accuracy | **90.98%**, same test split |
| Complete reachable minified browser ESM | **64,908 bytes raw** |
| ESM including embedded weights, compressed per module | **24,031 bytes Brotli / 28,867 gzip** |

Exact results live in `artifacts/metrics.json`, `artifacts/web/packing.json`, and
`artifacts/bundle-size.json`. The bundle measure includes preprocessing, decoding,
shaders, GPU lifecycle, and dynamically imported model code. It excludes HTTP
headers and consuming-framework overhead. Rebundling may change compressed size.

The exporter enforces **the complete ONNX file < 134,000 bytes**. Packing picks the
widest tested 3/4/5/6/8-bit width that fits `--max-packed-bytes` (25,000) and meets
validation accuracy/loss tolerances, then evaluates the selected model on test.
It writes both a packed binary and an equivalent dequantized ONNX. The latter
is useful for parity checks and has the same packed-model predictions; the
compact codes are what ship to the browser.

These are synthetic-corpus metrics. Names and search topics overlap, and a bare
name does not reveal actual intent. Random splits can share related names,
products, and phrasing. Check independent real HHQ queries before relying on
routing, especially mononyms, handles, prefixes, and languages underrepresented
in the seed data. Scores are uncalibrated. The feature extractor lowercases all input
before hashing and shape features, so "Solar Energy" and "solar energy" score identically.

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
It is implemented here for WebGL 2 using three fragment-shader passes and one
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

## 0.3.0: 2x architecture

0.2.x scored "Dark Matter" 0.75 and "Ada Lovelace" 0.29 for person. 0.3.0 spends up to 2x the 0.2.1 size
(FP32 ONNX 1.95x, packed weights 1.95x, Brotli bundle 1.83x, npm tarball 1.64x) as follows. Offline comparisons
on the same data, all at about 33k parameters: char hash bins 1024 -> 2028 (inputs 2048, the texture width WebGL 2
guarantees) was the clear win; hidden 16 -> 32 on 1024 bins, whole-word hash blocks (overfit in ~20 epochs),
2-layer nets and 2-5-grams were not. Training changes: hand-authored rows weigh 3x mined rows
(`--clean-weight 3`), early stopping follows the clean validation rows only (`--mined-val-weight 0`),
L2 3e-4, no casing augmentation. Packing now spends a byte budget (`--max-packed-bytes 23168`) instead of the
smallest width within 0.2 points: 5-bit, 0.7 points of validation accuracy under FP32, practically lossless on the
probes. Held-out probes (`scripts/compare.py`): topic phrases 0.776 -> 0.939, unseen famous names 0.964, one-word
and tech 1.0. Still wrong: "Ada Lovelace" (0.27). Seed-to-seed spread is a few points, so treat single results loosely.

### 0.3.0 final: per-word pooling + name lexicon

The 2x bag-of-n-grams model above still scored "Ada Lovelace" 0.27: the corpus had more "Ada Lovelace
biography"-style queries than the bare name, and fragments of "lovelace" (love, lace) are query-heavy. Offline
comparisons under a 25 KB packed budget: per-word scoring with soft-min pooling was the clear win, the hashed
name/word lexicon helped topics, position-tagged n-grams hurt. The shipped model adds 3,000 real one- and
multi-word names (local only, see data/README.md). Init seed is chosen on clean validation loss.

Held-out probes (`scripts/compare.py`): topic phrases 0.949, unseen famous names 0.893 (was 0.964 for the
pre-release 2x model), hand one-word/tech 0.917, authored multi-word names 1.0 / queries 0.931.
"ada lovelace" 0.97, "solar energy" 0.02, "dark matter" 0.15. Known misses: "steve jobs" 0.01 ("jobs" is a
search-intent word) and "virginia woolf" 0.16.
