import {
  FEATURE_VERSION, GLOBAL_BINS, GLOBAL_DIM, LEX_BINS, MAX_CHARS, MAX_WORDS, SHAPE_FEATURES, WORD_BINS, WORD_DIM,
  decodeLexicon, type Lexicon,
} from "./features.js";

export interface ModelManifest {
  format: "qclassify-webgl-v3";
  featureVersion: string;
  wordDim: number;
  wordBins: number;
  globalDim: number;
  globalBins: number;
  lexBins: number;
  maxWords: number;
  hiddenDim: number;
  globalHiddenDim: number;
  maxChars: number;
  weightBytes: number;
  bits: number;
  groupSize: number;
  sha256: string;
  layout: string[];
  labels: string[];
  threshold: number;
}

/** Decoded model. w1 is hidden-major (row j = unit j's WORD_DIM weights), w3 likewise over GLOBAL_DIM. */
export interface ModelWeights {
  lexicon: Lexicon;
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: number;
  mix: [number, number];
  w3: Float32Array;
  b3: Float32Array;
  w4: Float32Array;
  vs: Float32Array;
  c: number;
}

const MAX_BYTES = 134_000;
const matrixBytes = (count: number, bits: number, group: number) =>
  Math.ceil(count / group) * 2 + Math.ceil(count * bits / 8);
const tailFloats = (h: number, g: number) => 2 * h + 1 + 2 + 2 * g + SHAPE_FEATURES + 1;

export function validateManifest(value: unknown): asserts value is ModelManifest {
  if (!value || typeof value !== "object") throw new Error("Invalid model manifest.");
  const m = value as ModelManifest;
  const sane = (n: unknown, max: number) => Number.isSafeInteger(n) && (n as number) >= 1 && (n as number) <= max;
  const ok = m.format === "qclassify-webgl-v3" && m.featureVersion === FEATURE_VERSION &&
    m.wordDim === WORD_DIM && m.wordBins === WORD_BINS && m.globalDim === GLOBAL_DIM &&
    m.globalBins === GLOBAL_BINS && m.lexBins === LEX_BINS && m.maxWords === MAX_WORDS && m.maxChars === MAX_CHARS &&
    sane(m.hiddenDim, 64) && sane(m.globalHiddenDim, 64) &&
    Number.isSafeInteger(m.bits) && m.bits >= 2 && m.bits <= 8 && m.groupSize === 32 && m.threshold === 0.5 &&
    JSON.stringify(m.labels) === '["search_query","person_name"]' &&
    JSON.stringify(m.layout) === '["lexicon-u4","w1-grouped","w3-grouped","tail-f32"]' &&
    typeof m.sha256 === "string" && /^[0-9a-f]{64}$/.test(m.sha256);
  if (!ok) throw new Error("Incompatible or malformed qclassify model manifest.");
  const expected = LEX_BINS / 2 + matrixBytes(WORD_DIM * m.hiddenDim, m.bits, 32) +
    matrixBytes(GLOBAL_DIM * m.globalHiddenDim, m.bits, 32) + tailFloats(m.hiddenDim, m.globalHiddenDim) * 4;
  if (m.weightBytes !== expected || expected >= MAX_BYTES) {
    throw new Error("Incompatible or malformed qclassify model manifest.");
  }
}

function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31;
  const mantissa = bits & 1023;
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24);
}

function decodeMatrix(bytes: Uint8Array, view: DataView, offset: number, count: number, bits: number,
                      group: number): Float32Array {
  const scaleCount = Math.ceil(count / group);
  const scales = new Float32Array(scaleCount);
  for (let i = 0; i < scaleCount; i++) {
    scales[i] = halfToFloat(view.getUint16(offset + i * 2, true));
    if (!(scales[i]! > 0) || !Number.isFinite(scales[i])) throw new Error("Invalid packed weight scale.");
  }
  const codeOffset = offset + scaleCount * 2;
  const codeBytes = Math.ceil(count * bits / 8);
  const out = new Float32Array(count);
  const levels = (1 << (bits - 1)) - 1;
  const mask = (1 << bits) - 1;
  for (let i = 0; i < count; i++) {
    const byte = Math.floor(i * bits / 8);
    const shift = (i * bits) % 8;
    const word = bytes[codeOffset + byte]! | ((byte + 1 < codeBytes ? bytes[codeOffset + byte + 1]! : 0) << 8);
    const code = (word >>> shift) & mask;
    if (code > 2 * levels) throw new Error("Invalid packed weight code.");
    out[i] = (code - levels) * scales[Math.floor(i / group)]!;
  }
  return out;
}

/** Decode the lexicon, group scales and low-bit codes once. */
export function decodePackedWeights(manifest: unknown, buffer: ArrayBuffer): ModelWeights {
  validateManifest(manifest);
  const m = manifest;
  if (buffer.byteLength !== m.weightBytes) throw new Error("Incorrect model weight byte length.");
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const h = m.hiddenDim, g = m.globalHiddenDim;
  let offset = LEX_BINS / 2;
  const lexicon = decodeLexicon(bytes.subarray(0, offset));
  const w1 = decodeMatrix(bytes, view, offset, WORD_DIM * h, m.bits, m.groupSize);
  offset += matrixBytes(WORD_DIM * h, m.bits, m.groupSize);
  const w3 = decodeMatrix(bytes, view, offset, GLOBAL_DIM * g, m.bits, m.groupSize);
  offset += matrixBytes(GLOBAL_DIM * g, m.bits, m.groupSize);
  const tail = new Float32Array(tailFloats(h, g));
  for (let i = 0; i < tail.length; i++) tail[i] = view.getFloat32(offset + i * 4, true);
  if (tail.some(v => !Number.isFinite(v))) throw new Error("Non-finite model weight.");
  let i = 0;
  const take = (n: number) => tail.slice(i, (i += n));
  const b1 = take(h), w2 = take(h), b2 = take(1)[0]!, mix = take(2), b3 = take(g), w4 = take(g);
  const vs = take(SHAPE_FEATURES), c = take(1)[0]!;
  return { lexicon, w1, b1, w2, b2, mix: [mix[0]!, mix[1]!], w3, b3, w4, vs, c };
}

/** Plain-JS reference forward pass (no GPU), used by tests and as a fallback check. */
export function referenceProbability(w: ModelWeights, words: Float32Array, count: number, global: Float32Array): number {
  const h = w.b1.length, g = w.b3.length;
  let expSum = 0, sum = 0;
  for (let k = 0; k < count; k++) {
    let s = w.b2;
    for (let j = 0; j < h; j++) {
      let v = w.b1[j]!;
      for (let i = 0; i < WORD_DIM; i++) v += w.w1[j * WORD_DIM + i]! * words[k * WORD_DIM + i]!;
      s += Math.max(v, 0) * w.w2[j]!;
    }
    s = Math.min(30, Math.max(-30, s));
    expSum += Math.exp(-s);
    sum += s;
  }
  let logit = w.mix[0] * -Math.log(expSum) + w.mix[1] * (sum / count) + w.c;
  for (let j = 0; j < g; j++) {
    let v = w.b3[j]!;
    for (let i = 0; i < GLOBAL_DIM; i++) v += w.w3[j * GLOBAL_DIM + i]! * global[i]!;
    logit += Math.max(v, 0) * w.w4[j]!;
  }
  for (let i = 0; i < SHAPE_FEATURES; i++) logit += w.vs[i]! * global[GLOBAL_BINS + i]!;
  return 1 / (1 + Math.exp(-logit));
}
