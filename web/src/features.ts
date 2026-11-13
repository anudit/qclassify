/** Mirrors src/qclassify/features.py, including code-point indexing, UTF-8 hashing and float32 rounding. */
export const FEATURE_VERSION = "nfkc-lower-words2to4-lex4096-global1024-v5";
export const WORD_BINS = 2028;
export const WORD_EXTRA = 12;
export const WORD_DIM = 2048;
export const GLOBAL_BINS = 1024;
export const SHAPE_FEATURES = 20;
export const GLOBAL_DIM = GLOBAL_BINS + SHAPE_FEATURES;
export const LEX_BINS = 4096;
export const MAX_WORDS = 16;
export const MAX_CHARS = 256;
const encoder = new TextEncoder();
// Python's Unicode whitespace set; JS \s differs for BOM, NEL, and U+001C..1F.
const SPACE = /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u;
const SPACES = new RegExp(SPACE.source + "+", "gu");
const LETTER = /\p{L}/u;
const NUMBER = /\p{N}/u;
const DECIMAL = /\p{Nd}/u;
const UPPER = /\p{Uppercase}/u;
const REQUEST_WORDS = new Set("how what where when why who find show latest recent best compare search".split(" "));
const ARTIFACT_WORDS = new Set("biography biographies papers publications tutorials courses interviews interview blog projects announcements github books news talks lecture lectures research tools libraries library frameworks software datasets opportunities jobs".split(" "));
const TECH_WORDS = new Set("ai ml llm llms api sdk sql gpu cpu webgl webgpu onnx pytorch tensorflow javascript typescript python rust open source".split(" "));
const trim = (s: string) => s.replace(new RegExp("^" + SPACE.source + "+|" + SPACE.source + "+$", "gu"), "");

export function canonicalize(text: string): string {
  return trim(text.normalize("NFKC").toLowerCase()).replace(SPACES, " ");
}

export function fnv1a(text: string): number {
  let hash = 2166136261;
  for (const byte of encoder.encode(text)) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
  return hash;
}

/** 4096 values in sevenths, -1 (ordinary word) .. +1 (usually a name); from the model's weight blob. */
export type Lexicon = Float32Array;

export function decodeLexicon(bytes: Uint8Array): Lexicon {
  if (bytes.length !== LEX_BINS / 2) throw new Error("Invalid lexicon length.");
  const table = new Float32Array(LEX_BINS);
  for (let i = 0; i < bytes.length; i++) {
    const lo = bytes[i]! & 15, hi = bytes[i]! >>> 4;
    if (lo > 14 || hi > 14) throw new Error("Invalid lexicon code.");
    table[2 * i] = (lo - 7) / 7;
    table[2 * i + 1] = (hi - 7) / 7;
  }
  return table;
}

export interface Features {
  /** count * WORD_DIM values, one row per word (1..MAX_WORDS). */
  words: Float32Array;
  count: number;
  /** GLOBAL_DIM values. */
  global: Float32Array;
}

function grams(padded: string[], out: Float32Array, offset: number, bins: number): void {
  for (const size of [2, 3, 4]) {
    for (let start = 0; start <= padded.length - size; start++) {
      const hash = fnv1a(padded.slice(start, start + size).join(""));
      const index = offset + (hash % bins);
      out[index] = out[index]! + ((hash & 0x80000000) ? 1 : -1);
    }
  }
  let squared = 0;
  for (let i = offset; i < offset + bins; i++) squared += out[i]! * out[i]!;
  // NumPy's float32 norm and float32 division.
  const norm = Math.fround(Math.sqrt(Math.fround(squared)));
  if (norm) for (let i = offset; i < offset + bins; i++) out[i] = out[i]! / norm;
}

const isAlpha = (w: string) => Array.from(w).every(c => LETTER.test(c));

export function featurize(text: string, lexicon: Lexicon): Features {
  if (typeof text !== "string" || !trim(text)) throw new TypeError("Input must be a nonempty string.");
  if (!(lexicon instanceof Float32Array) || lexicon.length !== LEX_BINS) throw new TypeError("Invalid lexicon.");
  // Lowercase first: every feature sees lowercase text only.
  const raw = Array.from(trim(text.normalize("NFKC"))).slice(0, MAX_CHARS).join("").toLowerCase();
  const normalized = canonicalize(raw);
  const tokens = normalized.split(" ").slice(0, MAX_WORDS);
  const n = tokens.length;
  const words = new Float32Array(n * WORD_DIM);
  tokens.forEach((tok, i) => {
    const o = i * WORD_DIM;
    grams(Array.from("^" + tok + "$"), words, o, WORD_BINS);
    const chars = Array.from(tok);
    const v = lexicon[fnv1a("#" + tok) % LEX_BINS]!;
    words.set([
      Math.min(chars.length / 16, 1), Number(chars.some(c => DECIMAL.test(c))),
      Number(chars.some(c => LETTER.test(c) && c.codePointAt(0)! > 127)),
      Number(REQUEST_WORDS.has(tok)), Number(ARTIFACT_WORDS.has(tok)), Number(TECH_WORDS.has(tok)),
      Number(n === 1), Number(i === 0 && n > 1), Number(i === n - 1 && n > 1), Number(!isAlpha(tok)),
      v, Number(v !== 0),
    ], o + WORD_BINS);
  });
  const global = new Float32Array(GLOBAL_DIM);
  grams(Array.from("^" + normalized + "$"), global, 0, GLOBAL_BINS);
  const chars = Array.from(raw);
  const rawWords = raw.split(SPACES).filter(Boolean);
  const all = normalized.split(" ");
  const count = (predicate: (c: string) => boolean) => chars.filter(predicate).length / chars.length;
  global.set([
    Math.min(chars.length / 80, 1), Math.min(rawWords.length / 10, 1),
    count(c => DECIMAL.test(c)), count(c => UPPER.test(c)), count(c => LETTER.test(c)),
    count(c => SPACE.test(c)), count(c => !(LETTER.test(c) || NUMBER.test(c) || SPACE.test(c))),
    rawWords.filter(w => UPPER.test(Array.from(w)[0]!)).length / rawWords.length,
    Number(raw.includes("?")), Number(raw.includes("@")), Number(raw.includes("/")),
    Number(raw.includes(".")), Number(raw.includes("-")), Number(raw.includes("'") || raw.includes("’")),
    Number(chars.some(c => LETTER.test(c) && c.codePointAt(0)! > 127)),
    Math.min(Math.max(...rawWords.map(w => Array.from(w).length)) / 24, 1),
    Number(REQUEST_WORDS.has(all[0]!)),
    all.filter(t => ARTIFACT_WORDS.has(t)).length / all.length,
    all.filter(t => TECH_WORDS.has(t)).length / all.length,
    Number(normalized.includes("http://") || normalized.includes("https://") || normalized.includes("www.")),
  ], GLOBAL_BINS);
  return { words, count: n, global };
}
