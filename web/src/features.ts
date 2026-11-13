/** Mirrors src/qclassify/features.py, including code-point indexing and UTF-8 hashing. */
export const FEATURE_VERSION = "nfkc-lower-fnv1a-char2to4-shape20-v4";
export const HASH_BINS = 2028;
export const INPUT_DIM = 2048;
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

export function featurize(text: string): Float32Array {
  if (typeof text !== "string" || !trim(text)) throw new TypeError("Input must be a nonempty string.");
  // Lowercase first: every feature, shape ones included, sees lowercase text only.
  const raw = Array.from(trim(text.normalize("NFKC"))).slice(0, MAX_CHARS).join("").toLowerCase();
  const chars = Array.from(raw);
  const normalized = Array.from("^" + canonicalize(raw) + "$");
  const vector = new Float32Array(INPUT_DIM);
  for (const size of [2, 3, 4]) {
    for (let start = 0; start <= normalized.length - size; start++) {
      const hash = fnv1a(normalized.slice(start, start + size).join(""));
      const index = hash % HASH_BINS;
      vector[index] = vector[index]! + ((hash & 0x80000000) ? 1 : -1);
    }
  }
  let squared = 0;
  for (let i = 0; i < HASH_BINS; i++) squared += vector[i]! * vector[i]!;
  // NumPy's float32 norm and float32 division.
  const norm = Math.fround(Math.sqrt(Math.fround(squared)));
  if (norm) for (let i = 0; i < HASH_BINS; i++) vector[i] = vector[i]! / norm;
  const words = raw.split(SPACES).filter(Boolean);
  const tokens = canonicalize(raw).split(" ");
  const count = (predicate: (c: string) => boolean) => chars.filter(predicate).length / chars.length;
  vector.set([
    Math.min(chars.length / 80, 1), Math.min(words.length / 10, 1),
    count(c => DECIMAL.test(c)), count(c => UPPER.test(c)), count(c => LETTER.test(c)),
    count(c => SPACE.test(c)), count(c => !(LETTER.test(c) || NUMBER.test(c) || SPACE.test(c))),
    words.filter(w => UPPER.test(Array.from(w)[0]!)).length / words.length,
    Number(raw.includes("?")), Number(raw.includes("@")), Number(raw.includes("/")),
    Number(raw.includes(".")), Number(raw.includes("-")), Number(raw.includes("'") || raw.includes("’")),
    Number(chars.some(c => LETTER.test(c) && c.codePointAt(0)! > 127)),
    Math.min(Math.max(...words.map(w => Array.from(w).length)) / 24, 1),
    Number(REQUEST_WORDS.has(tokens[0]!)),
    tokens.filter(t => ARTIFACT_WORDS.has(t)).length / tokens.length,
    tokens.filter(t => TECH_WORDS.has(t)).length / tokens.length,
    Number(raw.toLowerCase().includes("http://") || raw.toLowerCase().includes("https://") || raw.toLowerCase().includes("www.")),
  ], HASH_BINS);
  return vector;
}
