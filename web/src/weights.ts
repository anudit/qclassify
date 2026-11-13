import { FEATURE_VERSION, HASH_BINS, INPUT_DIM, MAX_CHARS } from "./features.js";

export interface ModelManifest {
  format: "qclassify-webgl-v2";
  featureVersion: string;
  inputDim: number;
  hashBins: number;
  hiddenDim: number;
  maxChars: number;
  weightBytes: number;
  bits: number;
  groupSize: number;
  sha256: string;
  layout: string[];
  labels: string[];
  threshold: number;
}

export function validateManifest(value: unknown): asserts value is ModelManifest {
  if (!value || typeof value !== "object") throw new Error("Invalid model manifest.");
  const m = value as ModelManifest;
  const count = INPUT_DIM * m.hiddenDim;
  const expectedBytes = Math.ceil(count / m.groupSize) * 2 + Math.ceil(count * m.bits / 8) + m.hiddenDim * 6 + 4;
  if (m.format !== "qclassify-webgl-v2" || m.featureVersion !== FEATURE_VERSION ||
      m.inputDim !== INPUT_DIM || m.hashBins !== HASH_BINS || m.maxChars !== MAX_CHARS ||
      !Number.isSafeInteger(m.hiddenDim) || m.hiddenDim < 1 ||
      !Number.isSafeInteger(m.bits) || m.bits < 2 || m.bits > 8 ||
      m.groupSize !== 32 || m.weightBytes !== expectedBytes || expectedBytes >= 134_000 ||
      (INPUT_DIM * m.hiddenDim + 2 * m.hiddenDim + 1) * 4 >= 134_000 || m.threshold !== 0.5 ||
      JSON.stringify(m.labels) !== '["search_query","person_name"]' ||
      JSON.stringify(m.layout) !== '["scales-f16","codes-lsb","b1-f32","w2-f16","b2-f32"]' ||
      typeof m.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(m.sha256)) {
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

/** Decode group scales + low-bit codes once. Output matches the GPU texture layout. */
export function decodePackedWeights(manifest: unknown, buffer: ArrayBuffer): Float32Array {
  validateManifest(manifest);
  const m = manifest;
  if (buffer.byteLength !== m.weightBytes) throw new Error("Incorrect model weight byte length.");
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const count = m.inputDim * m.hiddenDim;
  const scaleCount = Math.ceil(count / m.groupSize);
  const codeOffset = scaleCount * 2;
  const codeBytes = Math.ceil(count * m.bits / 8);
  const scales = new Float32Array(scaleCount);
  for (let i = 0; i < scaleCount; i++) {
    scales[i] = halfToFloat(view.getUint16(i * 2, true));
    if (!(scales[i]! > 0) || !Number.isFinite(scales[i])) throw new Error("Invalid packed weight scale.");
  }
  const output = new Float32Array(count + m.hiddenDim * 2 + 1);
  const levels = (1 << (m.bits - 1)) - 1;
  const mask = (1 << m.bits) - 1;
  for (let i = 0; i < count; i++) {
    const byte = Math.floor(i * m.bits / 8);
    const shift = (i * m.bits) % 8;
    const word = bytes[codeOffset + byte]! | ((byte + 1 < codeBytes ? bytes[codeOffset + byte + 1]! : 0) << 8);
    const code = (word >>> shift) & mask;
    if (code > 2 * levels) throw new Error("Invalid packed weight code.");
    output[i] = (code - levels) * scales[Math.floor(i / m.groupSize)]!;
  }
  let offset = codeOffset + codeBytes;
  for (let i = 0; i < m.hiddenDim; i++, offset += 4) output[count + i] = view.getFloat32(offset, true);
  for (let i = 0; i < m.hiddenDim; i++, offset += 2) output[count + m.hiddenDim + i] = halfToFloat(view.getUint16(offset, true));
  output[output.length - 1] = view.getFloat32(offset, true);
  if (output.some(value => !Number.isFinite(value))) throw new Error("Non-finite model weight.");
  return output;
}
