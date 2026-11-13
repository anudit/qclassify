import { GLOBAL_BINS, GLOBAL_DIM, MAX_WORDS, SHAPE_FEATURES, WORD_DIM, featurize, type Features } from "./features.js";
import { hiddenShader, outputShader, vertexShader } from "./shaders.js";
import { decodePackedWeights, validateManifest, type ModelManifest, type ModelWeights } from "./weights.js";

export { featurize, canonicalize, decodeLexicon, FEATURE_VERSION } from "./features.js";
export { validateManifest, decodePackedWeights, referenceProbability } from "./weights.js";
export type { ModelManifest, ModelWeights } from "./weights.js";
export type { Features, Lexicon } from "./features.js";
export type Label = "search_query" | "person_name";
export interface Prediction {
  input: string;
  label: Label;
  personProbability: number;
  searchProbability: number;
}
export type Canvas = HTMLCanvasElement | OffscreenCanvas;

function defaultCanvas(): Canvas {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(1, 1);
  if (typeof document !== "undefined") return document.createElement("canvas");
  throw new Error("WebGL 2 requires a browser canvas or OffscreenCanvas.");
}

const MAX_ROWS = 256;

/** Owns its WebGL context. Do not share the supplied canvas with another renderer. */
export class Classifier {
  private readonly gl: WebGL2RenderingContext;
  private readonly textures: WebGLTexture[] = [];
  private readonly framebuffers: WebGLFramebuffer[] = [];
  private readonly programs: WebGLProgram[] = [];
  private vao: WebGLVertexArrayObject | null = null;
  private disposed = false;
  private rowCapacity = 0;
  private wordCapacity = 0;
  private readonly weights: ModelWeights;
  private readonly hiddenDim: number;
  private readonly globalHiddenDim: number;
  private readonly maxWordRows: number;
  private wordFeatures!: WebGLTexture;
  private wordHidden!: WebGLTexture;
  private globalFeatures!: WebGLTexture;
  private globalHidden!: WebGLTexture;
  private rows!: WebGLTexture;
  private output!: WebGLTexture;
  private w1!: WebGLTexture;
  private b1!: WebGLTexture;
  private w3!: WebGLTexture;
  private b3!: WebGLTexture;
  private wordHiddenFB!: WebGLFramebuffer;
  private globalHiddenFB!: WebGLFramebuffer;
  private outputFB!: WebGLFramebuffer;
  private wordProgram!: WebGLProgram;
  private globalProgram!: WebGLProgram;
  private outputProgram!: WebGLProgram;

  /** Fetches and verifies a manifest and raw weights; no ONNX/WASM runtime is loaded. */
  static async create(options: {
    manifestUrl?: string | URL;
    weightsUrl?: string | URL;
    canvas?: Canvas;
  } = {}): Promise<Classifier> {
    if (!options.manifestUrl && !options.weightsUrl) {
      const { manifest, decodeWeights } = await import("./model.js");
      return Classifier.fromWeights(manifest, decodeWeights(), options.canvas);
    }
    if (!options.manifestUrl || !options.weightsUrl) {
      throw new Error("Supply both manifestUrl and weightsUrl, or neither for the bundled model.");
    }
    const [manifestResponse, weightResponse] = await Promise.all([
      fetch(options.manifestUrl), fetch(options.weightsUrl),
    ]);
    if (!manifestResponse.ok || !weightResponse.ok) {
      throw new Error(`Model fetch failed: manifest=${manifestResponse.status}, weights=${weightResponse.status}`);
    }
    const manifest: unknown = await manifestResponse.json();
    validateManifest(manifest);
    const weights = await weightResponse.arrayBuffer();
    if (!globalThis.crypto?.subtle) throw new Error("Model verification requires WebCrypto (HTTPS or localhost).");
    const digest = await crypto.subtle.digest("SHA-256", weights);
    const actual = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("");
    if (actual !== manifest.sha256) throw new Error("Model weight checksum mismatch.");
    return Classifier.fromWeights(manifest, weights, options.canvas);
  }

  /** For already-loaded, trusted assets. Validates shapes/finite values, but does not hash. */
  static fromWeights(manifest: unknown, weights: ArrayBuffer, canvas?: Canvas): Classifier {
    validateManifest(manifest);
    const values = decodePackedWeights(manifest, weights);
    return new Classifier(manifest, values, canvas ?? defaultCanvas());
  }

  private constructor(manifest: ModelManifest, weights: ModelWeights, canvas: Canvas) {
    this.weights = weights;
    this.hiddenDim = manifest.hiddenDim;
    this.globalHiddenDim = manifest.globalHiddenDim;
    const gl = canvas.getContext("webgl2", {
      alpha: false, antialias: false, depth: false, stencil: false,
      preserveDrawingBuffer: false, premultipliedAlpha: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL 2 is unavailable.");
    this.gl = gl;
    const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    this.maxWordRows = Math.min(4096, maxTexture);
    try {
      if (maxTexture < WORD_DIM) throw new Error("GPU texture size is too small for this model.");
      gl.disable(gl.DITHER);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.CULL_FACE);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
      this.vao = gl.createVertexArray();
      if (!this.vao) throw new Error("Cannot allocate WebGL vertex array.");
      gl.bindVertexArray(this.vao);
      const h = this.hiddenDim, g = this.globalHiddenDim;
      this.wordProgram = this.program(hiddenShader(WORD_DIM));
      this.globalProgram = this.program(hiddenShader(GLOBAL_DIM));
      this.outputProgram = this.program(outputShader(h, g, MAX_WORDS, GLOBAL_BINS, SHAPE_FEATURES));
      this.w1 = this.texture(WORD_DIM, h, weights.w1);
      this.b1 = this.texture(h, 1, weights.b1);
      this.w3 = this.texture(GLOBAL_DIM, g, weights.w3);
      this.b3 = this.texture(g, 1, weights.b3);
      this.wordFeatures = this.texture(WORD_DIM, 1, null);
      this.wordHidden = this.texture(h, 1, null, true);
      this.globalFeatures = this.texture(GLOBAL_DIM, 1, null);
      this.globalHidden = this.texture(g, 1, null, true);
      this.rows = this.texture(2, 1, null);
      this.output = this.texture(1, 1, null, true);
      this.wordHiddenFB = this.framebuffer(this.wordHidden);
      this.globalHiddenFB = this.framebuffer(this.globalHidden);
      this.outputFB = this.framebuffer(this.output);
      this.rowCapacity = this.wordCapacity = 1;
      const p = this.outputProgram;
      gl.useProgram(p);
      gl.uniform1fv(gl.getUniformLocation(p, "w2"), weights.w2);
      gl.uniform1f(gl.getUniformLocation(p, "b2"), weights.b2);
      gl.uniform2f(gl.getUniformLocation(p, "pool"), weights.mix[0], weights.mix[1]);
      gl.uniform1fv(gl.getUniformLocation(p, "w4"), weights.w4);
      gl.uniform1fv(gl.getUniformLocation(p, "vs"), weights.vs);
      gl.uniform1f(gl.getUniformLocation(p, "c"), weights.c);
      this.checkError("initialization");
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  private program(source: string): WebGLProgram {
    const gl = this.gl;
    const shaders: WebGLShader[] = [];
    let program: WebGLProgram | null = null;
    try {
      for (const [type, code] of [[gl.VERTEX_SHADER, vertexShader], [gl.FRAGMENT_SHADER, source]] as const) {
        const shader = gl.createShader(type);
        if (!shader) throw new Error("Cannot allocate WebGL shader.");
        shaders.push(shader);
        gl.shaderSource(shader, code);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
          throw new Error(`GLSL compilation failed: ${gl.getShaderInfoLog(shader)}`);
        }
      }
      program = gl.createProgram();
      if (!program) throw new Error("Cannot allocate WebGL program.");
      shaders.forEach(shader => gl.attachShader(program!, shader));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(`GLSL link failed: ${gl.getProgramInfoLog(program)}`);
      }
      this.programs.push(program);
      return program;
    } catch (error) {
      if (program) gl.deleteProgram(program);
      throw error;
    } finally {
      shaders.forEach(shader => gl.deleteShader(shader));
    }
  }

  private texture(width: number, height: number, data: Float32Array | null, bytes = false): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error("Cannot allocate WebGL texture.");
    this.textures.push(texture);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, bytes ? gl.RGBA8 : gl.R32F, width, height, 0,
                  bytes ? gl.RGBA : gl.RED, bytes ? gl.UNSIGNED_BYTE : gl.FLOAT, data);
    return texture;
  }

  private framebuffer(texture: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fb = gl.createFramebuffer();
    if (!fb) throw new Error("Cannot allocate WebGL framebuffer.");
    this.framebuffers.push(fb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("WebGL framebuffer is incomplete.");
    }
    return fb;
  }

  private bind(program: WebGLProgram, name: string, texture: WebGLTexture, unit: number): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, name), unit);
  }

  private resize(texture: WebGLTexture, width: number, height: number, bytes: boolean): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, bytes ? gl.RGBA8 : gl.R32F, width, height, 0,
                  bytes ? gl.RGBA : gl.RED, bytes ? gl.UNSIGNED_BYTE : gl.FLOAT, null);
  }

  private ensureCapacity(rows: number, words: number): void {
    if (rows > this.rowCapacity) {
      this.rowCapacity = Math.min(MAX_ROWS, Math.max(rows, this.rowCapacity * 2));
      this.resize(this.globalFeatures, GLOBAL_DIM, this.rowCapacity, false);
      this.resize(this.globalHidden, this.globalHiddenDim, this.rowCapacity, true);
      this.resize(this.rows, 2, this.rowCapacity, false);
      this.resize(this.output, 1, this.rowCapacity, true);
    }
    if (words > this.wordCapacity) {
      this.wordCapacity = Math.min(this.maxWordRows, Math.max(words, this.wordCapacity * 2));
      this.resize(this.wordFeatures, WORD_DIM, this.wordCapacity, false);
      this.resize(this.wordHidden, this.hiddenDim, this.wordCapacity, true);
    }
  }

  /** Synchronous GPU inference. Use a worker + OffscreenCanvas for large workloads. */
  predict(input: string): Prediction {
    return this.predictBatch([input])[0]!;
  }

  predictBatch(inputs: readonly string[]): Prediction[] {
    if (this.disposed) throw new Error("Classifier has been disposed.");
    if (this.gl.isContextLost()) throw new Error("WebGL context lost; create a new classifier.");
    const results: Prediction[] = [];
    let chunk: Features[] = [];
    let words = 0;
    const flush = () => {
      if (!chunk.length) return;
      const probabilities = this.run(chunk, words);
      for (let i = 0; i < chunk.length; i++) {
        const p = probabilities[i]!;
        if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error("Invalid GPU prediction.");
        results.push({ input: inputs[results.length]!, label: p >= 0.5 ? "person_name" : "search_query",
                       personProbability: p, searchProbability: 1 - p });
      }
      chunk = [];
      words = 0;
    };
    for (const text of inputs) {
      const features = featurize(text, this.weights.lexicon);
      if (chunk.length === MAX_ROWS || words + features.count > this.maxWordRows) flush();
      chunk.push(features);
      words += features.count;
    }
    flush();
    return results;
  }

  private run(chunk: Features[], wordCount: number): Float32Array {
    const gl = this.gl;
    const count = chunk.length;
    this.ensureCapacity(count, wordCount);
    const words = new Float32Array(wordCount * WORD_DIM);
    const globals = new Float32Array(count * GLOBAL_DIM);
    const rows = new Float32Array(count * 2);
    let first = 0;
    chunk.forEach((f, i) => {
      words.set(f.words, first * WORD_DIM);
      globals.set(f.global, i * GLOBAL_DIM);
      rows[2 * i] = first;
      rows[2 * i + 1] = f.count;
      first += f.count;
    });
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.wordFeatures);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, WORD_DIM, wordCount, gl.RED, gl.FLOAT, words);
    gl.bindTexture(gl.TEXTURE_2D, this.globalFeatures);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, GLOBAL_DIM, count, gl.RED, gl.FLOAT, globals);
    gl.bindTexture(gl.TEXTURE_2D, this.rows);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 2, count, gl.RED, gl.FLOAT, rows);
    // 1. every word's hidden layer   2. every row's global hidden layer   3. pool + combine per row
    gl.useProgram(this.wordProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.wordHiddenFB);
    gl.viewport(0, 0, this.hiddenDim, wordCount);
    this.bind(this.wordProgram, "features", this.wordFeatures, 0);
    this.bind(this.wordProgram, "weights", this.w1, 1);
    this.bind(this.wordProgram, "biases", this.b1, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(this.globalProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.globalHiddenFB);
    gl.viewport(0, 0, this.globalHiddenDim, count);
    this.bind(this.globalProgram, "features", this.globalFeatures, 0);
    this.bind(this.globalProgram, "weights", this.w3, 1);
    this.bind(this.globalProgram, "biases", this.b3, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(this.outputProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outputFB);
    gl.viewport(0, 0, 1, count);
    this.bind(this.outputProgram, "wordHidden", this.wordHidden, 0);
    this.bind(this.outputProgram, "globalHidden", this.globalHidden, 1);
    this.bind(this.outputProgram, "globalFeatures", this.globalFeatures, 2);
    this.bind(this.outputProgram, "rows", this.rows, 3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const bytes = new Uint8Array(count * 4);
    gl.readPixels(0, 0, 1, count, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    this.checkError("inference");
    const view = new DataView(bytes.buffer);
    const output = new Float32Array(count);
    for (let i = 0; i < count; i++) output[i] = view.getFloat32(i * 4, true);
    return output;
  }

  private checkError(stage: string): void {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR) throw new Error(`WebGL ${stage} failed: 0x${error.toString(16)}`);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    this.textures.forEach(texture => gl.deleteTexture(texture));
    this.framebuffers.forEach(fb => gl.deleteFramebuffer(fb));
    this.programs.forEach(program => gl.deleteProgram(program));
    if (this.vao) gl.deleteVertexArray(this.vao);
  }
}
