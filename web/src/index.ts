import { INPUT_DIM, featurize } from "./features.js";
import { hiddenShader, outputShader, vertexShader } from "./shaders.js";
import { decodePackedWeights, validateManifest, type ModelManifest } from "./weights.js";

export { featurize, canonicalize, FEATURE_VERSION } from "./features.js";
export { validateManifest, decodePackedWeights } from "./weights.js";
export type { ModelManifest } from "./weights.js";
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

/** Owns its WebGL context. Do not share the supplied canvas with another renderer. */
export class Classifier {
  private readonly gl: WebGL2RenderingContext;
  private readonly textures: WebGLTexture[] = [];
  private readonly framebuffers: WebGLFramebuffer[] = [];
  private readonly programs: WebGLProgram[] = [];
  private vao: WebGLVertexArrayObject | null = null;
  private disposed = false;
  private capacity = 0;
  private features!: WebGLTexture;
  private hidden!: WebGLTexture;
  private output!: WebGLTexture;
  private w1!: WebGLTexture;
  private b1!: WebGLTexture;
  private w2!: WebGLTexture;
  private b2 = 0;
  private hiddenFB!: WebGLFramebuffer;
  private outputFB!: WebGLFramebuffer;
  private hiddenProgram!: WebGLProgram;
  private outputProgram!: WebGLProgram;
  private readonly hiddenDim: number;
  private readonly maxBatch: number;

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

  private constructor(manifest: ModelManifest, weights: Float32Array, canvas: Canvas) {
    this.hiddenDim = manifest.hiddenDim;
    const gl = canvas.getContext("webgl2", {
      alpha: false, antialias: false, depth: false, stencil: false,
      preserveDrawingBuffer: false, premultipliedAlpha: false,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL 2 is unavailable.");
    this.gl = gl;
    this.maxBatch = Math.min(256, gl.getParameter(gl.MAX_TEXTURE_SIZE) as number);
    try {
      if ((gl.getParameter(gl.MAX_TEXTURE_SIZE) as number) < INPUT_DIM) {
        throw new Error("GPU texture size is too small for this model.");
      }
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
      this.hiddenProgram = this.program(hiddenShader(INPUT_DIM));
      this.outputProgram = this.program(outputShader(this.hiddenDim));
      const w1End = INPUT_DIM * this.hiddenDim;
      const b1End = w1End + this.hiddenDim;
      const w2End = b1End + this.hiddenDim;
      this.w1 = this.texture(INPUT_DIM, this.hiddenDim, weights.subarray(0, w1End));
      this.b1 = this.texture(this.hiddenDim, 1, weights.subarray(w1End, b1End));
      this.w2 = this.texture(this.hiddenDim, 1, weights.subarray(b1End, w2End));
      this.b2 = weights[w2End]!;
      this.features = this.texture(INPUT_DIM, 1, null);
      this.hidden = this.texture(this.hiddenDim, 1, null, true);
      this.output = this.texture(1, 1, null, true);
      this.hiddenFB = this.framebuffer(this.hidden);
      this.outputFB = this.framebuffer(this.output);
      this.capacity = 1;
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

  private ensureCapacity(rows: number): void {
    if (rows <= this.capacity) return;
    const gl = this.gl;
    this.capacity = Math.min(this.maxBatch, Math.max(rows, this.capacity * 2));
    for (const [texture, width, bytes] of [
      [this.features, INPUT_DIM, false], [this.hidden, this.hiddenDim, true], [this.output, 1, true],
    ] as const) {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, bytes ? gl.RGBA8 : gl.R32F, width, this.capacity, 0,
                    bytes ? gl.RGBA : gl.RED, bytes ? gl.UNSIGNED_BYTE : gl.FLOAT, null);
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
    for (let start = 0; start < inputs.length; start += this.maxBatch) {
      const batch = inputs.slice(start, start + this.maxBatch);
      const features = new Float32Array(batch.length * INPUT_DIM);
      batch.forEach((text, i) => features.set(featurize(text), i * INPUT_DIM));
      const probabilities = this.run(features, batch.length);
      batch.forEach((input, i) => {
        const p = probabilities[i]!;
        if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error("Invalid GPU prediction.");
        results.push({ input, label: p >= 0.5 ? "person_name" : "search_query",
                       personProbability: p, searchProbability: 1 - p });
      });
    }
    return results;
  }

  private run(features: Float32Array, count: number): Float32Array {
    const gl = this.gl;
    this.ensureCapacity(count);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.features);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, INPUT_DIM, count, gl.RED, gl.FLOAT, features);
    gl.useProgram(this.hiddenProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hiddenFB);
    gl.viewport(0, 0, this.hiddenDim, count);
    this.bind(this.hiddenProgram, "features", this.features, 0);
    this.bind(this.hiddenProgram, "weights", this.w1, 1);
    this.bind(this.hiddenProgram, "biases", this.b1, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(this.outputProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outputFB);
    gl.viewport(0, 0, 1, count);
    this.bind(this.outputProgram, "hidden", this.hidden, 0);
    this.bind(this.outputProgram, "weights", this.w2, 1);
    gl.uniform1f(gl.getUniformLocation(this.outputProgram, "bias"), this.b2);
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
