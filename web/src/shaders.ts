export const vertexShader = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const header = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 result;
// Preserve a float32 as four bytes in a core RGBA8 render target.
// This avoids requiring EXT_color_buffer_float. Dithering must be disabled.
vec4 encodeFloat(float value) {
  uint b = floatBitsToUint(value);
  return vec4(float(b & 255u), float((b >> 8u) & 255u),
              float((b >> 16u) & 255u), float((b >> 24u) & 255u)) / 255.0;
}
float decodeFloat(vec4 value) {
  uvec4 b = uvec4(round(value * 255.0));
  return uintBitsToFloat(b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u));
}
`;

/** One fragment per (hidden unit x, row y): relu(bias + features[y] . weights[x]). Used for words and globals. */
export function hiddenShader(inputDim: number): string {
  return header + `
uniform sampler2D features;
uniform sampler2D weights;
uniform sampler2D biases;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float value = texelFetch(biases, ivec2(p.x, 0), 0).r;
  for (int i = 0; i < ${inputDim}; i++) {
    value += texelFetch(features, ivec2(i, p.y), 0).r *
             texelFetch(weights, ivec2(i, p.x), 0).r;
  }
  result = encodeFloat(max(value, 0.0));
}`;
}

/**
 * One fragment per input row: score each of its words, pool them (soft-min + mean), add the global branch and
 * the shape term, then sigmoid. `rows` holds (first word index, word count) per input row.
 */
export function outputShader(hidden: number, globalHidden: number, maxWords: number, globalBins: number,
                             shapeFeatures: number): string {
  return header + `
uniform sampler2D wordHidden;
uniform sampler2D globalHidden;
uniform sampler2D globalFeatures;
uniform sampler2D rows;
uniform float w2[${hidden}];
uniform float b2;
uniform vec2 pool;  // (soft-min, mean) weights
uniform float w4[${globalHidden}];
uniform float vs[${shapeFeatures}];
uniform float c;
void main() {
  int row = int(gl_FragCoord.y);
  int first = int(texelFetch(rows, ivec2(0, row), 0).r);
  int count = int(texelFetch(rows, ivec2(1, row), 0).r);
  float expSum = 0.0;
  float sum = 0.0;
  for (int k = 0; k < ${maxWords}; k++) {
    if (k >= count) break;
    float s = b2;
    for (int j = 0; j < ${hidden}; j++) {
      s += decodeFloat(texelFetch(wordHidden, ivec2(j, first + k), 0)) * w2[j];
    }
    s = clamp(s, -30.0, 30.0);
    expSum += exp(-s);
    sum += s;
  }
  float logit = pool.x * -log(expSum) + pool.y * (sum / float(count)) + c;
  for (int j = 0; j < ${globalHidden}; j++) {
    logit += decodeFloat(texelFetch(globalHidden, ivec2(j, row), 0)) * w4[j];
  }
  for (int i = 0; i < ${shapeFeatures}; i++) {
    logit += texelFetch(globalFeatures, ivec2(${globalBins} + i, row), 0).r * vs[i];
  }
  result = encodeFloat(1.0 / (1.0 + exp(-logit)));
}`;
}
