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

export function outputShader(hiddenDim: number): string {
  return header + `
uniform sampler2D hidden;
uniform sampler2D weights;
uniform float bias;
void main() {
  int row = int(gl_FragCoord.y);
  float value = bias;
  for (int i = 0; i < ${hiddenDim}; i++) {
    value += decodeFloat(texelFetch(hidden, ivec2(i, row), 0)) *
             texelFetch(weights, ivec2(i, 0), 0).r;
  }
  result = encodeFloat(1.0 / (1.0 + exp(-value)));
}`;
}
