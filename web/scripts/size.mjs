import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync, brotliCompressSync, constants } from 'node:zlib';
const root = fileURLToPath(new URL('../', import.meta.url));
const size = data => ({ raw: Buffer.byteLength(data), gzip: gzipSync(data, { level: 9 }).length,
  brotli: brotliCompressSync(data, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length });
const visited = new Set();
const files = [];
function visit(file) {
  if (visited.has(file)) return;
  visited.add(file);
  const body = readFileSync(resolve(root, 'dist', file), 'utf8');
  files.push({ file, ...size(body) });
  for (const match of body.matchAll(/["']\.\/([^"']+\.js)["']/g)) visit(match[1]);
}
visit('index.js');
const totals = files.reduce((sum, row) => ({ raw: sum.raw + row.raw,
  gzip: sum.gzip + row.gzip, brotli: sum.brotli + row.brotli }), { raw: 0, gzip: 0, brotli: 0 });
const weights = readFileSync(resolve(root, '../artifacts/web/weights.bin'));
const base64 = `const s="${weights.toString('base64')}";export function decodeWeights(){return Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer}`;
const hex = `const s="${weights.toString('hex')}";export function decodeWeights(){const b=new Uint8Array(s.length/2);for(let i=0;i<b.length;i++)b[i]=parseInt(s.slice(i*2,i*2+2),16);return b.buffer}`;
const report = { package: '@omnid/qclassify', scope: 'Reachable minified ESM including dynamically imported model; each file compressed separately; excludes HTTP headers',
  runtimeDependencies: 0, weightBytes: weights.length, files, totals,
  embeddingComparison: { base64: size(base64), hex: size(hex), chosen: 'hex (smaller Brotli delivery)' } };
writeFileSync(resolve(root, '../artifacts/bundle-size.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
