import { rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = fileURLToPath(new URL('../', import.meta.url));
rmSync(new URL('../dist', import.meta.url), { recursive: true, force: true });
execFileSync(process.execPath, [fileURLToPath(new URL('../node_modules/typescript/bin/tsc', import.meta.url)),
  '-p', fileURLToPath(new URL('../tsconfig.json', import.meta.url)), '--emitDeclarationOnly'], { stdio: 'inherit' });
await build({
  absWorkingDir: root,
  entryPoints: { index: 'src/index.ts', features: 'src/features.ts', weights: 'src/weights.ts' },
  outdir: 'dist', bundle: true, minify: true, format: 'esm', splitting: true,
  target: 'es2022', legalComments: 'none', metafile: true,
});
