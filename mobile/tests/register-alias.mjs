/**
 * Lets the test runner load app modules that import through the `@/` alias.
 *
 * Node resolves paths, not tsconfig aliases, so `@/src/lib/nutrition` is
 * meaningless to it. This maps the alias to the project root and fills in the
 * `.ts` extension the app code omits.
 *
 * Two modules are swapped for stubs: `api-client` and `session` pull in
 * `expo-constants` and `expo-secure-store`, which cannot load outside a device.
 * Nothing under test calls them — they sit behind the remote-parsing branch,
 * which ships disabled — so a stub keeps the pure logic reachable without
 * pretending the native side exists.
 */
import fs from 'node:fs';
import { registerHooks } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const STUBBED = {
  '@/src/lib/api-client': 'tests/stubs/api-client.ts',
  '@/src/lib/session': 'tests/stubs/session.ts',
};

function withExtension(filePath) {
  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) return filePath;
  for (const candidate of [`${filePath}.ts`, `${filePath}.tsx`, path.join(filePath, 'index.ts')]) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return filePath;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    const stub = STUBBED[specifier];
    if (stub) return nextResolve(pathToFileURL(path.join(ROOT, stub)).href, context);
    if (specifier.startsWith('@/')) {
      const resolved = withExtension(path.join(ROOT, specifier.slice(2)));
      return nextResolve(pathToFileURL(resolved).href, context);
    }
    return nextResolve(specifier, context);
  },
});
