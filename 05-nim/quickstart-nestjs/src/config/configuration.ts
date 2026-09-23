/**
 * Settings. The NGC/NVIDIA key is read from the environment — never hard-coded.
 *
 * Paths are resolved from the compiled location (`dist/config`), so `../..` is
 * the project root whether you run `nest start` or `node dist/main.js`.
 */
import { join, resolve } from 'node:path';

export const PROJECT_ROOT = resolve(__dirname, '..', '..');
export const LOCAL_ENV = join(PROJECT_ROOT, '.env');
export const REPO_ENV = resolve(PROJECT_ROOT, '..', '..', '00-environment', '.env');
export const PUBLIC_DIR = join(PROJECT_ROOT, 'public');

/**
 * @nestjs/config keeps the FIRST definition it sees, so the local .env wins over
 * the shared repo one, and the real process environment wins over both.
 */
export const ENV_FILES = [LOCAL_ENV, REPO_ENV];

export interface AppConfig {
  nvidiaApiKey: string | null;
  ngcApiKey: string | null;
  nimBaseUrl: string;
  nimModel: string;
  requestTimeoutMs: number;
  corsOrigins: string[];
  port: number;
}

function splitOrigins(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
}

export default (): { app: AppConfig } => ({
  app: {
    nvidiaApiKey: process.env.NVIDIA_API_KEY ?? null,
    ngcApiKey: process.env.NGC_API_KEY ?? null,
    nimBaseUrl: process.env.NIM_BASE_URL ?? 'https://integrate.api.nvidia.com/v1',
    // Hosted catalogue churns, and /models lists more than your key can invoke:
    // retired ids answer 410, un-entitled ones 404. Verified working for this key
    // on 2026-09-23. See README for how to re-probe.
    nimModel: process.env.NIM_MODEL ?? 'meta/llama-3.2-11b-vision-instruct',
    requestTimeoutMs: Number(process.env.REQUEST_TIMEOUT_MS ?? 60_000),
    corsOrigins: splitOrigins(process.env.CORS_ORIGINS),
    port: Number(process.env.PORT ?? 8081),
  },
});
