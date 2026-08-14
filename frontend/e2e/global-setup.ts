/**
 * Global setup for e2e tests.
 *
 * Ensures the backend has deterministic test data before any test runs:
 *  1. The fixed e2e test user exists (registered via API, idempotent).
 *  2. The problem catalog is seeded (runs the backend's idempotent seed
 *     script inside the docker compose backend container when empty).
 *
 * All credentials and data are fixed — no random values — so runs are
 * reproducible against a fresh compose database.
 */
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..');
const API_BASE = 'http://localhost:8000/api/v1';

const TEST_USER = {
  email: 'e2etest@example.com',
  username: 'e2etestuser',
  password: 'TestPassword123!',
};

async function ensureTestUser(): Promise<void> {
  const resp = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(TEST_USER),
  });

  if (resp.status === 201) {
    console.log('[global-setup] Test user registered.');
    return;
  }

  // Any non-5xx response here means the user already exists (409/400) —
  // auth.setup.ts verifies the credentials actually work via the UI.
  if (resp.status < 500) {
    console.log(`[global-setup] Test user already exists (HTTP ${resp.status}).`);
    return;
  }

  const body = await resp.text();
  throw new Error(
    `[global-setup] Registering the e2e test user failed with HTTP ${resp.status}: ${body}`
  );
}

async function problemCount(): Promise<number> {
  const resp = await fetch(`${API_BASE}/problems?size=1`);
  if (!resp.ok) {
    throw new Error(`[global-setup] Listing problems failed with HTTP ${resp.status}`);
  }
  const data = (await resp.json()) as { total: number };
  return data.total;
}

async function ensureProblemsSeeded(): Promise<void> {
  let total = await problemCount();
  if (total > 0) {
    console.log(`[global-setup] Problem catalog already seeded (${total} problems).`);
    return;
  }

  console.log('[global-setup] Problem catalog empty — running backend seed script...');
  execSync(
    'docker compose exec -T backend python -m code_tutor.scripts.seed_problems',
    { cwd: REPO_ROOT, stdio: 'inherit' }
  );

  total = await problemCount();
  if (total === 0) {
    throw new Error('[global-setup] Seed script ran but the problem catalog is still empty.');
  }
  console.log(`[global-setup] Seeded problem catalog (${total} problems).`);
}

export default async function globalSetup(): Promise<void> {
  await ensureTestUser();
  await ensureProblemsSeeded();
}
