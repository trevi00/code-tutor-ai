import type { APIRequestContext } from '@playwright/test';

export const API_BASE = 'http://localhost:8000/api/v1';

/**
 * Resolve a problem id by its exact title (with a prefix-match fallback).
 *
 * The seed script generates problem ids randomly, so tests must never
 * hardcode problem UUIDs — titles are the deterministic key.
 */
export async function getProblemIdByTitle(
  request: APIRequestContext,
  title: string
): Promise<string> {
  const resp = await request.get(`${API_BASE}/problems?size=100`);
  if (!resp.ok()) {
    throw new Error(`Failed to list problems: HTTP ${resp.status()}`);
  }
  const data = (await resp.json()) as { items: { id: string; title: string }[] };
  const match =
    data.items.find((p) => p.title === title) ??
    data.items.find((p) => p.title.startsWith(title));
  if (!match) {
    throw new Error(`No problem found with title "${title}"`);
  }
  return match.id;
}
