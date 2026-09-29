/**
 * Public model-catalog cache used to prefill metadata when importing
 * custom-provider models. One D1 row holds a slimmed OpenRouter catalog;
 * sync-models refreshes it lazily (24h TTL) and matches gateway model IDs
 * against it. Catalog data only fills import defaults — a stale or failed
 * refresh degrades to hand-entered defaults, never a sync failure.
 */

import {
  CUSTOM_MODEL_MODALITIES,
  type CustomModelModality,
  type ModelCatalogMatch,
} from "@open-inspect/shared/types/custom-providers";
import type { SqlDatabase } from "./sql-database";

const OPENROUTER_SOURCE = "openrouter";
const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const CATALOG_REFRESH_TTL_MS = 24 * 60 * 60 * 1000;
const CATALOG_FETCH_TIMEOUT_MS = 5_000;
const MAX_CATALOG_ENTRIES = 2_000;
const CACHE_PAYLOAD_MAX_BYTES = 1024 * 1024;

interface CatalogMetadata {
  contextWindowTokens: number;
  maxOutputTokens: number | null;
  inputModalities: CustomModelModality[];
  outputModalities: CustomModelModality[];
}

/** Catalog lookups by normalized full id (`vendor/model`) and bare id (`model`). */
export interface CatalogIndex {
  byFullId: Map<string, CatalogMetadata>;
  /**
   * Bare id → metadata, or null when two catalog vendors share the bare id:
   * an ambiguous suffix must not pick one arbitrarily.
   */
  byBareId: Map<string, CatalogMetadata | null>;
}

function normalizeModelKey(value: string): string {
  const colon = value.indexOf(":");
  const withoutVariant = colon >= 0 ? value.slice(0, colon) : value;
  return withoutVariant.trim().toLowerCase();
}

/**
 * Match one gateway model ID against the catalog: an exact full-id match
 * wins, then a unique bare-id (suffix) match. Gateway lists usually carry
 * bare IDs while catalog ids are `vendor/model`; `:variant` suffixes are
 * stripped on both sides.
 */
export function matchCatalogModel(modelId: string, index: CatalogIndex): ModelCatalogMatch | null {
  const normalized = normalizeModelKey(modelId);
  const direct = index.byFullId.get(normalized);
  if (direct) return direct;
  const slash = normalized.indexOf("/");
  const bare = slash >= 0 ? normalized.slice(slash + 1) : normalized;
  if (!bare) return null;
  return index.byBareId.get(bare) ?? null;
}

function asPositiveInt(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

/** Modality strings the system models, dropped from the rest (`file`, …). */
function asModalityList(value: unknown): CustomModelModality[] {
  if (!Array.isArray(value)) return [];
  return [
    ...new Set(
      value.filter(
        (item): item is CustomModelModality =>
          typeof item === "string" && (CUSTOM_MODEL_MODALITIES as readonly string[]).includes(item)
      )
    ),
  ];
}

function parseCatalogEntry(entry: unknown): [string, CatalogMetadata] | null {
  if (typeof entry !== "object" || entry === null) return null;
  const record = entry as Record<string, unknown>;
  const id = typeof record.id === "string" ? record.id : null;
  if (!id) return null;
  const topProvider = (record.top_provider ?? null) as Record<string, unknown> | null;
  const contextWindowTokens =
    asPositiveInt(record.context_length) ??
    (topProvider === null ? null : asPositiveInt(topProvider.context_length));
  if (contextWindowTokens === null) return null;
  const maxOutputTokens =
    topProvider === null ? null : asPositiveInt(topProvider.max_completion_tokens);
  const architecture = (record.architecture ?? null) as Record<string, unknown> | null;
  return [
    id,
    {
      contextWindowTokens,
      maxOutputTokens,
      inputModalities: architecture === null ? [] : asModalityList(architecture.input_modalities),
      outputModalities: architecture === null ? [] : asModalityList(architecture.output_modalities),
    },
  ];
}

function parseCatalogPayload(payload: string): Map<string, CatalogMetadata> | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return null;
  }
  if (!Array.isArray(parsed)) return null;
  const entries = parsed
    .map(parseCatalogEntry)
    .filter((entry): entry is [string, CatalogMetadata] => entry !== null)
    .slice(0, MAX_CATALOG_ENTRIES);
  return new Map(entries);
}

export function buildCatalogIndex(entries: Map<string, CatalogMetadata>): CatalogIndex {
  const byFullId = new Map<string, CatalogMetadata>();
  const byBareId = new Map<string, CatalogMetadata | null>();
  for (const [id, metadata] of entries) {
    const normalized = normalizeModelKey(id);
    // The catalog lists a model's base entry before its `:variant` entries;
    // first wins so variant metadata neither overwrites the base nor marks
    // the shared bare id ambiguous.
    if (byFullId.has(normalized)) continue;
    byFullId.set(normalized, metadata);
    const slash = normalized.indexOf("/");
    const bare = slash >= 0 ? normalized.slice(slash + 1) : normalized;
    if (!bare) continue;
    byBareId.set(bare, byBareId.has(bare) ? null : metadata);
  }
  return { byFullId, byBareId };
}

interface OpenRouterListResponse {
  data?: unknown;
}

/** The slimmed catalog as it is cached: `[id, metadata]` tuples. */
type CatalogPayload = Array<[string, CatalogMetadata]>;

async function fetchOpenRouterCatalog(): Promise<CatalogPayload | null> {
  let response: Response;
  try {
    response = await fetch(OPENROUTER_MODELS_URL, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(CATALOG_FETCH_TIMEOUT_MS),
      redirect: "follow",
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  let body: OpenRouterListResponse;
  try {
    body = (await response.json()) as OpenRouterListResponse;
  } catch {
    return null;
  }
  if (!Array.isArray(body.data)) return null;
  const payload: CatalogPayload = [];
  for (const item of body.data) {
    const entry = parseCatalogEntry(item);
    if (entry === null) continue;
    payload.push(entry);
    if (payload.length >= MAX_CATALOG_ENTRIES) break;
  }
  return payload;
}

export class ModelCatalogCache {
  constructor(private readonly db: SqlDatabase) {}

  /**
   * Matched catalog metadata per gateway model ID. Refreshes the cache when
   * stale and falls back to the stale payload when the refresh fails; both
   * failures degrade to an empty result.
   */
  async matchAll(modelIds: readonly string[]): Promise<Map<string, ModelCatalogMatch>> {
    const index = await this.index();
    if (index === null) return new Map();
    const matches = new Map<string, ModelCatalogMatch>();
    for (const modelId of modelIds) {
      const match = matchCatalogModel(modelId, index);
      if (match !== null) matches.set(modelId, match);
    }
    return matches;
  }

  private async index(): Promise<CatalogIndex | null> {
    const cached = await this.read();
    if (cached !== null && Date.now() - cached.fetchedAt < CATALOG_REFRESH_TTL_MS) {
      return cached.index;
    }
    const payload = await fetchOpenRouterCatalog();
    if (payload !== null) return await this.write(payload);
    return cached?.index ?? null;
  }

  private async read(): Promise<{ index: CatalogIndex; fetchedAt: number } | null> {
    const row = await this.db
      .prepare(`SELECT payload, fetched_at FROM model_catalog_cache WHERE id = 1`)
      .first<{ payload: unknown; fetched_at: unknown }>();
    if (row === null || typeof row.payload !== "string" || typeof row.fetched_at !== "number") {
      return null;
    }
    const entries = parseCatalogPayload(row.payload);
    if (entries === null) return null;
    return { index: buildCatalogIndex(entries), fetchedAt: row.fetched_at };
  }

  /** Best-effort cache write; the fresh index is returned either way. */
  private async write(payload: CatalogPayload): Promise<CatalogIndex> {
    const index = buildCatalogIndex(new Map(payload));
    const serialized = JSON.stringify(payload);
    if (serialized.length > CACHE_PAYLOAD_MAX_BYTES) return index;
    try {
      await this.db
        .prepare(
          `INSERT INTO model_catalog_cache (id, source, payload, fetched_at)
           VALUES (1, ?, ?, ?)
           ON CONFLICT (id) DO UPDATE SET
             source = excluded.source,
             payload = excluded.payload,
             fetched_at = excluded.fetched_at`
        )
        .bind(OPENROUTER_SOURCE, serialized, Date.now())
        .run();
    } catch {
      // A failed cache write costs the next sync a refetch, nothing more.
    }
    return index;
  }
}
