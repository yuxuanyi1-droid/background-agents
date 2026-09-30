/**
 * Custom model provider routes: administrator CRUD for gateway definitions,
 * on-demand model-list sync against the gateway, import/updates of model
 * metadata, and the flat catalog the model pickers read.
 */

import { parseJsonBody } from "./body";
import {
  createCustomProviderRequestSchema,
  updateCustomProviderRequestSchema,
  importCustomProviderModelsRequestSchema,
  updateCustomProviderModelRequestSchema,
  customProviderConnectionTestRequestSchema,
  type CustomModelRecord,
  type CustomProviderRecord,
  type ModelCatalogMatch,
} from "@open-inspect/shared/types/custom-providers";
import { Hono } from "hono";
import { CustomProviderStore } from "../db/custom-providers";
import { ModelCatalogCache } from "../db/model-catalog";
import { createLogger } from "../logger";
import { admit, dispatch } from "../routing/admit";
import type { ControlPlaneHonoEnv } from "../routing/hono-env";
import {
  SCM_AGNOSTIC_HUMAN_USER_ROUTE,
  error,
  json,
  requirePermission,
  type RequestContext,
} from "./shared";
import type { Env } from "../types";

const logger = createLogger("router:custom-providers");
const NO_STORE = "private, no-store" as const;
const SYNC_TIMEOUT_MS = 10_000;
const MAX_SYNCED_MODELS = 500;
const CONNECTION_TEST_TIMEOUT_MS = 20_000;
const CONNECTION_TEST_DETAIL_MAX_CHARS = 300;

function store(env: Env, ctx: RequestContext): CustomProviderStore | null {
  if (!ctx.db || !env.PROVIDER_ACCOUNTS_ENCRYPTION_KEY) return null;
  return new CustomProviderStore(ctx.db, env.PROVIDER_ACCOUNTS_ENCRYPTION_KEY);
}

/** Auth headers a gateway expects for its wire protocol, plus admin extras. */
function gatewayRequestHeaders(
  provider: Pick<CustomProviderRecord, "protocol" | "headers">,
  apiKey: string
): Record<string, string> {
  return {
    Accept: "application/json",
    ...(provider.protocol === "anthropic"
      ? { "x-api-key": apiKey, "anthropic-version": "2023-06-01" }
      : { Authorization: `Bearer ${apiKey}` }),
    ...Object.fromEntries(provider.headers.map((header) => [header.name, header.value])),
  };
}

async function listProviders(_request: Request, env: Env, _params: object, ctx: RequestContext) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  return json({ providers: await providers.list() });
}

async function createProvider(request: Request, env: Env, _params: object, ctx: RequestContext) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const body = await request.json().catch(() => null);
  const parsed = createCustomProviderRequestSchema.safeParse(body);
  if (!parsed.success)
    return error(`Invalid custom provider: ${parsed.error.issues[0]?.message}`, 400);
  const createdBy = ctx.principal?.kind === "user" ? ctx.principal.userId : null;
  const record = await providers.create({
    ...parsed.data,
    headers: parsed.data.headers ?? [],
    createdBy,
  });
  logger.info("custom_provider.created", {
    event: "custom_provider.created",
    provider_id: record.id,
    protocol: record.protocol,
    request_id: ctx.request_id,
  });
  return json({ provider: record }, 201);
}

async function getProvider(
  _request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const provider = await providers.getById(params.id);
  if (!provider) return error("Custom provider not found", 404);
  return json({ provider });
}

async function updateProvider(
  request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const parsed = updateCustomProviderRequestSchema.safeParse(
    await request.json().catch(() => null)
  );
  if (!parsed.success)
    return error(`Invalid custom provider update: ${parsed.error.issues[0]?.message}`, 400);
  let record: Awaited<ReturnType<typeof providers.update>>;
  try {
    record = await providers.update(params.id, parsed.data);
  } catch (updateError) {
    // Crossing the Anthropic/OpenAI boundary re-keys the provider; another
    // provider already holding the target key prefix is the only conflict.
    const message = updateError instanceof Error ? updateError.message : String(updateError);
    if (message.includes("idx_custom_provider_key_prefix")) {
      return error("Another provider already owns this protocol's provider key", 409);
    }
    throw updateError;
  }
  if (!record) return error("Custom provider not found", 404);
  logger.info("custom_provider.updated", {
    event: "custom_provider.updated",
    provider_id: record.id,
    protocol: record.protocol,
    request_id: ctx.request_id,
  });
  return json({ provider: record });
}

async function deleteProvider(
  _request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const deleted = await providers.delete(params.id);
  if (!deleted) return error("Custom provider not found", 404);
  logger.info("custom_provider.deleted", {
    event: "custom_provider.deleted",
    provider_id: params.id,
    request_id: ctx.request_id,
  });
  return json({ deleted: true });
}

interface SyncedModelListEntry {
  modelId: string;
  displayName: string;
}

function parseSyncedModelList(body: unknown): SyncedModelListEntry[] {
  if (
    typeof body !== "object" ||
    body === null ||
    !Array.isArray((body as { data?: unknown }).data)
  ) {
    return [];
  }
  const entries: SyncedModelListEntry[] = [];
  for (const item of (body as { data: unknown[] }).data) {
    if (typeof item !== "object" || item === null) continue;
    const id = (item as { id?: unknown }).id;
    if (typeof id !== "string" || id.length === 0 || id.length > 200) continue;
    const displayName = (item as { display_name?: unknown }).display_name;
    entries.push({ modelId: id, displayName: typeof displayName === "string" ? displayName : id });
    if (entries.length >= MAX_SYNCED_MODELS) break;
  }
  return entries;
}

async function syncProviderModels(
  _request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const provider = await providers.getById(params.id);
  if (!provider) return error("Custom provider not found", 404);
  const apiKey = await providers.readApiKey(provider.id);
  if (apiKey === null) return error("Custom provider has no stored API key", 409);

  const url = `${provider.baseUrl.replace(/\/+$/, "")}/models`;
  const headers = gatewayRequestHeaders(provider, apiKey);
  let response: Response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(SYNC_TIMEOUT_MS),
      redirect: "follow",
    });
  } catch (e) {
    logger.warn("custom_provider.sync_failed", {
      event: "custom_provider.sync_failed",
      provider_id: provider.id,
      error: e instanceof Error ? e.message : String(e),
      request_id: ctx.request_id,
    });
    return error(`Model list request to ${url} failed`, 502);
  }
  if (!response.ok) {
    return error(`Model list request to ${url} returned ${response.status}`, 502);
  }
  const models = parseSyncedModelList(await response.json().catch(() => null));
  if (models.length === 0) {
    return error(`No models found at ${url} (expected a data[] list)`, 502);
  }
  // Models already imported keep their admin-edited metadata, so the sync
  // result only offers the ones an import would actually add.
  const imported = new Set((await providers.listModels(provider.id)).map((model) => model.modelId));
  const visible = models.filter((model) => !imported.has(model.modelId));
  // Public-catalog metadata only prefills import defaults; a cache or fetch
  // failure must never fail the sync.
  const catalog = ctx.db ? new ModelCatalogCache(ctx.db) : null;
  const matches = catalog
    ? await catalog.matchAll(visible.map((model) => model.modelId)).catch(() => new Map())
    : new Map<string, ModelCatalogMatch>();
  return json({
    models: visible.map((model) => {
      const match = matches.get(model.modelId);
      return match ? { ...model, catalog: match } : model;
    }),
  });
}

/** A compact, relayable rendering of a gateway's error response body. */
function gatewayErrorDetail(status: number, bodyText: string): string {
  const text = bodyText.replace(/\s+/g, " ").trim().slice(0, CONNECTION_TEST_DETAIL_MAX_CHARS);
  return text.length > 0 ? `HTTP ${status}: ${text}` : `HTTP ${status}`;
}

/** The minimal one-token request each wire protocol accepts for a model test. */
function generationTestRequest(
  protocol: CustomProviderRecord["protocol"],
  modelId: string
): { path: string; body: string } {
  if (protocol === "anthropic") {
    return {
      path: "/messages",
      body: JSON.stringify({
        model: modelId,
        max_tokens: 1,
        messages: [{ role: "user", content: "ping" }],
      }),
    };
  }
  if (protocol === "openai_responses") {
    return {
      path: "/responses",
      body: JSON.stringify({ model: modelId, max_output_tokens: 1, input: "ping" }),
    };
  }
  return {
    path: "/chat/completions",
    body: JSON.stringify({
      model: modelId,
      max_tokens: 1,
      messages: [{ role: "user", content: "ping" }],
    }),
  };
}

/**
 * On-demand connectivity check. The result always answers 200 with `ok`
 * reflecting the gateway's verdict, so the UI can surface the detail text for
 * both outcomes; only configuration problems (missing provider/key) error out.
 */
async function testProviderConnection(
  request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const provider = await providers.getById(params.id);
  if (!provider) return error("Custom provider not found", 404);
  const apiKey = await providers.readApiKey(provider.id);
  if (apiKey === null) return error("Custom provider has no stored API key", 409);
  const parsed = customProviderConnectionTestRequestSchema.safeParse(
    (await request.json().catch(() => null)) ?? {}
  );
  if (!parsed.success)
    return error(`Invalid connection test: ${parsed.error.issues[0]?.message}`, 400);

  const root = provider.baseUrl.replace(/\/+$/, "");
  const modelId = parsed.data.modelId;
  const withModel = modelId !== undefined;
  const generation = withModel ? generationTestRequest(provider.protocol, modelId) : null;
  const target = generation ? `${root}${generation.path}` : `${root}/models`;
  const init: RequestInit = {
    method: generation ? "POST" : "GET",
    headers: generation
      ? { ...gatewayRequestHeaders(provider, apiKey), "Content-Type": "application/json" }
      : gatewayRequestHeaders(provider, apiKey),
    signal: AbortSignal.timeout(CONNECTION_TEST_TIMEOUT_MS),
    redirect: "follow",
    ...(generation ? { body: generation.body } : {}),
  };

  const startedAt = Date.now();
  let response: Response;
  try {
    response = await fetch(target, init);
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    logger.warn("custom_provider.connection_test_failed", {
      event: "custom_provider.connection_test_failed",
      provider_id: provider.id,
      error: reason,
      request_id: ctx.request_id,
    });
    return json({
      ok: false,
      mode: generation ? "generation" : "models",
      latencyMs: Date.now() - startedAt,
      detail: `Request to ${target} failed: ${reason}`,
    });
  }
  const latencyMs = Date.now() - startedAt;
  const mode = generation ? "generation" : "models";
  if (!response.ok) {
    const bodyText = await response.text().catch(() => "");
    return json({
      ok: false,
      mode,
      latencyMs,
      detail: gatewayErrorDetail(response.status, bodyText),
    });
  }
  if (generation) {
    return json({
      ok: true,
      mode,
      latencyMs,
      detail: `${modelId} responded to a one-token request`,
    });
  }
  const listed = parseSyncedModelList(await response.json().catch(() => null)).length;
  return json({
    ok: true,
    mode,
    latencyMs,
    detail:
      listed > 0
        ? `Model list reachable (${listed} models)`
        : "Model list reachable (no data[] entries returned)",
  });
}

async function listProviderModels(
  _request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  if (!(await providers.getById(params.id))) return error("Custom provider not found", 404);
  return json({ models: await providers.listModels(params.id) });
}

async function importProviderModels(
  request: Request,
  env: Env,
  params: { id: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  if (!(await providers.getById(params.id))) return error("Custom provider not found", 404);
  const body = await parseJsonBody(request);
  if (body instanceof Response) return body;
  const parsed = importCustomProviderModelsRequestSchema.safeParse(body);
  if (!parsed.success) return error(`Invalid import: ${parsed.error.issues[0]?.message}`, 400);
  const seen = new Set<string>();
  for (const model of parsed.data.models) {
    if (seen.has(model.modelId)) return error(`Duplicate model: ${model.modelId}`, 400);
    seen.add(model.modelId);
  }
  await providers.importModels(params.id, parsed.data.models);
  logger.info("custom_provider.models_imported", {
    event: "custom_provider.models_imported",
    provider_id: params.id,
    count: parsed.data.models.length,
    request_id: ctx.request_id,
  });
  return json({ models: await providers.listModels(params.id) });
}

function modelIdFromParams(params: { id: string; modelId: string }): string {
  return decodeURIComponent(params.modelId);
}

async function updateProviderModel(
  request: Request,
  env: Env,
  params: { id: string; modelId: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const parsed = updateCustomProviderModelRequestSchema.safeParse(
    await request.json().catch(() => null)
  );
  if (!parsed.success)
    return error(`Invalid model update: ${parsed.error.issues[0]?.message}`, 400);
  const updated = await providers.updateModel(params.id, modelIdFromParams(params), parsed.data);
  if (!updated) return error("Custom provider model not found", 404);
  return json({ models: await providers.listModels(params.id) });
}

async function deleteProviderModel(
  _request: Request,
  env: Env,
  params: { id: string; modelId: string },
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const deleted = await providers.deleteModel(params.id, modelIdFromParams(params));
  if (!deleted) return error("Custom provider model not found", 404);
  return json({ deleted: true });
}

async function getCustomModelsCatalog(
  _request: Request,
  env: Env,
  _params: object,
  ctx: RequestContext
) {
  const providers = store(env, ctx);
  if (!providers) return error("Custom provider storage is not configured", 503);
  const models: CustomModelRecord[] = await providers.getEnabledCatalog();
  return json({ models });
}

export const customProviderRoutes = new Hono<ControlPlaneHonoEnv>();

customProviderRoutes.get(
  "/custom-providers",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.read"),
    cacheControl: NO_STORE,
  }),
  (c) => dispatch(c, listProviders)
);

customProviderRoutes.post(
  "/custom-providers",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, createProvider)
);

customProviderRoutes.get(
  "/custom-providers/:id",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.read"),
    cacheControl: NO_STORE,
  }),
  (c) => dispatch(c, getProvider)
);

customProviderRoutes.patch(
  "/custom-providers/:id",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, updateProvider)
);

customProviderRoutes.delete(
  "/custom-providers/:id",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, deleteProvider)
);

customProviderRoutes.post(
  "/custom-providers/:id/sync-models",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, syncProviderModels)
);

customProviderRoutes.post(
  "/custom-providers/:id/test-connection",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, testProviderConnection)
);

customProviderRoutes.get(
  "/custom-providers/:id/models",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.read"),
    cacheControl: NO_STORE,
  }),
  (c) => dispatch(c, listProviderModels)
);

customProviderRoutes.put(
  "/custom-providers/:id/models",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, importProviderModels)
);

customProviderRoutes.patch(
  "/custom-providers/:id/models/:modelId",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, updateProviderModel)
);

customProviderRoutes.delete(
  "/custom-providers/:id/models/:modelId",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.manage"),
  }),
  (c) => dispatch(c, deleteProviderModel)
);

customProviderRoutes.get(
  "/custom-models",
  admit({
    ...SCM_AGNOSTIC_HUMAN_USER_ROUTE,
    authorization: requirePermission("custom_providers.read"),
    cacheControl: NO_STORE,
  }),
  (c) => dispatch(c, getCustomModelsCatalog)
);
