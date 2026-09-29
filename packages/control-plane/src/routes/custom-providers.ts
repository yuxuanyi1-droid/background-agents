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
  type CustomModelRecord,
} from "@open-inspect/shared/types/custom-providers";
import { Hono } from "hono";
import { CustomProviderStore } from "../db/custom-providers";
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

function store(env: Env, ctx: RequestContext): CustomProviderStore | null {
  if (!ctx.db || !env.PROVIDER_ACCOUNTS_ENCRYPTION_KEY) return null;
  return new CustomProviderStore(ctx.db, env.PROVIDER_ACCOUNTS_ENCRYPTION_KEY);
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
  const record = await providers.update(params.id, parsed.data);
  if (!record) return error("Custom provider not found", 404);
  logger.info("custom_provider.updated", {
    event: "custom_provider.updated",
    provider_id: record.id,
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
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(provider.protocol === "anthropic"
      ? { "x-api-key": apiKey, "anthropic-version": "2023-06-01" }
      : { Authorization: `Bearer ${apiKey}` }),
    ...Object.fromEntries(
      provider.headers.map((header: { name: string; value: string }) => [header.name, header.value])
    ),
  };
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
  return json({ models });
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
