/**
 * Custom model provider storage: gateway definitions, their encrypted API
 * keys, and the models imported from each gateway's model list. Serves three
 * readers — the settings routes (CRUD), the web model catalog (enabled
 * models across active providers), and the sandbox environment assembly
 * (a manifest plus one decrypted API key per provider).
 */

import {
  decryptProviderAccountPayload,
  encryptProviderAccountPayload,
} from "../auth/provider-account-crypto";
import { generateId } from "../auth/crypto";
import {
  CUSTOM_PROVIDER_ID_PATTERN,
  customProviderKey,
  isCustomModelId,
  type CustomModelModality,
  type CustomModelRecord,
  type CustomProviderHeader,
  type CustomProviderProtocol,
  type CustomProviderRecord,
} from "@open-inspect/shared/types/custom-providers";
import type { ReasoningEffort } from "@open-inspect/shared/models";
import type { SqlDatabase } from "./sql-database";
import { z } from "zod";

const CREDENTIAL_SCHEMA_VERSION = 1;

const providerRowSchema = z.object({
  id: z.string().regex(CUSTOM_PROVIDER_ID_PATTERN),
  name: z.string().min(1).max(100),
  protocol: z.enum(["anthropic", "openai_compatible"]),
  base_url: z.string().min(1),
  custom_headers: z.string(),
  status: z.enum(["active", "disabled"]),
  created_by: z.string().nullable(),
  created_at: z.number().int().nonnegative(),
  updated_at: z.number().int().nonnegative(),
});

const modelRowSchema = z.object({
  provider_id: z.string().regex(CUSTOM_PROVIDER_ID_PATTERN),
  model_id: z.string().min(1).max(200),
  display_name: z.string().min(1).max(200),
  modalities: z.string(),
  reasoning_efforts: z.string(),
  context_window_tokens: z.number().int().positive(),
  max_output_tokens: z.number().int().positive(),
  enabled: z.number().int().min(0).max(1),
  created_at: z.number().int().nonnegative(),
  updated_at: z.number().int().nonnegative(),
});

type ProviderRow = z.infer<typeof providerRowSchema>;
type ModelRow = z.infer<typeof modelRowSchema>;

export interface ImportedCustomModel {
  modelId: string;
  displayName: string;
  modalities: CustomModelModality[];
  reasoningEfforts: ReasoningEffort[];
  contextWindowTokens: number;
  maxOutputTokens: number;
}

/** One provider's sandbox-facing manifest entry; the key travels separately. */
export interface CustomProviderSandboxEntry {
  id: string;
  providerKey: string;
  protocol: CustomProviderProtocol;
  baseUrl: string;
  headers: CustomProviderHeader[];
  apiKeyEnv: string;
  models: {
    modelId: string;
    displayName: string;
    reasoningEfforts: ReasoningEffort[];
    contextWindowTokens: number;
    maxOutputTokens: number;
  }[];
}

function parseHeaders(raw: string): CustomProviderHeader[] {
  const parsed: unknown = JSON.parse(raw);
  if (!Array.isArray(parsed)) return [];
  return parsed.flatMap((entry) => {
    if (
      typeof entry === "object" &&
      entry !== null &&
      typeof (entry as { name?: unknown }).name === "string" &&
      typeof (entry as { value?: unknown }).value === "string"
    ) {
      return [
        { name: (entry as { name: string }).name, value: (entry as { value: string }).value },
      ];
    }
    return [];
  });
}

function parseJsonArray(raw: string): string[] {
  const parsed: unknown = JSON.parse(raw);
  return Array.isArray(parsed)
    ? parsed.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function providerFromRow(row: ProviderRow): CustomProviderRecord {
  return {
    id: row.id,
    name: row.name,
    protocol: row.protocol,
    baseUrl: row.base_url,
    headers: parseHeaders(row.custom_headers),
    status: row.status,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    createdBy: row.created_by,
    providerKey: customProviderKey(row.id, row.protocol),
  };
}

export class CustomProviderStore {
  constructor(
    private readonly db: SqlDatabase,
    private readonly encryptionKey: string
  ) {}

  async list(): Promise<CustomProviderRecord[]> {
    const results = await this.db
      .prepare(`SELECT * FROM custom_providers ORDER BY created_at ASC, id ASC`)
      .all();
    return results.results.flatMap((row) => {
      const parsed = providerRowSchema.safeParse(row);
      return parsed.success ? [providerFromRow(parsed.data)] : [];
    });
  }

  async getById(providerId: string): Promise<CustomProviderRecord | null> {
    const row = await this.db
      .prepare(`SELECT * FROM custom_providers WHERE id = ?`)
      .bind(providerId)
      .first();
    if (row === null) return null;
    const parsed = providerRowSchema.safeParse(row);
    if (!parsed.success) throw new Error(`Malformed custom provider row: ${providerId}`);
    return providerFromRow(parsed.data);
  }

  async create(input: {
    name: string;
    protocol: CustomProviderProtocol;
    baseUrl: string;
    apiKey: string;
    headers: CustomProviderHeader[];
    createdBy: string | null;
  }): Promise<CustomProviderRecord> {
    const now = Date.now();
    // The 8-hex key prefix must be unique so model IDs route unambiguously.
    for (let attempt = 0; attempt < 5; attempt += 1) {
      const id = generateId(16);
      try {
        await this.db
          .prepare(
            `INSERT INTO custom_providers (id, name, protocol, base_url, custom_headers, status,
               created_by, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)`
          )
          .bind(
            id,
            input.name,
            input.protocol,
            input.baseUrl,
            JSON.stringify(input.headers),
            input.createdBy,
            now,
            now
          )
          .run();
        await this.writeApiKey(id, input.apiKey, now);
        const record = await this.getById(id);
        if (!record) throw new Error("Custom provider vanished after insert");
        return record;
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        if (message.includes("idx_custom_provider_key_prefix") || attempt === 4) throw e;
      }
    }
    throw new Error("Could not allocate a unique custom provider ID");
  }

  async update(
    providerId: string,
    input: {
      name?: string;
      baseUrl?: string;
      headers?: CustomProviderHeader[];
      status?: "active" | "disabled";
      apiKey?: string;
    }
  ): Promise<CustomProviderRecord | null> {
    const now = Date.now();
    const assignments: string[] = ["updated_at = ?"];
    const values: unknown[] = [now];
    if (input.name !== undefined) {
      assignments.push("name = ?");
      values.push(input.name);
    }
    if (input.baseUrl !== undefined) {
      assignments.push("base_url = ?");
      values.push(input.baseUrl);
    }
    if (input.headers !== undefined) {
      assignments.push("custom_headers = ?");
      values.push(JSON.stringify(input.headers));
    }
    if (input.status !== undefined) {
      assignments.push("status = ?");
      values.push(input.status);
    }
    const result = await this.db
      .prepare(`UPDATE custom_providers SET ${assignments.join(", ")} WHERE id = ?`)
      .bind(...values, providerId)
      .run();
    if (result.meta.changes === 0) return null;
    if (input.apiKey !== undefined) {
      await this.writeApiKey(providerId, input.apiKey, now);
    }
    return this.getById(providerId);
  }

  async delete(providerId: string): Promise<boolean> {
    const result = await this.db
      .prepare(`DELETE FROM custom_providers WHERE id = ?`)
      .bind(providerId)
      .run();
    return result.meta.changes > 0;
  }

  async importModels(providerId: string, models: ImportedCustomModel[]): Promise<void> {
    const now = Date.now();
    const statements = [
      this.db.prepare(`DELETE FROM custom_provider_models WHERE provider_id = ?`).bind(providerId),
      ...models.map((model) =>
        this.db
          .prepare(
            `INSERT INTO custom_provider_models (
               provider_id, model_id, display_name, modalities, reasoning_efforts,
               context_window_tokens, max_output_tokens, enabled, created_at, updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)`
          )
          .bind(
            providerId,
            model.modelId,
            model.displayName,
            JSON.stringify(model.modalities),
            JSON.stringify(model.reasoningEfforts),
            model.contextWindowTokens,
            model.maxOutputTokens,
            now,
            now
          )
      ),
    ];
    await this.db.batch(statements);
  }

  async updateModel(
    providerId: string,
    modelId: string,
    input: {
      displayName?: string;
      modalities?: CustomModelModality[];
      reasoningEfforts?: ReasoningEffort[];
      contextWindowTokens?: number;
      maxOutputTokens?: number;
      enabled?: boolean;
    }
  ): Promise<boolean> {
    const assignments: string[] = ["updated_at = ?"];
    const values: unknown[] = [Date.now()];
    if (input.displayName !== undefined) {
      assignments.push("display_name = ?");
      values.push(input.displayName);
    }
    if (input.modalities !== undefined) {
      assignments.push("modalities = ?");
      values.push(JSON.stringify(input.modalities));
    }
    if (input.reasoningEfforts !== undefined) {
      assignments.push("reasoning_efforts = ?");
      values.push(JSON.stringify(input.reasoningEfforts));
    }
    if (input.contextWindowTokens !== undefined) {
      assignments.push("context_window_tokens = ?");
      values.push(input.contextWindowTokens);
    }
    if (input.maxOutputTokens !== undefined) {
      assignments.push("max_output_tokens = ?");
      values.push(input.maxOutputTokens);
    }
    if (input.enabled !== undefined) {
      assignments.push("enabled = ?");
      values.push(input.enabled ? 1 : 0);
    }
    const result = await this.db
      .prepare(
        `UPDATE custom_provider_models SET ${assignments.join(", ")}
         WHERE provider_id = ? AND model_id = ?`
      )
      .bind(...values, providerId, modelId)
      .run();
    return result.meta.changes > 0;
  }

  async deleteModel(providerId: string, modelId: string): Promise<boolean> {
    const result = await this.db
      .prepare(`DELETE FROM custom_provider_models WHERE provider_id = ? AND model_id = ?`)
      .bind(providerId, modelId)
      .run();
    return result.meta.changes > 0;
  }

  private async listModelRows(providerId: string): Promise<ModelRow[]> {
    const results = await this.db
      .prepare(
        `SELECT * FROM custom_provider_models WHERE provider_id = ?
         ORDER BY model_id ASC`
      )
      .bind(providerId)
      .all();
    return results.results.flatMap((row) => {
      const parsed = modelRowSchema.safeParse(row);
      return parsed.success ? [parsed.data] : [];
    });
  }

  async listModels(providerId: string): Promise<CustomModelRecord[]> {
    const provider = await this.getById(providerId);
    if (!provider) return [];
    const rows = await this.listModelRows(providerId);
    return rows.map((row) => this.modelRecord(provider, row));
  }

  /** The web catalog: enabled models across active providers. */
  async getEnabledCatalog(): Promise<CustomModelRecord[]> {
    const providers = (await this.list()).filter((provider) => provider.status === "active");
    const catalog: CustomModelRecord[] = [];
    for (const provider of providers) {
      const rows = await this.listModelRows(provider.id);
      for (const row of rows) {
        if (row.enabled === 1) catalog.push(this.modelRecord(provider, row));
      }
    }
    return catalog;
  }

  /**
   * Resolve a well-formed custom model ID against the registry: the routed
   * provider must exist and be active, and the model must be imported and
   * enabled. The protocol embedded in the ID must match the provider's.
   */
  async resolveCustomModel(
    modelId: string
  ): Promise<{ provider: CustomProviderRecord; model: CustomModelRecord } | null> {
    if (!isCustomModelId(modelId)) return null;
    const providerKey = modelId.slice(0, modelId.indexOf("/"));
    const providers = await this.list();
    const provider = providers.find((candidate) => candidate.providerKey === providerKey);
    if (!provider || provider.status !== "active") return null;
    const upstreamModelId = modelId.slice(modelId.indexOf("/") + 1);
    const rows = await this.listModelRows(provider.id);
    const row = rows.find((candidate) => candidate.model_id === upstreamModelId);
    if (!row || row.enabled !== 1) return null;
    return { provider, model: this.modelRecord(provider, row) };
  }

  /** The sandbox manifest: active providers, their enabled models, and keys. */
  async getSandboxEntries(): Promise<CustomProviderSandboxEntry[]> {
    const providers = (await this.list()).filter((provider) => provider.status === "active");
    const entries: CustomProviderSandboxEntry[] = [];
    for (const provider of providers) {
      const apiKey = await this.readApiKey(provider.id);
      if (apiKey === null) continue;
      const rows = (await this.listModelRows(provider.id)).filter((row) => row.enabled === 1);
      if (rows.length === 0) continue;
      entries.push({
        id: provider.id,
        providerKey: provider.providerKey,
        protocol: provider.protocol,
        baseUrl: provider.baseUrl,
        headers: provider.headers,
        apiKeyEnv: customProviderApiKeyEnv(provider.id),
        models: rows.map((row) => ({
          modelId: row.model_id,
          displayName: row.display_name,
          reasoningEfforts: parseJsonArray(row.reasoning_efforts) as ReasoningEffort[],
          contextWindowTokens: row.context_window_tokens,
          maxOutputTokens: row.max_output_tokens,
        })),
      });
    }
    return entries;
  }

  async readApiKey(providerId: string): Promise<string | null> {
    const row = await this.db
      .prepare(`SELECT encrypted_payload FROM custom_provider_secrets WHERE provider_id = ?`)
      .bind(providerId)
      .first<{ encrypted_payload: string }>();
    if (row === null) return null;
    const payload = await decryptProviderAccountPayload(row.encrypted_payload, this.encryptionKey, {
      providerAccountId: providerId,
      provider: "custom",
      credentialSchemaVersion: CREDENTIAL_SCHEMA_VERSION,
    });
    return typeof payload === "string" ? payload : null;
  }

  private async writeApiKey(providerId: string, apiKey: string, now: number): Promise<void> {
    const encrypted = await encryptProviderAccountPayload(apiKey, this.encryptionKey, {
      providerAccountId: providerId,
      provider: "custom",
      credentialSchemaVersion: CREDENTIAL_SCHEMA_VERSION,
    });
    await this.db
      .prepare(
        `INSERT INTO custom_provider_secrets (provider_id, encrypted_payload, schema_version, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT (provider_id) DO UPDATE SET
           encrypted_payload = excluded.encrypted_payload,
           schema_version = excluded.schema_version,
           updated_at = excluded.updated_at`
      )
      .bind(providerId, encrypted, CREDENTIAL_SCHEMA_VERSION, now)
      .run();
  }

  private modelRecord(provider: CustomProviderRecord, row: ModelRow): CustomModelRecord {
    return {
      id: `${provider.providerKey}/${row.model_id}`,
      providerId: provider.id,
      providerName: provider.name,
      protocol: provider.protocol,
      modelId: row.model_id,
      displayName: row.display_name,
      modalities: parseJsonArray(row.modalities) as CustomModelModality[],
      reasoningEfforts: parseJsonArray(row.reasoning_efforts) as ReasoningEffort[],
      contextWindowTokens: row.context_window_tokens,
      maxOutputTokens: row.max_output_tokens,
      enabled: row.enabled === 1,
    };
  }
}

/** The env var a provider's decrypted API key lands in: `CP_{ID8}_API_KEY`. */
export function customProviderApiKeyEnv(providerId: string): string {
  if (!CUSTOM_PROVIDER_ID_PATTERN.test(providerId)) {
    throw new Error(`Invalid custom provider id: ${providerId}`);
  }
  return `CP_${providerId.slice(0, 8).toUpperCase()}_API_KEY`;
}
