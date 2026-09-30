/**
 * Custom model providers: deployment-level gateway definitions an
 * administrator registers (base URL, API key, protocol, custom headers),
 * plus the per-model metadata imported from the gateway's model list.
 *
 * Model IDs are namespaced by provider: `cpa-{8hex}/{upstream-model-id}`
 * for Anthropic-protocol gateways and `cpo-{8hex}/{upstream-model-id}` for
 * the two OpenAI wire protocols (chat completions and responses). The 8-hex
 * segment is the first 8 characters of the provider's canonical 32-hex ID,
 * so the provider a model routes to is recoverable from the model ID alone
 * without a catalog lookup.
 */

import { z } from "zod";

export const CUSTOM_PROVIDER_PROTOCOLS = [
  "anthropic",
  "openai_compatible",
  "openai_responses",
] as const;
export type CustomProviderProtocol = (typeof CUSTOM_PROVIDER_PROTOCOLS)[number];
export const customProviderProtocolSchema = z.enum(CUSTOM_PROVIDER_PROTOCOLS);

/**
 * UI label for each wire protocol, shared by the settings form and every
 * model picker. The two OpenAI protocols must read apart: Codex gateways run
 * Responses only, and collapsing them hides which protocol a model routes on.
 */
export const CUSTOM_PROVIDER_PROTOCOL_LABELS: Record<CustomProviderProtocol, string> = {
  anthropic: "Anthropic Messages API",
  openai_compatible: "OpenAI chat completions",
  openai_responses: "OpenAI Responses API",
};

/** Custom provider IDs use the installation's canonical 16-byte hex ID format. */
export const CUSTOM_PROVIDER_ID_PATTERN = /^[0-9a-f]{32}$/;
export const customProviderIdSchema = z.string().regex(CUSTOM_PROVIDER_ID_PATTERN);

/** Short provider key embedded in model IDs: `cpa`/`cpo` plus 8 hex chars.
 * Both OpenAI wire protocols share the `cpo` prefix: the protocol field (not
 * the key) tells chat completions and responses apart at routing time. */
export const CUSTOM_PROVIDER_KEY_PATTERN = /^cp[ao]-[0-9a-f]{8}$/;

export const CUSTOM_MODEL_MODALITIES = ["text", "image", "video", "audio"] as const;
export type CustomModelModality = (typeof CUSTOM_MODEL_MODALITIES)[number];
export const customModelModalitySchema = z.enum(CUSTOM_MODEL_MODALITIES);

export const CUSTOM_MODEL_REASONING_EFFORTS = [
  "none",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;
export const customModelReasoningEffortSchema = z.enum(CUSTOM_MODEL_REASONING_EFFORTS);

/** The stable 12-char provider key (`cpa-xxxxxxxx` / `cpo-xxxxxxxx`) for a provider ID. */
export function customProviderKey(providerId: string, protocol: CustomProviderProtocol): string {
  if (!CUSTOM_PROVIDER_ID_PATTERN.test(providerId)) {
    throw new Error(`Invalid custom provider id: ${providerId}`);
  }
  return `${protocol === "anthropic" ? "cpa" : "cpo"}-${providerId.slice(0, 8)}`;
}

/** Whether a model ID's provider segment is a custom provider key. */
export function isCustomProviderKey(value: string): boolean {
  return CUSTOM_PROVIDER_KEY_PATTERN.test(value);
}

/** Whether a full model ID routes to a custom provider. */
export function isCustomModelId(modelId: string): boolean {
  const slash = modelId.indexOf("/");
  if (slash <= 0) return false;
  return isCustomProviderKey(modelId.slice(0, slash));
}

/** Whether a model ID's custom provider speaks the Anthropic protocol. */
export function isCustomAnthropicModelId(modelId: string): boolean {
  const slash = modelId.indexOf("/");
  return slash > 0 && modelId.startsWith("cpa-", 0) && isCustomProviderKey(modelId.slice(0, slash));
}

const headerNameSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/, "Invalid HTTP header name");

export const customProviderHeaderSchema = z.strictObject({
  name: headerNameSchema,
  value: z.string().min(1).max(2048),
});
export type CustomProviderHeader = z.infer<typeof customProviderHeaderSchema>;

export const customProviderNameSchema = z.string().min(1).max(100);
export const customProviderBaseUrlSchema = z
  .string()
  .url()
  .max(2048)
  .refine((value) => value.startsWith("https://") || value.startsWith("http://"), {
    message: "Base URL must be an http(s) URL",
  });
export const customProviderApiKeySchema = z.string().min(1).max(4096);

export const createCustomProviderRequestSchema = z.strictObject({
  name: customProviderNameSchema,
  protocol: customProviderProtocolSchema,
  baseUrl: customProviderBaseUrlSchema,
  apiKey: customProviderApiKeySchema,
  headers: z.array(customProviderHeaderSchema).max(32).optional(),
});
export type CreateCustomProviderRequest = z.infer<typeof createCustomProviderRequestSchema>;

export const updateCustomProviderRequestSchema = z.strictObject({
  name: customProviderNameSchema.optional(),
  /**
   * Protocol is editable after creation. Switching across the Anthropic/OpenAI
   * boundary re-keys the provider (`cpa-…` ↔ `cpo-…`), which changes every
   * imported model ID; the two OpenAI protocols share a key and do not.
   */
  protocol: customProviderProtocolSchema.optional(),
  baseUrl: customProviderBaseUrlSchema.optional(),
  headers: z.array(customProviderHeaderSchema).max(32).optional(),
  status: z.enum(["active", "disabled"]).optional(),
  apiKey: customProviderApiKeySchema.optional(),
});
export type UpdateCustomProviderRequest = z.infer<typeof updateCustomProviderRequestSchema>;

export const customProviderStatusSchema = z.enum(["active", "disabled"]);

export const customProviderRecordSchema = z.strictObject({
  id: customProviderIdSchema,
  name: customProviderNameSchema,
  protocol: customProviderProtocolSchema,
  baseUrl: customProviderBaseUrlSchema,
  headers: z.array(customProviderHeaderSchema),
  status: customProviderStatusSchema,
  createdAt: z.number().int().nonnegative(),
  updatedAt: z.number().int().nonnegative(),
  createdBy: z.string().nullable(),
  /** The model-ID provider key (`cpa-xxxxxxxx`); present once models exist. */
  providerKey: z.string().regex(CUSTOM_PROVIDER_KEY_PATTERN),
});
export type CustomProviderRecord = z.infer<typeof customProviderRecordSchema>;

export const customProviderListResponseSchema = z.strictObject({
  providers: z.array(customProviderRecordSchema),
});

export const customProviderResponseSchema = z.strictObject({
  provider: customProviderRecordSchema,
});

/** Metadata matched from a public catalog (e.g. OpenRouter) for one synced model. */
export const modelCatalogMatchSchema = z.strictObject({
  contextWindowTokens: z.number().int().positive(),
  maxOutputTokens: z.number().int().positive().nullable(),
  inputModalities: z.array(customModelModalitySchema),
  outputModalities: z.array(customModelModalitySchema),
});
export type ModelCatalogMatch = z.infer<typeof modelCatalogMatchSchema>;

/** One entry of a gateway's synced model list; not persisted until imported. */
export const syncedCustomProviderModelSchema = z.strictObject({
  modelId: z.string().min(1).max(200),
  displayName: z.string().min(1).max(200),
  /** Present when the model matched a public catalog entry. */
  catalog: modelCatalogMatchSchema.optional(),
});
export type SyncedCustomProviderModel = z.infer<typeof syncedCustomProviderModelSchema>;

export const syncCustomProviderModelsResponseSchema = z.strictObject({
  models: z.array(syncedCustomProviderModelSchema).max(500),
});

/**
 * On-demand connectivity check against a saved provider. Without `modelId` the
 * check only lists models (auth + reachability); with it, the check issues a
 * one-token generation request so the model is proven usable end to end.
 */
export const customProviderConnectionTestRequestSchema = z.strictObject({
  modelId: z.string().min(1).max(200).optional(),
});
export type CustomProviderConnectionTestRequest = z.infer<
  typeof customProviderConnectionTestRequestSchema
>;

export const customProviderConnectionTestResultSchema = z.strictObject({
  ok: z.boolean(),
  /** `models` for a list reachability check, `generation` for a model test. */
  mode: z.enum(["models", "generation"]),
  latencyMs: z.number().int().nonnegative(),
  detail: z.string(),
});
export type CustomProviderConnectionTestResult = z.infer<
  typeof customProviderConnectionTestResultSchema
>;

export const importCustomProviderModelSchema = z.strictObject({
  modelId: z.string().min(1).max(200),
  displayName: z.string().min(1).max(200),
  modalities: z.array(customModelModalitySchema).min(1).max(4),
  reasoningEfforts: z.array(customModelReasoningEffortSchema).max(6),
  contextWindowTokens: z.number().int().positive().max(100_000_000),
  maxOutputTokens: z.number().int().positive().max(10_000_000),
});

export const importCustomProviderModelsRequestSchema = z.strictObject({
  models: z.array(importCustomProviderModelSchema).min(1).max(500),
});

export const updateCustomProviderModelRequestSchema = z.strictObject({
  displayName: z.string().min(1).max(200).optional(),
  modalities: z.array(customModelModalitySchema).min(1).max(4).optional(),
  reasoningEfforts: z.array(customModelReasoningEffortSchema).max(6).optional(),
  contextWindowTokens: z.number().int().positive().max(100_000_000).optional(),
  maxOutputTokens: z.number().int().positive().max(10_000_000).optional(),
  enabled: z.boolean().optional(),
});
export type UpdateCustomProviderModelRequest = z.infer<
  typeof updateCustomProviderModelRequestSchema
>;

/** A custom provider model as the web catalog serves it. */
export const customModelRecordSchema = z.strictObject({
  /** Full selectable model ID: `{providerKey}/{modelId}`. */
  id: z.string().min(1),
  providerId: customProviderIdSchema,
  providerName: customProviderNameSchema,
  protocol: customProviderProtocolSchema,
  modelId: z.string().min(1).max(200),
  displayName: z.string().min(1).max(200),
  modalities: z.array(customModelModalitySchema),
  reasoningEfforts: z.array(customModelReasoningEffortSchema),
  contextWindowTokens: z.number().int().positive(),
  maxOutputTokens: z.number().int().positive(),
  enabled: z.boolean(),
});
export type CustomModelRecord = z.infer<typeof customModelRecordSchema>;

export const customModelsCatalogResponseSchema = z.strictObject({
  models: z.array(customModelRecordSchema),
});

export const customProviderModelRecordSchema = customModelRecordSchema;
export const customProviderModelsResponseSchema = z.strictObject({
  models: z.array(
    customModelRecordSchema.extend({
      providerId: customProviderIdSchema,
    })
  ),
});
