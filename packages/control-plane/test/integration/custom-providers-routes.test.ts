/**
 * Custom provider routes over the real app + D1: protocol edits (including
 * the re-keying that crossing the Anthropic/OpenAI family boundary causes),
 * imported-model deletion, and the on-demand connection test's outcomes.
 *
 * The connection tests point at the reserved `.invalid` domain, which can
 * never resolve: the gateway is unreachable, so both test modes exercise
 * their failure rendering without any outbound dependency.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { env } from "cloudflare:test";
import { cleanD1Tables } from "./cleanup";
import { serviceFetch } from "./helpers";

const BASE = "https://test.local";

interface ProviderRecordJson {
  id: string;
  protocol: string;
  providerKey: string;
}

async function createProvider(
  overrides?: Partial<{ protocol: string; baseUrl: string }>
): Promise<ProviderRecordJson> {
  const response = await serviceFetch(`${BASE}/custom-providers`, {
    method: "POST",
    body: JSON.stringify({
      name: "Gateway",
      protocol: "openai_compatible",
      baseUrl: "https://example.invalid/v1",
      apiKey: "sk-test",
      ...overrides,
    }),
  });
  expect(response.status).toBe(201);
  return ((await response.json()) as { provider: ProviderRecordJson }).provider;
}

async function patchProvider(
  id: string,
  body: Record<string, unknown>
): Promise<{ status: number; provider?: ProviderRecordJson }> {
  const response = await serviceFetch(`${BASE}/custom-providers/${id}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
  const parsed = (await response.json().catch(() => null)) as {
    provider?: ProviderRecordJson;
  } | null;
  return { status: response.status, provider: parsed?.provider };
}

async function importModel(providerId: string, modelId = "gate-1"): Promise<void> {
  const response = await serviceFetch(`${BASE}/custom-providers/${providerId}/models`, {
    method: "PUT",
    body: JSON.stringify({
      models: [
        {
          modelId,
          displayName: "Gateway One",
          modalities: ["text"],
          reasoningEfforts: [],
          contextWindowTokens: 8_192,
          maxOutputTokens: 1_024,
        },
      ],
    }),
  });
  expect(response.status).toBe(200);
}

async function listModelIds(providerId: string): Promise<string[]> {
  const response = await serviceFetch(`${BASE}/custom-providers/${providerId}/models`);
  expect(response.status).toBe(200);
  return ((await response.json()) as { models: { id: string }[] }).models.map((model) => model.id);
}

interface ConnectionTestJson {
  ok: boolean;
  mode: string;
  latencyMs: number;
  detail: string;
}

async function runConnectionTest(
  providerId: string,
  body: string
): Promise<{ status: number; result?: ConnectionTestJson }> {
  const response = await serviceFetch(`${BASE}/custom-providers/${providerId}/test-connection`, {
    method: "POST",
    body,
  });
  const parsed = (await response.json().catch(() => null)) as ConnectionTestJson | null;
  return { status: response.status, result: parsed ?? undefined };
}

describe("Custom provider routes", () => {
  beforeEach(cleanD1Tables);

  describe("PATCH /custom-providers/:id (protocol)", () => {
    it("keeps the provider key when switching between the two OpenAI protocols", async () => {
      const provider = await createProvider();
      const { status, provider: updated } = await patchProvider(provider.id, {
        protocol: "openai_responses",
      });
      expect(status).toBe(200);
      expect(updated?.protocol).toBe("openai_responses");
      expect(updated?.providerKey).toBe(provider.providerKey);
    });

    it("re-keys the provider and its imported model IDs across the family boundary", async () => {
      const provider = await createProvider();
      await importModel(provider.id);
      expect(await listModelIds(provider.id)).toEqual([`${provider.providerKey}/gate-1`]);

      const { status, provider: updated } = await patchProvider(provider.id, {
        protocol: "anthropic",
      });
      expect(status).toBe(200);
      expect(provider.providerKey).toBe(`cpo-${provider.id.slice(0, 8)}`);
      expect(updated?.providerKey).toBe(`cpa-${provider.id.slice(0, 8)}`);
      expect(await listModelIds(provider.id)).toEqual([`${updated?.providerKey}/gate-1`]);
    });

    it("rejects an unknown protocol", async () => {
      const provider = await createProvider();
      const { status } = await patchProvider(provider.id, { protocol: "google_gemini" });
      expect(status).toBe(400);
    });
  });

  describe("DELETE /custom-providers/:id/models/:modelId", () => {
    it("removes an imported model and 404s on repeat", async () => {
      const provider = await createProvider();
      await importModel(provider.id, "gone-1");
      const path = `${BASE}/custom-providers/${provider.id}/models/${encodeURIComponent("gone-1")}`;

      const removed = await serviceFetch(path, { method: "DELETE" });
      expect(removed.status).toBe(200);
      expect(await listModelIds(provider.id)).toEqual([]);

      const repeat = await serviceFetch(path, { method: "DELETE" });
      expect(repeat.status).toBe(404);
    });
  });

  describe("POST /custom-providers/:id/test-connection", () => {
    it(
      "reports an unreachable gateway as a failed model-list test",
      { timeout: 30_000 },
      async () => {
        const provider = await createProvider();
        const { status, result } = await runConnectionTest(provider.id, "{}");
        expect(status).toBe(200);
        expect(result?.ok).toBe(false);
        expect(result?.mode).toBe("models");
        expect(result?.detail).toBeTruthy();
        expect(typeof result?.latencyMs).toBe("number");
      }
    );

    it(
      "tests a model through its protocol's generation endpoint",
      { timeout: 30_000 },
      async () => {
        const provider = await createProvider({ protocol: "anthropic" });
        const { status, result } = await runConnectionTest(
          provider.id,
          JSON.stringify({ modelId: "gate-1" })
        );
        expect(status).toBe(200);
        expect(result?.ok).toBe(false);
        expect(result?.mode).toBe("generation");
        expect(result?.detail).toBeTruthy();
      }
    );

    it("refuses the test when the provider has no stored API key", async () => {
      // The API always stores a key on create, so seed the keyless row directly.
      const id = `${"abcdef01"}${"0".repeat(24)}`;
      const now = Date.now();
      await env.DB.prepare(
        `INSERT INTO custom_providers
           (id, name, protocol, base_url, custom_headers, status, created_by, created_at, updated_at)
         VALUES (?, 'Keyless', 'anthropic', 'https://example.invalid/v1', '[]', 'active', NULL, ?, ?)`
      )
        .bind(id, now, now)
        .run();
      const { status } = await runConnectionTest(id, "{}");
      expect(status).toBe(409);
    });
  });
});
