import { describe, expect, it } from "vitest";
import {
  DEFAULT_HARNESS,
  HARNESS_CATALOG,
  HARNESS_IDS,
  checkHarnessCompatibility,
  filterModelsForHarness,
  getValidHarnessOrDefault,
  harnessSupportsCustomModel,
  harnessSupportsModel,
  harnessSupportsProviderAuth,
  isValidHarness,
  selectedProviderAuthModes,
  reconcileProviderSelectionsForHarness,
} from "./harnesses";
import { VALID_MODELS } from "./models";

describe("harness catalog", () => {
  it("lists only harnesses the runtime can boot, built-in first", () => {
    expect(HARNESS_IDS).toEqual(["opencode", "claude", "codex", "pi", "dsh", "zcode"]);
    expect(DEFAULT_HARNESS).toBe("opencode");
  });

  it("never brands the Claude harness as Claude Code", () => {
    expect(HARNESS_CATALOG.claude.label).toBe("Claude Agent");
  });

  it("resolves an absent or unknown harness to the default", () => {
    expect(getValidHarnessOrDefault(undefined)).toBe("opencode");
    expect(getValidHarnessOrDefault(null)).toBe("opencode");
    expect(getValidHarnessOrDefault("not-a-harness")).toBe("opencode");
    expect(getValidHarnessOrDefault("claude")).toBe("claude");
    expect(getValidHarnessOrDefault("codex")).toBe("codex");
    expect(isValidHarness("claude")).toBe(true);
    expect(isValidHarness("not-a-harness")).toBe(false);
    expect(isValidHarness(42)).toBe(false);
  });
});

describe("harnessSupportsModel", () => {
  it("lets OpenCode run every catalog model", () => {
    for (const model of VALID_MODELS) {
      expect(harnessSupportsModel("opencode", model)).toBe(true);
    }
  });

  it("restricts the Claude harness to Anthropic models", () => {
    expect(harnessSupportsModel("claude", "anthropic/claude-sonnet-4-6")).toBe(true);
    expect(harnessSupportsModel("claude", "claude-sonnet-4-6")).toBe(true);
    expect(harnessSupportsModel("claude", "openai/gpt-5.5")).toBe(false);
    expect(harnessSupportsModel("claude", "xai/grok-4.6")).toBe(false);
  });

  it("routes each vendor harness to the models its vendor serves", () => {
    expect(harnessSupportsModel("codex", "openai/gpt-5.5")).toBe(true);
    expect(harnessSupportsModel("codex", "anthropic/claude-sonnet-4-6")).toBe(false);
    expect(harnessSupportsModel("dsh", "deepseek/deepseek-v4-pro")).toBe(true);
    expect(harnessSupportsModel("dsh", "openai/gpt-5.5")).toBe(false);
    expect(harnessSupportsModel("zcode", "zai-coding-plan/glm-5.3")).toBe(true);
    expect(harnessSupportsModel("zcode", "openai/gpt-5.5")).toBe(false);
  });

  it("lets Pi run every catalog model", () => {
    for (const model of VALID_MODELS) {
      expect(harnessSupportsModel("pi", model)).toBe(true);
    }
  });

  it("runs Anthropic-protocol custom providers on Claude and rejects OpenAI-protocol ones", () => {
    expect(harnessSupportsModel("claude", "cpa-00112233/glm-4.7")).toBe(true);
    expect(harnessSupportsModel("claude", "cpo-00112233/deepseek-v4-pro")).toBe(false);
    expect(harnessSupportsModel("opencode", "cpa-00112233/glm-4.7")).toBe(true);
    expect(harnessSupportsModel("opencode", "cpo-00112233/deepseek-v4-pro")).toBe(true);
  });

  it("runs OpenAI-protocol custom providers on Codex; both wire protocols share one family", () => {
    expect(harnessSupportsModel("codex", "cpo-00112233/gpt-x")).toBe(true);
    expect(harnessSupportsModel("codex", "cpa-00112233/glm-4.7")).toBe(false);
    expect(harnessSupportsModel("codex", "anthropic/claude-sonnet-4-6")).toBe(false);
  });

  it("filters a model list by harness", () => {
    const filtered = filterModelsForHarness("claude", VALID_MODELS);
    expect(filtered.length).toBeGreaterThan(0);
    expect(filtered.every((model) => model.startsWith("anthropic/"))).toBe(true);
    expect(filterModelsForHarness("opencode", VALID_MODELS)).toEqual([...VALID_MODELS]);
  });
});

describe("harnessSupportsProviderAuth", () => {
  it("passes resolver-assigned legacy mode through on every harness", () => {
    expect(harnessSupportsProviderAuth("opencode", "anthropic", "legacy_scoped_oauth")).toBe(true);
    expect(harnessSupportsProviderAuth("claude", "anthropic", "legacy_scoped_oauth")).toBe(true);
    expect(harnessSupportsProviderAuth("claude", "openai", "legacy_scoped_oauth")).toBe(true);
  });

  it("only the Claude harness may select an Anthropic provider account", () => {
    expect(harnessSupportsProviderAuth("opencode", "anthropic", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("opencode", "anthropic", "api_key")).toBe(true);
    expect(harnessSupportsProviderAuth("claude", "anthropic", "provider_account")).toBe(true);
  });

  it("keeps OpenAI and xAI provider accounts on OpenCode", () => {
    expect(harnessSupportsProviderAuth("opencode", "openai", "provider_account")).toBe(true);
    expect(harnessSupportsProviderAuth("opencode", "xai", "provider_account")).toBe(true);
  });

  it("restricts the API-key-only harnesses to API-key auth", () => {
    expect(harnessSupportsProviderAuth("codex", "openai", "api_key")).toBe(true);
    expect(harnessSupportsProviderAuth("codex", "openai", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("dsh", "deepseek", "api_key")).toBe(true);
    expect(harnessSupportsProviderAuth("dsh", "deepseek", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("zcode", "zai-coding-plan", "api_key")).toBe(true);
    expect(harnessSupportsProviderAuth("zcode", "zai-coding-plan", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("pi", "deepseek", "api_key")).toBe(true);
    expect(harnessSupportsProviderAuth("pi", "google", "api_key")).toBe(false);
  });

  it("selects no auth mode for a provider the harness has no row for", () => {
    expect(harnessSupportsProviderAuth("claude", "openai", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("claude", "openai", "api_key")).toBe(false);
    expect(harnessSupportsProviderAuth("claude", "xai", "provider_account")).toBe(false);
    expect(harnessSupportsProviderAuth("opencode", "google", "api_key")).toBe(false);
  });
});

describe("reconcileProviderSelectionsForHarness", () => {
  const account = { mode: "provider_account" as const, accountId: "a".repeat(32) };

  it("drops a selection the harness runs the provider without", () => {
    expect(
      reconcileProviderSelectionsForHarness("opencode", { anthropic: account, openai: account })
    ).toEqual({ openai: account });
  });

  it("keeps selections for providers the harness does not run", () => {
    const selections = { openai: account, xai: { mode: "api_key" as const } };
    expect(reconcileProviderSelectionsForHarness("claude", selections)).toBe(selections);
  });

  it("returns the same object when every selection is usable", () => {
    const selections = { anthropic: account };
    expect(reconcileProviderSelectionsForHarness("claude", selections)).toBe(selections);
    expect(
      reconcileProviderSelectionsForHarness("opencode", { anthropic: { mode: "api_key" } })
    ).toEqual({ anthropic: { mode: "api_key" } });
  });
});

describe("selectedProviderAuthModes", () => {
  it("maps explicit selections to their modes and skips absent providers", () => {
    expect(
      selectedProviderAuthModes({
        openai: { mode: "provider_account", accountId: "0123456789abcdef0123456789abcdef" },
        xai: { mode: "api_key" },
      })
    ).toEqual({ openai: "provider_account", xai: "api_key" });
    expect(selectedProviderAuthModes({ openai: undefined })).toEqual({});
  });
});

describe("checkHarnessCompatibility", () => {
  it("accepts a compatible harness, model and auth", () => {
    expect(checkHarnessCompatibility("opencode", "anthropic/claude-sonnet-4-6")).toBeNull();
    expect(
      checkHarnessCompatibility("claude", "anthropic/claude-sonnet-4-6", {
        anthropic: "provider_account",
        openai: "provider_account",
      })
    ).toBeNull();
  });

  it("rejects a model the harness cannot run", () => {
    const result = checkHarnessCompatibility("claude", "openai/gpt-5.5");
    expect(result?.code).toBe("model");
    expect(result?.message).toContain("Claude Agent");
  });

  it("rejects an auth mode the harness cannot select for the model's provider", () => {
    const result = checkHarnessCompatibility("opencode", "anthropic/claude-sonnet-4-6", {
      anthropic: "provider_account",
    });
    expect(result?.code).toBe("provider_auth");
    expect(result?.message).toContain("API key");
  });

  it("ignores auth modes for providers the model does not use", () => {
    expect(
      checkHarnessCompatibility("opencode", "openai/gpt-5.5", {
        anthropic: "provider_account",
        openai: "api_key",
      })
    ).toBeNull();
  });
});

describe("harnessSupportsCustomModel", () => {
  it("restricts codex custom gateways to the Responses wire protocol", () => {
    expect(harnessSupportsCustomModel("codex", "cpo-99887766/glm-5.3", "openai_responses")).toBe(
      true
    );
    expect(harnessSupportsCustomModel("codex", "cpo-99887766/glm-5.3", "openai_compatible")).toBe(
      false
    );
    // Without registry metadata the family check alone decides.
    expect(harnessSupportsCustomModel("codex", "cpo-99887766/glm-5.3")).toBe(true);
    expect(harnessSupportsCustomModel("codex", "cpa-00112233/glm-5.3", "anthropic")).toBe(false);
  });

  it("runs every custom protocol on pi and dsh", () => {
    for (const harness of ["pi", "dsh"] as const) {
      for (const protocol of ["anthropic", "openai_compatible", "openai_responses"] as const) {
        expect(harnessSupportsCustomModel(harness, "cpo-99887766/glm-5.3", protocol)).toBe(true);
        expect(harnessSupportsCustomModel(harness, "cpa-00112233/glm-5.3", protocol)).toBe(true);
      }
    }
  });

  it("keeps zcode on its static family until its runtime support lands", () => {
    expect(harnessSupportsCustomModel("zcode", "cpo-99887766/glm-5.3", "openai_responses")).toBe(
      false
    );
  });
});
