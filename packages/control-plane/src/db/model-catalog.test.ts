import { describe, expect, it } from "vitest";
import { buildCatalogIndex, matchCatalogModel } from "./model-catalog";

function indexFrom(entries: Array<[string, { contextWindowTokens: number }]>) {
  return buildCatalogIndex(
    new Map(
      entries.map(([id, metadata]) => [
        id,
        {
          ...metadata,
          maxOutputTokens: null,
          inputModalities: [],
          outputModalities: [],
        },
      ])
    )
  );
}

describe("matchCatalogModel", () => {
  const index = indexFrom([
    ["anthropic/claude-sonnet-5.5", { contextWindowTokens: 1_000_000 }],
    ["openai/gpt-5.6-luna", { contextWindowTokens: 128_000 }],
    ["z-ai/glm-5.3", { contextWindowTokens: 200_000 }],
    ["fireworks/glm-5.3", { contextWindowTokens: 200_001 }],
    ["perceptron/perceptron-mk1.5", { contextWindowTokens: 32_000 }],
  ]);

  it("matches a bare gateway id against the catalog suffix", () => {
    expect(matchCatalogModel("claude-sonnet-5.5", index)?.contextWindowTokens).toBe(1_000_000);
    expect(matchCatalogModel("gpt-5.6-luna", index)?.contextWindowTokens).toBe(128_000);
  });

  it("matches a vendor-prefixed gateway id exactly", () => {
    expect(matchCatalogModel("anthropic/claude-sonnet-5.5", index)?.contextWindowTokens).toBe(
      1_000_000
    );
  });

  it("strips variant suffixes on both sides", () => {
    expect(matchCatalogModel("gpt-5.6-luna:free", index)?.contextWindowTokens).toBe(128_000);
    const withVariants = indexFrom([
      ["anthropic/claude-sonnet-5.5:batch", { contextWindowTokens: 1_000_000 }],
    ]);
    expect(matchCatalogModel("claude-sonnet-5.5", withVariants)?.contextWindowTokens).toBe(
      1_000_000
    );
  });

  it("normalizes case and surrounding whitespace", () => {
    expect(matchCatalogModel("  Claude-Sonnet-5.5 ", index)?.contextWindowTokens).toBe(1_000_000);
  });

  it("refuses to guess when two vendors share a bare id", () => {
    expect(matchCatalogModel("glm-5.3", index)).toBeNull();
  });

  it("prefers the full-id match over an ambiguous bare id", () => {
    expect(matchCatalogModel("z-ai/glm-5.3", index)?.contextWindowTokens).toBe(200_000);
  });

  it("returns null for unknown ids", () => {
    expect(matchCatalogModel("not-in-catalog", index)).toBeNull();
    expect(matchCatalogModel("anthropic/not-in-catalog", index)).toBeNull();
    expect(matchCatalogModel("", index)).toBeNull();
  });

  it("matches a catalog entry stored without a vendor prefix", () => {
    const bareCatalog = indexFrom([["perceptron-mk1.5", { contextWindowTokens: 32_000 }]]);
    expect(matchCatalogModel("perceptron-mk1.5", bareCatalog)?.contextWindowTokens).toBe(32_000);
  });

  it("keeps the base entry over later variants of the same model", () => {
    const withVariants = indexFrom([
      ["openai/gpt-5.6-luna", { contextWindowTokens: 128_000 }],
      ["openai/gpt-5.6-luna:free", { contextWindowTokens: 40_000 }],
      ["openai/gpt-5.6-luna:batch", { contextWindowTokens: 64_000 }],
    ]);
    expect(matchCatalogModel("gpt-5.6-luna", withVariants)?.contextWindowTokens).toBe(128_000);
    expect(matchCatalogModel("openai/gpt-5.6-luna", withVariants)?.contextWindowTokens).toBe(
      128_000
    );
  });
});
