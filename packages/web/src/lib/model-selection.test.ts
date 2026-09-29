import { describe, it, expect } from "vitest";
import { DEFAULT_MODEL, getDefaultReasoningEffort } from "@open-inspect/shared/models";
import { defaultReasoningEffort, resolveModelPreference } from "./model-selection";
import type { ModelCategory } from "@open-inspect/shared/models";

describe("resolveModelPreference", () => {
  it("keeps a valid model and reasoning effort", () => {
    expect(
      resolveModelPreference({ model: "anthropic/claude-opus-4-8", reasoningEffort: "high" }, [
        "anthropic/claude-opus-4-8",
      ])
    ).toEqual({ model: "anthropic/claude-opus-4-8", reasoningEffort: "high" });
  });

  it("normalizes the model before validating reasoning effort", () => {
    expect(
      resolveModelPreference({ model: "claude-opus-4-8", reasoningEffort: "high" }, [
        "anthropic/claude-opus-4-8",
      ])
    ).toEqual({ model: "anthropic/claude-opus-4-8", reasoningEffort: "high" });
  });

  it("preserves the upstream model while enabled models are loading", () => {
    expect(
      resolveModelPreference({ model: "claude-opus-4-8", reasoningEffort: "high" }, undefined)
    ).toEqual({ model: "anthropic/claude-opus-4-8", reasoningEffort: "high" });
  });

  it("preserves an omitted effort so the model default remains selectable", () => {
    expect(resolveModelPreference({ model: "openai/gpt-5.6-sol" }, ["openai/gpt-5.6-sol"])).toEqual(
      { model: "openai/gpt-5.6-sol", reasoningEffort: undefined }
    );
  });

  it("uses the default when the loaded enabled-model list is empty", () => {
    expect(
      resolveModelPreference({ model: "anthropic/claude-opus-4-8", reasoningEffort: "high" }, [])
    ).toEqual({
      model: DEFAULT_MODEL,
      reasoningEffort: getDefaultReasoningEffort(DEFAULT_MODEL),
    });
  });

  it("uses the fallback model default when reasoning is invalid", () => {
    expect(
      resolveModelPreference({ model: "anthropic/claude-opus-4-8", reasoningEffort: "not-valid" }, [
        DEFAULT_MODEL,
      ])
    ).toEqual({
      model: DEFAULT_MODEL,
      reasoningEffort: getDefaultReasoningEffort(DEFAULT_MODEL),
    });
  });

  it("uses the selected model default when only reasoning is invalid", () => {
    const model = "anthropic/claude-opus-4-8";
    expect(resolveModelPreference({ model, reasoningEffort: "not-valid" }, [model])).toEqual({
      model,
      reasoningEffort: getDefaultReasoningEffort(model),
    });
  });

  it("omits reasoning for models without reasoning controls", () => {
    expect(
      resolveModelPreference({ model: "opencode/kimi-k2.5", reasoningEffort: "high" }, [
        "opencode/kimi-k2.5",
      ])
    ).toEqual({ model: "opencode/kimi-k2.5", reasoningEffort: undefined });
  });
  const customOptions: ModelCategory[] = [
    {
      category: "zhipu (custom)",
      models: [
        {
          id: "cpa-0799807a/glm-5.3",
          name: "GLM 5.3",
          description: "",
          reasoningEfforts: ["low", "medium", "high"],
        },
      ],
    },
  ];

  it("accepts a registry-listed effort for a custom-provider model", () => {
    expect(
      resolveModelPreference(
        { model: "cpa-0799807a/glm-5.3", reasoningEffort: "high" },
        ["cpa-0799807a/glm-5.3"],
        customOptions
      )
    ).toEqual({ model: "cpa-0799807a/glm-5.3", reasoningEffort: "high" });
  });

  it("falls back to the custom model default for an unlisted effort", () => {
    expect(
      resolveModelPreference(
        { model: "cpa-0799807a/glm-5.3", reasoningEffort: "max" },
        ["cpa-0799807a/glm-5.3"],
        customOptions
      )
    ).toEqual({ model: "cpa-0799807a/glm-5.3", reasoningEffort: "high" });
  });

  it("keeps an omitted custom effort undefined rather than defaulting", () => {
    expect(
      resolveModelPreference({ model: "cpa-0799807a/glm-5.3" }, ["cpa-0799807a/glm-5.3"], customOptions)
    ).toEqual({ model: "cpa-0799807a/glm-5.3", reasoningEffort: undefined });
  });

  it("derives a default effort for a custom model from its options entry", () => {
    expect(defaultReasoningEffort("cpa-0799807a/glm-5.3", customOptions)).toBe("high");
    expect(defaultReasoningEffort("cpa-0799807a/glm-5.3")).toBeUndefined();
    expect(defaultReasoningEffort("anthropic/claude-sonnet-4-5", customOptions)).toBe("max");
  });
});
