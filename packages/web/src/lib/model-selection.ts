import {
  DEFAULT_MODEL,
  customModelReasoningConfig,
  getReasoningConfig,
  getValidModelOrDefault,
  resolveEnabledModel,
  type ModelCategory,
  type ReasoningEffort,
} from "@open-inspect/shared/models";

export interface ModelPreference {
  model: string;
  reasoningEffort?: string;
}

export interface ResolvedModelPreference {
  model: string;
  reasoningEffort?: ReasoningEffort;
}

function customReasoningConfigFor(
  model: string,
  items?: readonly ModelCategory[]
): ReturnType<typeof customModelReasoningConfig> {
  const efforts = items
    ?.flatMap((group) => group.models)
    .find(({ id }) => id === model)?.reasoningEfforts;
  return customModelReasoningConfig(efforts ?? []);
}

export function resolveModelPreference(
  preference: ModelPreference,
  enabledModels?: readonly string[],
  enabledModelOptions?: readonly ModelCategory[]
): ResolvedModelPreference {
  const model = enabledModels
    ? resolveEnabledModel({
        model: preference.model,
        enabledModels,
        fallbackModel: DEFAULT_MODEL,
      })
    : getValidModelOrDefault(preference.model);
  const reasoningConfig =
    getReasoningConfig(model) ?? customReasoningConfigFor(model, enabledModelOptions);
  return {
    model,
    reasoningEffort:
      preference.reasoningEffort === undefined
        ? undefined
        : (reasoningConfig?.efforts.find((effort) => effort === preference.reasoningEffort) ??
          reasoningConfig?.default),
  };
}

/** Registry-listed efforts for a custom model, for effort validation. */
export function customModelEfforts(
  model: string,
  items?: readonly ModelCategory[]
): readonly ReasoningEffort[] | undefined {
  return customReasoningConfigFor(model, items)?.efforts;
}

/** The effort a model should start at when freshly selected. */
export function defaultReasoningEffort(
  model: string,
  items?: readonly ModelCategory[]
): ReasoningEffort | undefined {
  return getReasoningConfig(model)?.default ?? customReasoningConfigFor(model, items)?.default;
}
