"use client";

import { useCallback, useMemo, useState } from "react";
import useSWR from "swr";
import { z } from "zod";
import {
  MODEL_OPTIONS,
  DEFAULT_ENABLED_MODELS,
  applyModelPreferenceChanges,
  isValidModel,
  normalizeValidModels,
  type ModelCategory,
  type ModelPreferenceChange,
} from "@open-inspect/shared/models";
import { customModelRecordSchema } from "@open-inspect/shared/types/custom-providers";
import { browserApiFetch } from "@/lib/browser-api-fetch";

export const MODEL_PREFERENCES_KEY = "/api/model-preferences";
export const CUSTOM_MODELS_KEY = "/api/custom-models";
const INITIAL_MODEL_PREFERENCES_REVISION = 0;

const canonicalModelSchema = z.string().min(1);
const modelPreferencesSchema = z.object({
  enabledModels: z.array(canonicalModelSchema).nonempty(),
  revision: z.number().int().nonnegative(),
});
type ModelPreferencesResponse = z.infer<typeof modelPreferencesSchema>;

const customModelsSchema = z.object({ models: z.array(customModelRecordSchema) });
type CustomModelsResponse = z.infer<typeof customModelsSchema>;

function responseError(body: unknown): string | null {
  if (typeof body !== "object" || body === null || !("error" in body)) return null;
  return typeof body.error === "string" ? body.error : null;
}

export function useEnabledModels(): {
  enabledModels: string[];
  enabledModelOptions: ModelCategory[];
  /** Enabled custom-provider models, grouped per provider for the picker. */
  customModelOptions: ModelCategory[];
  loading: boolean;
  error: unknown;
  saving: boolean;
  updateModels: (changes: readonly ModelPreferenceChange[]) => Promise<void>;
} {
  const { data, error, isLoading, mutate } =
    useSWR<ModelPreferencesResponse>(MODEL_PREFERENCES_KEY);
  // Custom models are deployment data, not user preferences: fetch alongside
  // so every picker sees them without separate wiring.
  const { data: customData } = useSWR<CustomModelsResponse>(CUSTOM_MODELS_KEY);
  const [activeWrites, setActiveWrites] = useState(0);

  const customModels = useMemo(() => {
    const parsed = customModelsSchema.safeParse(customData ?? { models: [] });
    return parsed.success ? parsed.data.models.filter((model) => model.enabled) : [];
  }, [customData]);

  const customModelOptions = useMemo<ModelCategory[]>(() => {
    const groups = new Map<string, ModelCategory>();
    for (const model of customModels) {
      const category = `${model.providerName} (custom)`;
      const group = groups.get(category) ?? { category, models: [] };
      group.models.push({
        id: model.id,
        name: model.displayName,
        description: `${model.providerName} · ${model.protocol === "anthropic" ? "Anthropic" : "OpenAI-compatible"} protocol`,
      });
      groups.set(category, group);
    }
    return [...groups.values()];
  }, [customModels]);

  const enabledModels = useMemo<string[]>(() => {
    if (isLoading) return [];
    const normalized = normalizeValidModels(data?.enabledModels ?? []);
    const statics = normalized.length > 0 ? normalized : DEFAULT_ENABLED_MODELS;
    return [...statics, ...customModels.map((model) => model.id)];
  }, [data, isLoading, customModels]);

  const enabledModelOptions = useMemo(() => {
    const enabledSet = new Set(enabledModels);
    return [
      ...MODEL_OPTIONS.map((group) => ({
        ...group,
        models: group.models.filter((model) => enabledSet.has(model.id)),
      })).filter((group) => group.models.length > 0),
      ...customModelOptions,
    ];
  }, [enabledModels, customModelOptions]);

  const updateModels = useCallback(
    async (changes: readonly ModelPreferenceChange[]): Promise<void> => {
      if (isLoading || error) {
        throw new Error("Model preferences must load before saving");
      }

      const next = applyModelPreferenceChanges(
        enabledModels.filter((model) => isValidModel(model)),
        changes
      );
      if (next.length === 0) throw new Error("At least one model must be enabled");

      setActiveWrites((current) => current + 1);
      try {
        await mutate(
          async () => {
            const res = await browserApiFetch(MODEL_PREFERENCES_KEY, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ changes }),
            });
            const body: unknown = await res.json().catch(() => null);
            if (!res.ok) throw new Error(responseError(body) ?? "Failed to save preferences");
            const parsed = modelPreferencesSchema.safeParse(body);
            if (!parsed.success) throw new Error("Invalid model preferences response");
            return parsed.data;
          },
          {
            optimisticData: {
              enabledModels: next,
              revision: data?.revision ?? INITIAL_MODEL_PREFERENCES_REVISION,
            },
            rollbackOnError: true,
            populateCache: (result, current) =>
              !current || result.revision >= current.revision ? result : current,
            revalidate: false,
          }
        );
      } catch (requestError) {
        await mutate().catch(() => undefined);
        throw requestError;
      } finally {
        setActiveWrites((current) => current - 1);
      }
    },
    [data?.revision, enabledModels, error, isLoading, mutate]
  );

  return {
    enabledModels,
    enabledModelOptions,
    customModelOptions,
    loading: isLoading,
    error,
    saving: activeWrites > 0,
    updateModels,
  };
}
