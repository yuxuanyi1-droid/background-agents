/**
 * Validate reasoning effort against a model's allowed values.
 * Returns the validated effort string or null if invalid/absent.
 */

import { isValidReasoningEffort, type ReasoningEffort } from "@open-inspect/shared/models";
import type { Logger } from "../logger";

export function validateReasoningEffort(
  model: string,
  effort: string | undefined,
  log: Logger,
  /** Registry-listed efforts for a custom-provider model; no static config exists. */
  customEfforts?: readonly ReasoningEffort[]
): string | null {
  if (!effort) return null;
  if (isValidReasoningEffort(model, effort, customEfforts)) return effort;
  log.warn("Invalid reasoning effort for model, ignoring", {
    model,
    reasoning_effort: effort,
  });
  return null;
}
