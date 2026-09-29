/**
 * Registry-listed reasoning efforts for a custom-provider model.
 *
 * Custom models have no static reasoning config, so every route that admits
 * a reasoning effort against a model resolves the registry here before the
 * shared `isValidReasoningEffort` check. Failures resolve to undefined so a
 * registry hiccup degrades to the static check, never to a hard failure.
 */

import { isCustomModelId } from "@open-inspect/shared/types/custom-providers";
import type { ReasoningEffort } from "@open-inspect/shared/models";
import { CustomProviderStore } from "../db/custom-providers";
import type { SqlDatabase } from "../db/sql-database";

export async function customModelReasoningEfforts(
  db: SqlDatabase,
  providerAccountsEncryptionKey: string | undefined,
  model: string
): Promise<readonly ReasoningEffort[] | undefined> {
  if (!isCustomModelId(model) || !providerAccountsEncryptionKey) return undefined;
  try {
    const resolved = await new CustomProviderStore(
      db,
      providerAccountsEncryptionKey
    ).resolveCustomModel(model);
    return resolved?.model.reasoningEfforts;
  } catch {
    return undefined;
  }
}
