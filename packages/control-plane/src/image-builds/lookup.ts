/**
 * Spawn-time lookup composition: binds the lifecycle manager's
 * ImageBuildLookup port to the image-build subsystem (scope resolver + store).
 * The Durable Object only calls this factory and injects the result.
 */

import { ImageBuildStore } from "../db/image-builds";
import type { ImageBuildLookup } from "../sandbox/lifecycle/image-selection";
import type { ImageBuildProvider } from "./model";
import { resolveScopeEnabled } from "./scope";
import type { SqlDatabase } from "../db/sql-database";

export function createImageBuildLookup(
  db: SqlDatabase,
  getProvider: () => ImageBuildProvider | null
): ImageBuildLookup {
  const store = new ImageBuildStore(db);
  return {
    getLatestReady: async (scope) => {
      // The session's backend is resolved on access — the graph is built
      // before init writes the session row that carries the choice.
      const provider = getProvider();
      if (!provider) return null;
      // Enablement (and entity existence) is the scope resolver's answer;
      // the store read is a plain row lookup.
      if (!(await resolveScopeEnabled(db, scope))) return null;
      return store.getLatestReadyForSpawn(scope, provider);
    },
    markRestoreFailed: (imageBuildId, error) => store.markRestoreFailed(imageBuildId, error),
  };
}
