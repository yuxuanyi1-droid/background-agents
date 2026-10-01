import { describe, expect, it } from "vitest";
import type { SqlDatabase } from "../db/sql-database";
import type { Env } from "../types";
import { imageBuildProviderDepsFromEnv } from "./workflow";

const db = {} as SqlDatabase;

function envWith(overrides: Partial<Env>): Env {
  return {
    SANDBOX_PROVIDER: "e2b",
    E2B_API_KEY: "e2b-key",
    E2B_TEMPLATE_ID: "tpl",
    ...overrides,
  } as Env;
}

describe("imageBuildProviderDepsFromEnv", () => {
  it("lists every configured image-build provider, default first", () => {
    const deps = imageBuildProviderDepsFromEnv(
      envWith({
        MODAL_API_SECRET: "secret",
        MODAL_WORKSPACE: "ws",
        DAYTONA_API_KEY: "dtn",
        DAYTONA_API_URL: "https://app.daytona.io/api",
        DAYTONA_BASE_SNAPSHOT: "snap",
      }),
      db
    );
    expect(deps.map((entry) => entry.provider)).toEqual(["e2b", "modal", "daytona"]);
    expect(deps.every((entry) => entry.planner !== null)).toBe(true);
  });

  it("skips providers whose credentials cannot construct", () => {
    const deps = imageBuildProviderDepsFromEnv(
      envWith({ MODAL_API_SECRET: "secret" }), // modal without a workspace
      db
    );
    expect(deps.map((entry) => entry.provider)).toEqual(["e2b"]);
  });

  it("is empty when no image-build provider can construct", () => {
    const deps = imageBuildProviderDepsFromEnv(
      { SANDBOX_PROVIDER: "modal" } as Env, // no credentials anywhere
      db
    );
    expect(deps).toEqual([]);
  });
});
