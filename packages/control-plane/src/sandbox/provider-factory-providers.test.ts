import { describe, expect, it } from "vitest";
import type { Env } from "../types";
import { resolveConfiguredSandboxProviders } from "./provider-factory";

function envWith(overrides: Partial<Env>): Env {
  return {
    SANDBOX_PROVIDER: "e2b",
    E2B_API_KEY: "e2b-key",
    E2B_TEMPLATE_ID: "tpl",
    ...overrides,
  } as Env;
}

describe("resolveConfiguredSandboxProviders", () => {
  it("lists exactly the providers whose credentials are complete", () => {
    const providers = resolveConfiguredSandboxProviders(
      envWith({
        MODAL_API_SECRET: "ak-secret",
        MODAL_WORKSPACE: "ws",
        DAYTONA_API_KEY: "dtn-key",
        DAYTONA_API_URL: "https://app.daytona.io/api",
        DAYTONA_BASE_SNAPSHOT: "open-inspect-sandbox",
      })
    );
    expect(providers).toEqual([
      { name: "modal", label: "Modal" },
      { name: "daytona", label: "Daytona" },
      { name: "e2b", label: "E2B" },
    ]);
  });

  it("drops a provider whose credentials are half configured", () => {
    // Modal without a workspace cannot construct; daytona without a snapshot
    // cannot create sandboxes; only e2b remains.
    const providers = resolveConfiguredSandboxProviders(
      envWith({ MODAL_API_SECRET: "ak-secret", DAYTONA_API_KEY: "dtn-key" })
    );
    expect(providers).toEqual([{ name: "e2b", label: "E2B" }]);
  });

  it("never lists modal-vm, which is an infra variant of modal", () => {
    const providers = resolveConfiguredSandboxProviders(
      envWith({ MODAL_API_SECRET: "ak-secret", MODAL_WORKSPACE: "ws" })
    );
    expect(providers.map((option) => option.name)).not.toContain("modal-vm");
  });
});
