/**
 * Sandbox-provider routes: what this deployment can run sessions on.
 */

import { Hono } from "hono";
import type { Env } from "../types";
import { resolveConfiguredSandboxProviders } from "../sandbox/provider-factory";
import { resolveSandboxBackendName } from "../sandbox/provider-name";
import { admit, dispatch } from "../routing/admit";
import type { ControlPlaneHonoEnv } from "../routing/hono-env";
import { GITHUB_USER_OR_SERVICE_ROUTE, json, requirePermission } from "./shared";

async function getSandboxProviders(_request: Request, env: Env): Promise<Response> {
  return json({
    providers: resolveConfiguredSandboxProviders(env),
    default: resolveSandboxBackendName(env.SANDBOX_PROVIDER),
  });
}

export const sandboxProviderRoutes = new Hono<ControlPlaneHonoEnv>();

sandboxProviderRoutes.get(
  "/sandbox-providers",
  admit({
    ...GITHUB_USER_OR_SERVICE_ROUTE,
    authorization: requirePermission("sessions.read"),
  }),
  (c) => dispatch(c, getSandboxProviders)
);
