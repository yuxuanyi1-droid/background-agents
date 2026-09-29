import { CUSTOM_PROVIDER_ID_PATTERN } from "@open-inspect/shared/types/custom-providers";
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { PRIVATE_NO_STORE_HEADERS } from "@/lib/control-plane-json-proxy";
import { settingsProxy } from "@/lib/settings-proxy";

export function validCustomProviderId(id: string): boolean {
  return CUSTOM_PROVIDER_ID_PATTERN.test(id);
}

function invalidParameter(): NextResponse {
  return NextResponse.json(
    { error: "Invalid custom provider parameter" },
    { status: 400, headers: PRIVATE_NO_STORE_HEADERS }
  );
}

/** A settings proxy that refuses malformed custom-provider path parameters. */
export function customProviderSettingsProxy<P>(
  buildPath: (params: P, request: NextRequest) => string,
  label: string,
  valid: (params: P) => boolean
) {
  const handlers = settingsProxy(buildPath, label);
  const handler =
    (method: keyof typeof handlers) =>
    async (request: NextRequest, context: { params: Promise<P> }) =>
      valid(await context.params) ? handlers[method](request, context) : invalidParameter();

  return {
    GET: handler("GET"),
    POST: handler("POST"),
    PATCH: handler("PATCH"),
    PUT: handler("PUT"),
    DELETE: handler("DELETE"),
  };
}
