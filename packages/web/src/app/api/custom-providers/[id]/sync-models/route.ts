import { customProviderSettingsProxy, validCustomProviderId } from "@/lib/custom-provider-proxy";

type Params = { id: string };
export const { POST } = customProviderSettingsProxy<Params>(
  ({ id }) => `/custom-providers/${encodeURIComponent(id)}/sync-models`,
  "custom provider model sync",
  ({ id }) => validCustomProviderId(id)
);
