import { customProviderSettingsProxy, validCustomProviderId } from "@/lib/custom-provider-proxy";

type Params = { id: string };
export const { POST } = customProviderSettingsProxy<Params>(
  ({ id }) => `/custom-providers/${encodeURIComponent(id)}/test-connection`,
  "custom provider connection test",
  ({ id }) => validCustomProviderId(id)
);
