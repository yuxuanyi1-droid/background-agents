import { customProviderSettingsProxy, validCustomProviderId } from "@/lib/custom-provider-proxy";

type Params = { id: string };
export const { GET, PUT } = customProviderSettingsProxy<Params>(
  ({ id }) => `/custom-providers/${encodeURIComponent(id)}/models`,
  "custom provider models",
  ({ id }) => validCustomProviderId(id)
);
