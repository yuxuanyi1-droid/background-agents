import { customProviderSettingsProxy, validCustomProviderId } from "@/lib/custom-provider-proxy";

type Params = { id: string };
const { GET, PATCH, DELETE } = customProviderSettingsProxy<Params>(
  ({ id }) => `/custom-providers/${encodeURIComponent(id)}`,
  "custom provider",
  ({ id }) => validCustomProviderId(id)
);

export { DELETE, GET, PATCH };
