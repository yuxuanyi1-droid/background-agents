import { customProviderSettingsProxy, validCustomProviderId } from "@/lib/custom-provider-proxy";

type Params = { id: string; modelId: string };
const { PATCH, DELETE } = customProviderSettingsProxy<Params>(
  ({ id, modelId }) =>
    `/custom-providers/${encodeURIComponent(id)}/models/${encodeURIComponent(modelId)}`,
  "custom provider model",
  ({ id, modelId }) => validCustomProviderId(id) && modelId.length > 0 && modelId.length <= 200
);

export { DELETE, PATCH };
