import { settingsProxy } from "@/lib/settings-proxy";

export const { GET } = settingsProxy(() => "/custom-models", "custom models catalog");
