import { settingsProxy } from "@/lib/settings-proxy";

export const { GET, POST } = settingsProxy(() => "/custom-providers", "custom providers");
