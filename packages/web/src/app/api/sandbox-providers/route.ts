import { settingsProxy } from "@/lib/settings-proxy";

export const { GET } = settingsProxy(() => "/sandbox-providers", "sandbox providers");
