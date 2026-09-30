"use client";

import useSWR from "swr";
import { z } from "zod";
import { sandboxProviderNameSchema } from "@open-inspect/shared/types/integrations";
import { browserApiFetch } from "@/lib/browser-api-fetch";

const sandboxProvidersSchema = z.object({
  providers: z.array(z.object({ name: sandboxProviderNameSchema, label: z.string().min(1) })),
  default: sandboxProviderNameSchema,
});

export type SandboxProviderOption = z.infer<typeof sandboxProvidersSchema>["providers"][number];

/**
 * The sandbox backends this deployment can run sessions on. The form offers a
 * selector only when more than one is configured.
 */
export function useSandboxProviders() {
  const { data, error, isLoading } = useSWR(
    "/api/sandbox-providers",
    async (url: "/api/sandbox-providers") => {
      const response = await browserApiFetch(url);
      if (!response.ok) throw new Error("Failed to load sandbox providers");
      return sandboxProvidersSchema.parse(await response.json());
    },
    { revalidateOnFocus: false, dedupingInterval: 60_000 }
  );
  return {
    providers: data?.providers ?? [],
    defaultProvider: data?.default,
    isLoading,
    error,
  };
}
