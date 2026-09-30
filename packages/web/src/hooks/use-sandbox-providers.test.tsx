import { renderHook, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useSandboxProviders } from "./use-sandbox-providers";

vi.mock("@/lib/browser-api-fetch", () => ({ browserApiFetch: vi.fn() }));

import { browserApiFetch } from "@/lib/browser-api-fetch";

const fetchMock = vi.mocked(browserApiFetch);

describe("useSandboxProviders", () => {
  beforeEach(() => {
    fetchMock.mockReset();
  });

  it("parses the configured provider list and default", async () => {
    fetchMock.mockResolvedValue(
      Response.json({
        providers: [
          { name: "modal", label: "Modal" },
          { name: "daytona", label: "Daytona" },
          { name: "e2b", label: "E2B" },
        ],
        default: "e2b",
      })
    );
    const { result } = renderHook(() => useSandboxProviders(), {
      wrapper: ({ children }) => (
        <SWRConfig value={{ provider: () => new Map() }}>{children}</SWRConfig>
      ),
    });
    await waitFor(() => expect(result.current.providers).toHaveLength(3));
    expect(result.current.defaultProvider).toBe("e2b");
  });

  it("starts with no providers while loading or on failure", async () => {
    fetchMock.mockRejectedValue(new Error("offline"));
    const { result } = renderHook(() => useSandboxProviders(), {
      wrapper: ({ children }) => (
        <SWRConfig value={{ provider: () => new Map() }}>{children}</SWRConfig>
      ),
    });
    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.providers).toEqual([]);
  });
});
