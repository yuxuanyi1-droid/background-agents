// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useThinkingDisplay } from "./use-thinking-display";

afterEach(() => localStorage.clear());

describe("useThinkingDisplay", () => {
  it("starts from the full display when nothing is stored", () => {
    const { result } = renderHook(() => useThinkingDisplay());

    expect(result.current.thinkingDisplay).toBe("full");
  });

  it("restores and persists an explicit display mode", async () => {
    localStorage.setItem("session-timeline.thinking-display", "full");
    const { result } = renderHook(() => useThinkingDisplay());

    await waitFor(() => expect(result.current.thinkingDisplay).toBe("full"));
    act(() => result.current.setThinkingDisplay("hidden"));

    expect(localStorage.getItem("session-timeline.thinking-display")).toBe("hidden");
  });

  it("ignores a stored value that is not a display mode", () => {
    localStorage.setItem("session-timeline.thinking-display", "expanded");
    const { result } = renderHook(() => useThinkingDisplay());

    expect(result.current.thinkingDisplay).toBe("full");
  });
});
