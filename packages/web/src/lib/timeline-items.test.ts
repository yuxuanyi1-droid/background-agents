import { describe, expect, it } from "vitest";
import type { SandboxEvent } from "@/types/session";
import { buildTimelineItems } from "./timeline-items";

function thinking(
  messageId: string,
  content: string,
  timestamp: number,
  partId?: string
): SandboxEvent {
  return {
    type: "thinking",
    content,
    messageId,
    sandboxId: "sandbox",
    timestamp,
    ...(partId ? { partId } : {}),
  };
}

function compaction(messageId: string, timestamp: number): SandboxEvent {
  return { type: "context_compacted", messageId, sandboxId: "sandbox", timestamp };
}

function toolCall(timestamp: number): SandboxEvent {
  return {
    type: "tool_call",
    tool: "Read",
    args: {},
    callId: `call-${timestamp}`,
    messageId: "msg-1",
    sandboxId: "sandbox",
    timestamp,
  };
}

function thinkingItems(events: SandboxEvent[]): SandboxEvent[] {
  return buildTimelineItems(events).flatMap((item) =>
    item.type === "single" && item.event.type === "thinking" ? [item.event] : []
  );
}

describe("reasoning dedupe", () => {
  it("drops a restreamed segment's stale copy instead of duplicating it", () => {
    // A reconnect mid-turn replays the stored partial segment; the completed
    // stream that follows carries the same cumulative text and supersedes it,
    // landing at its arrival position like the final token does.
    const events = [
      thinking("msg-1", "partial reasoning", 1),
      toolCall(2),
      thinking("msg-1", "complete reasoning", 3),
    ];

    expect(thinkingItems(events)).toEqual([thinking("msg-1", "complete reasoning", 3)]);
    expect(buildTimelineItems(events).map((item) => item.type)).toEqual(["tool_group", "single"]);
  });

  it("keeps distinct reasoning parts as separate blocks", () => {
    const first = thinking("msg-1", "part one", 1, "part-1");
    const second = thinking("msg-1", "part two", 2, "part-2");

    expect(thinkingItems([first, second])).toEqual([first, second]);
  });

  it("drops only the restreamed part's stale copy", () => {
    const events = [
      thinking("msg-1", "part one partial", 1, "part-1"),
      thinking("msg-1", "part two", 2, "part-2"),
      thinking("msg-1", "part one full", 3, "part-1"),
    ];

    expect(thinkingItems(events)).toEqual([
      thinking("msg-1", "part two", 2, "part-2"),
      thinking("msg-1", "part one full", 3, "part-1"),
    ]);
  });

  it("keeps reasoning from both sides of a compaction boundary", () => {
    const before = thinking("msg-1", "before compaction", 1);
    const after = thinking("msg-1", "after compaction", 3);

    expect(thinkingItems([before, compaction("msg-1", 2), after])).toEqual([before, after]);
  });
});
