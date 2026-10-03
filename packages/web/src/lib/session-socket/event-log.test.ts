import { describe, expect, it } from "vitest";
import type { SandboxEvent } from "@/types/session";
import {
  collapseReplayTokenEvents,
  ingestLiveSandboxEvent,
  pendingToThinkingEvent,
  pendingToTokenEvent,
  toUiSandboxEvent,
  type PendingAssistantText,
  type PendingAssistantThinking,
} from "./event-log";

function tokenEvent(messageId: string, content: string, timestamp = 1): SandboxEvent {
  return { type: "token", content, messageId, sandboxId: "sb-1", timestamp };
}

function thinkingEvent(messageId: string, content: string, timestamp = 1): SandboxEvent {
  return { type: "thinking", content, messageId, sandboxId: "sb-1", timestamp };
}

function partThinkingEvent(
  messageId: string,
  content: string,
  timestamp: number,
  partId: string
): Extract<SandboxEvent, { type: "thinking" }> {
  return { type: "thinking", content, messageId, sandboxId: "sb-1", timestamp, partId };
}

function completionEvent(messageId: string, timestamp = 2): SandboxEvent {
  return { type: "execution_complete", messageId, success: true, sandboxId: "sb-1", timestamp };
}

function compactionEvent(messageId: string, timestamp = 2): SandboxEvent {
  return { type: "context_compacted", messageId, sandboxId: "sb-1", timestamp };
}

describe("toUiSandboxEvent", () => {
  it("keeps a numeric timestamp", () => {
    expect(toUiSandboxEvent(tokenEvent("msg-1", "hi", 42)).timestamp).toBe(42);
  });

  it("fills a missing timestamp with the current time in seconds", () => {
    const event = { ...tokenEvent("msg-1", "hi"), timestamp: undefined } as unknown as SandboxEvent;
    const before = Date.now() / 1000;
    const result = toUiSandboxEvent(event);
    expect(result.timestamp).toBeGreaterThanOrEqual(before);
    expect(result.timestamp).toBeLessThanOrEqual(Date.now() / 1000);
  });
});

describe("collapseReplayTokenEvents", () => {
  it("returns events unchanged when there are no renderable tokens", () => {
    const events = [completionEvent("msg-1")];
    expect(collapseReplayTokenEvents(events)).toBe(events);
  });

  it("keeps only the final token per message, placed before its completion", () => {
    const events = [
      tokenEvent("msg-1", "partial", 1),
      tokenEvent("msg-1", "final", 2),
      completionEvent("msg-1", 3),
    ];
    expect(collapseReplayTokenEvents(events)).toEqual([
      tokenEvent("msg-1", "final", 2),
      completionEvent("msg-1", 3),
    ]);
  });

  it("shows only the last stored text part after tool calls in one turn", () => {
    const first = { ...tokenEvent("msg-1", "Let me check", 1), partId: "part-1" };
    const last = { ...tokenEvent("msg-1", "Here is the answer", 3), partId: "part-2" };
    const tool: SandboxEvent = {
      type: "tool_call",
      tool: "bash",
      args: {},
      callId: "call-1",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 2,
    };
    expect(collapseReplayTokenEvents([first, tool, last, completionEvent("msg-1", 4)])).toEqual([
      tool,
      last,
      completionEvent("msg-1", 4),
    ]);
  });

  it("keeps final tokens from both sides of a compaction boundary", () => {
    const events = [
      tokenEvent("msg-1", "pre-partial", 1),
      tokenEvent("msg-1", "before compaction", 2),
      compactionEvent("msg-1", 3),
      tokenEvent("msg-1", "post-partial", 4),
      tokenEvent("msg-1", "after compaction", 5),
      completionEvent("msg-1", 6),
    ];

    expect(collapseReplayTokenEvents(events)).toEqual([
      tokenEvent("msg-1", "before compaction", 2),
      compactionEvent("msg-1", 3),
      tokenEvent("msg-1", "after compaction", 5),
      completionEvent("msg-1", 6),
    ]);
  });

  it("moves a token ahead of its completion when storage ordering is tied", () => {
    const events = [completionEvent("msg-1", 2), tokenEvent("msg-1", "final", 1)];
    expect(collapseReplayTokenEvents(events)).toEqual([
      tokenEvent("msg-1", "final", 1),
      completionEvent("msg-1", 2),
    ]);
  });

  it("appends tokens whose completion never arrived", () => {
    const events = [tokenEvent("msg-1", "orphan"), completionEvent("msg-2")];
    expect(collapseReplayTokenEvents(events)).toEqual([
      completionEvent("msg-2"),
      tokenEvent("msg-1", "orphan"),
    ]);
  });

  it("ignores token events without content or messageId", () => {
    const empty = { ...tokenEvent("msg-1", ""), content: "" } as SandboxEvent;
    const events = [empty, completionEvent("msg-1")];
    expect(collapseReplayTokenEvents(events)).toBe(events);
  });

  it("leaves thinking events in place: each replay row is one already-collapsed segment", () => {
    const events = [
      thinkingEvent("msg-1", "before compaction", 1),
      compactionEvent("msg-1", 2),
      thinkingEvent("msg-1", "after compaction", 3),
      completionEvent("msg-1", 4),
    ];
    expect(collapseReplayTokenEvents(events)).toEqual(events);
  });
});

describe("ingestLiveSandboxEvent", () => {
  it("buffers token events without appending", () => {
    const result = ingestLiveSandboxEvent(null, null, tokenEvent("msg-1", "streaming", 5));
    expect(result.append).toEqual([]);
    expect(result.pendingText).toEqual({
      content: "streaming",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 5,
    });
  });

  it("replaces the pending text with the latest cumulative token", () => {
    const first = ingestLiveSandboxEvent(null, null, tokenEvent("msg-1", "he", 1));
    const second = ingestLiveSandboxEvent(
      first.pendingText,
      first.pendingThinking,
      tokenEvent("msg-1", "hello", 2)
    );
    expect(second.pendingText?.content).toBe("hello");
  });

  it("flushes pending text before the completion, keeping the token timestamp", () => {
    const pending: PendingAssistantText = {
      content: "final",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const result = ingestLiveSandboxEvent(pending, null, completionEvent("msg-1", 2));
    expect(result.pendingText).toBeNull();
    expect(result.append).toEqual([tokenEvent("msg-1", "final", 1), completionEvent("msg-1", 2)]);
  });

  it("appends a completion alone when nothing is pending", () => {
    const result = ingestLiveSandboxEvent(null, null, completionEvent("msg-1"));
    expect(result.append).toEqual([completionEvent("msg-1")]);
    expect(result.pendingText).toBeNull();
    expect(result.pendingThinking).toBeNull();
  });

  it("passes other events through without touching pending buffers", () => {
    const pending: PendingAssistantText = {
      content: "in flight",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const pendingThinking: PendingAssistantThinking = {
      content: "reasoning",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const toolCall: SandboxEvent = {
      type: "tool_call",
      tool: "bash",
      args: {},
      callId: "call-1",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 3,
    };
    const result = ingestLiveSandboxEvent(pending, pendingThinking, toolCall);
    expect(result.pendingText).toBe(pending);
    expect(result.pendingThinking).toBe(pendingThinking);
    expect(result.append).toEqual([toolCall]);
  });

  it("flushes pending text before context compaction", () => {
    const pending: PendingAssistantText = {
      content: "in flight",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const compaction: SandboxEvent = {
      type: "context_compacted",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 2,
    };

    const result = ingestLiveSandboxEvent(pending, null, compaction);

    expect(result.pendingText).toBeNull();
    expect(result.append).toEqual([pendingToTokenEvent(pending), compaction]);
  });

  it("buffers thinking events without appending", () => {
    const result = ingestLiveSandboxEvent(null, null, thinkingEvent("msg-1", "considering", 5));
    expect(result.append).toEqual([]);
    expect(result.pendingThinking).toEqual({
      content: "considering",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 5,
    });
    expect(result.pendingText).toBeNull();
  });

  it("replaces the pending thinking with the latest cumulative event", () => {
    const first = ingestLiveSandboxEvent(null, null, thinkingEvent("msg-1", "check", 1));
    const second = ingestLiveSandboxEvent(
      first.pendingText,
      first.pendingThinking,
      thinkingEvent("msg-1", "check the file", 2)
    );
    expect(second.pendingThinking?.content).toBe("check the file");
  });

  it("flushes the previous reasoning part when a new part starts", () => {
    const first = partThinkingEvent("msg-1", "part one", 1, "part-1");
    const buffered = ingestLiveSandboxEvent(null, null, first);
    const second = partThinkingEvent("msg-1", "part two", 2, "part-2");
    const result = ingestLiveSandboxEvent(null, buffered.pendingThinking, second);

    expect(result.append).toEqual([first]);
    expect(result.pendingThinking).toEqual({
      content: "part two",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 2,
      partId: "part-2",
    });
  });

  it("keeps accumulating while the reasoning part is unchanged", () => {
    const buffered = ingestLiveSandboxEvent(
      null,
      null,
      partThinkingEvent("msg-1", "check", 1, "part-1")
    );
    const result = ingestLiveSandboxEvent(
      null,
      buffered.pendingThinking,
      partThinkingEvent("msg-1", "check the file", 2, "part-1")
    );

    expect(result.append).toEqual([]);
    expect(result.pendingThinking?.content).toBe("check the file");
    expect(result.pendingThinking?.partId).toBe("part-1");
  });

  it("flushes the buffered trail when the message changes", () => {
    const buffered = ingestLiveSandboxEvent(null, null, thinkingEvent("msg-1", "first turn", 1));
    const result = ingestLiveSandboxEvent(
      null,
      buffered.pendingThinking,
      thinkingEvent("msg-2", "second turn", 2)
    );

    expect(result.append).toEqual([thinkingEvent("msg-1", "first turn", 1)]);
    expect(result.pendingThinking?.messageId).toBe("msg-2");
  });

  it("flushes the reasoning trail before the answer at completion", () => {
    const pendingThinking: PendingAssistantThinking = {
      content: "weighing options",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const pendingText: PendingAssistantText = {
      content: "final",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 2,
    };
    const result = ingestLiveSandboxEvent(
      pendingText,
      pendingThinking,
      completionEvent("msg-1", 3)
    );

    expect(result.pendingText).toBeNull();
    expect(result.pendingThinking).toBeNull();
    expect(result.append).toEqual([
      pendingToThinkingEvent(pendingThinking),
      pendingToTokenEvent(pendingText),
      completionEvent("msg-1", 3),
    ]);
  });

  it("flushes reasoning and text whose segments end at the same compaction", () => {
    const pendingThinking: PendingAssistantThinking = {
      content: "reasoning",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const pendingText: PendingAssistantText = {
      content: "in flight",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 2,
    };
    const compaction = compactionEvent("msg-1", 3);
    const result = ingestLiveSandboxEvent(pendingText, pendingThinking, compaction);

    expect(result.pendingText).toBeNull();
    expect(result.pendingThinking).toBeNull();
    expect(result.append).toEqual([
      pendingToThinkingEvent(pendingThinking),
      pendingToTokenEvent(pendingText),
      compaction,
    ]);
  });

  it("keeps a buffer whose message does not match the compaction", () => {
    const pendingThinking: PendingAssistantThinking = {
      content: "reasoning",
      messageId: "msg-1",
      sandboxId: "sb-1",
      timestamp: 1,
    };
    const result = ingestLiveSandboxEvent(null, pendingThinking, compactionEvent("msg-2", 3));

    expect(result.pendingThinking).toBe(pendingThinking);
    expect(result.append).toEqual([compactionEvent("msg-2", 3)]);
  });
});

describe("pendingToTokenEvent", () => {
  it("rebuilds a token event from pending text", () => {
    expect(
      pendingToTokenEvent({ content: "final", messageId: "msg-1", sandboxId: "sb-1", timestamp: 1 })
    ).toEqual(tokenEvent("msg-1", "final", 1));
  });
});

describe("pendingToThinkingEvent", () => {
  it("rebuilds a thinking event from pending reasoning", () => {
    expect(
      pendingToThinkingEvent({
        content: "reasoning",
        messageId: "msg-1",
        sandboxId: "sb-1",
        timestamp: 1,
      })
    ).toEqual(thinkingEvent("msg-1", "reasoning", 1));
  });

  it("carries the reasoning part id through the rebuild", () => {
    expect(
      pendingToThinkingEvent({
        content: "reasoning",
        messageId: "msg-1",
        sandboxId: "sb-1",
        timestamp: 1,
        partId: "part-1",
      })
    ).toEqual(partThinkingEvent("msg-1", "reasoning", 1, "part-1"));
  });
});
