import type { SandboxEvent } from "@/types/session";

/**
 * The displayable event log built from raw sandbox events.
 *
 * Token events carry the full accumulated text for one assistant segment (not
 * incremental deltas), so the log keeps one final token per segment. Context
 * compaction ends the current segment before another starts for the same
 * message. Thinking events are the same shape and are collapsed the same way;
 * they ride the live pipeline as a separate buffered stream so the reasoning
 * trail never mixes into the assistant answer text.
 */

export type AssistantTokenEvent = Extract<SandboxEvent, { type: "token" }>;
export type AssistantThinkingEvent = Extract<SandboxEvent, { type: "thinking" }>;

/**
 * The latest streamed assistant text for an in-flight segment. Only the most
 * recent token within that segment needs to be retained because it supersedes
 * the last.
 */
export type PendingAssistantText = Pick<
  AssistantTokenEvent,
  "content" | "messageId" | "sandboxId" | "timestamp"
>;

/**
 * The latest streamed reasoning text for one in-flight thinking segment.
 * Thinking events are cumulative per segment, so the most recent one
 * supersedes the last; when the stream switches segments (a reasoning part
 * ends, or a new message starts), the finished segment is flushed as its own
 * block instead of being overwritten so the whole trail survives.
 */
export type PendingAssistantThinking = Pick<
  AssistantThinkingEvent,
  "content" | "messageId" | "sandboxId" | "timestamp" | "partId"
>;

export function toUiSandboxEvent(event: SandboxEvent): SandboxEvent {
  return {
    ...event,
    timestamp: typeof event.timestamp === "number" ? event.timestamp : Date.now() / 1000,
  };
}

function isRenderableTokenEvent(event: SandboxEvent): event is AssistantTokenEvent {
  return event.type === "token" && Boolean(event.content) && Boolean(event.messageId);
}

/**
 * Replay should show one final token per compaction-delimited segment,
 * independent of tied storage ordering between token and completion.
 */
export function collapseReplayTokenEvents(events: SandboxEvent[]): SandboxEvent[] {
  const tokenBySegment = new Map<string, AssistantTokenEvent>();
  const segmentByMessageId = new Map<string, number>();
  const segmentKey = (messageId: string) =>
    JSON.stringify([messageId, segmentByMessageId.get(messageId) ?? 0]);

  for (const event of events) {
    if (isRenderableTokenEvent(event)) {
      tokenBySegment.set(segmentKey(event.messageId), event);
    } else if (event.type === "context_compacted") {
      segmentByMessageId.set(event.messageId, (segmentByMessageId.get(event.messageId) ?? 0) + 1);
    }
  }

  if (tokenBySegment.size === 0) {
    return events;
  }

  const result: SandboxEvent[] = [];
  const emittedSegments = new Set<string>();
  segmentByMessageId.clear();

  const emitSegmentToken = (messageId: string) => {
    const key = segmentKey(messageId);
    const token = tokenBySegment.get(key);
    if (token && !emittedSegments.has(key)) {
      result.push(token);
      emittedSegments.add(key);
    }
  };

  for (const evt of events) {
    if (isRenderableTokenEvent(evt)) {
      continue;
    }

    if (evt.type === "context_compacted") {
      emitSegmentToken(evt.messageId);
      result.push(evt);
      segmentByMessageId.set(evt.messageId, (segmentByMessageId.get(evt.messageId) ?? 0) + 1);
      continue;
    }

    if (evt.type === "execution_complete") {
      emitSegmentToken(evt.messageId);
    }

    result.push(evt);
  }

  for (const [key, token] of tokenBySegment) {
    if (!emittedSegments.has(key)) {
      result.push(token);
    }
  }

  return result;
}

export interface LiveEventIngestion {
  /** The pending assistant text after processing this event. */
  pendingText: PendingAssistantText | null;
  /** The pending reasoning text after processing this event. */
  pendingThinking: PendingAssistantThinking | null;
  /** Events ready to append to the visible event log. */
  append: SandboxEvent[];
}

/**
 * Step function for live sandbox events. Streamed token and thinking text is
 * buffered (not displayed) until its execution completes, at which point the
 * final text is emitted once with its original timestamp. All other events
 * pass through unchanged.
 */
export function ingestLiveSandboxEvent(
  pendingText: PendingAssistantText | null,
  pendingThinking: PendingAssistantThinking | null,
  event: SandboxEvent
): LiveEventIngestion {
  if (event.type === "token" && event.content && event.messageId) {
    return {
      pendingText: {
        content: event.content,
        messageId: event.messageId,
        sandboxId: event.sandboxId,
        timestamp: event.timestamp,
      },
      pendingThinking,
      append: [],
    };
  }

  if (event.type === "thinking" && event.content && event.messageId) {
    const next: PendingAssistantThinking = {
      content: event.content,
      messageId: event.messageId,
      sandboxId: event.sandboxId,
      timestamp: event.timestamp,
      ...(event.partId ? { partId: event.partId } : {}),
    };
    // A changed identity means the buffered segment is finished: flush it as
    // its own block so a new reasoning part does not overwrite the trail of
    // the previous one.
    if (
      pendingThinking &&
      (pendingThinking.messageId !== next.messageId ||
        (pendingThinking.partId ?? null) !== (next.partId ?? null))
    ) {
      return {
        pendingText,
        pendingThinking: next,
        append: [pendingToThinkingEvent(pendingThinking)],
      };
    }
    return { pendingText, pendingThinking: next, append: [] };
  }

  if (event.type === "execution_complete") {
    return {
      pendingText: null,
      pendingThinking: null,
      append: [
        ...(pendingThinking ? [pendingToThinkingEvent(pendingThinking)] : []),
        ...(pendingText ? [pendingToTokenEvent(pendingText)] : []),
        event,
      ],
    };
  }

  if (event.type === "context_compacted") {
    const compactedText = pendingText?.messageId === event.messageId ? pendingText : null;
    const compactedThinking =
      pendingThinking?.messageId === event.messageId ? pendingThinking : null;
    if (compactedText || compactedThinking) {
      return {
        pendingText: compactedText ? null : pendingText,
        pendingThinking: compactedThinking ? null : pendingThinking,
        append: [
          ...(compactedThinking ? [pendingToThinkingEvent(compactedThinking)] : []),
          ...(compactedText ? [pendingToTokenEvent(compactedText)] : []),
          event,
        ],
      };
    }
  }

  return { pendingText, pendingThinking, append: [event] };
}

export function pendingToTokenEvent(pending: PendingAssistantText): AssistantTokenEvent {
  return { type: "token", ...pending };
}

export function pendingToThinkingEvent(pending: PendingAssistantThinking): AssistantThinkingEvent {
  return { type: "thinking", ...pending };
}
