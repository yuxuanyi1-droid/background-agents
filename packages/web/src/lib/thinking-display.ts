/**
 * How the model's reasoning trail renders in the timeline: `summary` keeps
 * each segment collapsed behind a one-line preview, `full` keeps the segments
 * expanded, and `hidden` drops them from the timeline entirely.
 */
export type ThinkingDisplay = "summary" | "full" | "hidden";

/**
 * The display used until the user chooses otherwise. Harnesses already emit
 * the most detailed reasoning they expose, so the default shows all of it.
 */
export const DEFAULT_THINKING_DISPLAY: ThinkingDisplay = "full";

/** The collapsed header shows at most this much of the segment's first line. */
export const THINKING_TEASER_MAX_CHARS = 120;

export function isThinkingDisplay(value: string | null): value is ThinkingDisplay {
  return value === "summary" || value === "full" || value === "hidden";
}

/**
 * The collapsed header's preview of one reasoning segment: its first
 * non-empty line, whitespace collapsed, truncated when it runs long.
 */
export function thinkingTeaser(content: string, maxChars = THINKING_TEASER_MAX_CHARS): string {
  const line = content.split("\n").find((candidate) => candidate.trim());
  if (!line) return "";

  const collapsed = line.trim().replace(/\s+/g, " ");
  return collapsed.length > maxChars ? `${collapsed.slice(0, maxChars).trimEnd()}…` : collapsed;
}
